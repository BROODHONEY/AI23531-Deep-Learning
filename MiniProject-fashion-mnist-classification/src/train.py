"""Training loop, augmentation and evaluation utilities."""
from __future__ import annotations

import copy
import random
import time
from dataclasses import asdict, dataclass

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score,
                             precision_recall_fscore_support)
from torch import nn

from .data import CLASS_NAMES
from .models import count_params


@dataclass
class TrainConfig:
    epochs: int = 20
    lr: float = 1e-3
    weight_decay: float = 0.0
    batch_size: int = 128
    scheduler: str = "none"        # "none" | "onecycle"
    label_smoothing: float = 0.0
    augment: bool = False          # random horizontal flip + random shift (pad 2)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def augment_batch(x: torch.Tensor, bg: float, pad: int = 2) -> torch.Tensor:
    """Random horizontal flip and random +/-`pad` pixel translation (per sample).

    `bg` is the normalised value of a black pixel, used to fill the border.
    """
    n = x.size(0)
    flip = torch.rand(n) < 0.5
    x = torch.where(flip[:, None, None, None], x.flip(3), x)
    xp = F.pad(x, (pad, pad, pad, pad), value=bg)
    ii = torch.randint(0, 2 * pad + 1, (n,)).tolist()
    jj = torch.randint(0, 2 * pad + 1, (n,)).tolist()
    return torch.stack([xp[k, :, i:i + 28, j:j + 28] for k, (i, j) in enumerate(zip(ii, jj))])


@torch.no_grad()
def predict(model: nn.Module, x: torch.Tensor, bs: int = 2000):
    """Return (logits, predictions)."""
    model.eval()
    logits = torch.cat([model(x[i:i + bs]) for i in range(0, len(x), bs)])
    return logits, logits.argmax(1)


def fit(model: nn.Module, data: dict, cfg: TrainConfig, seed: int, log=print) -> dict:
    """Train `model`, select the epoch with the best validation macro-F1, evaluate on test."""
    set_seed(seed)
    x_tr, y_tr = data["x_train"], data["y_train"]
    x_va, y_va, x_te, y_te = data["x_val"], data["y_val"], data["x_test"], data["y_test"]
    bg = -data["mean"] / data["std"]
    n = len(x_tr)
    steps_per_epoch = (n + cfg.batch_size - 1) // cfg.batch_size

    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    sched = None
    if cfg.scheduler == "onecycle":
        sched = torch.optim.lr_scheduler.OneCycleLR(
            opt, max_lr=cfg.lr, epochs=cfg.epochs, steps_per_epoch=steps_per_epoch, pct_start=0.2
        )

    hist = {k: [] for k in ("train_loss", "train_acc", "val_loss", "val_acc", "val_f1")}
    best = {"f1": -1.0, "epoch": -1, "state": None}
    t0 = time.perf_counter()
    for epoch in range(1, cfg.epochs + 1):
        model.train()
        perm = torch.randperm(n)
        tot_loss, tot_correct = 0.0, 0
        for s in range(0, n, cfg.batch_size):
            idx = perm[s:s + cfg.batch_size]
            xb, yb = x_tr[idx], y_tr[idx]
            if cfg.augment:
                xb = augment_batch(xb, bg)
            logits = model(xb)
            loss = F.cross_entropy(logits, yb, label_smoothing=cfg.label_smoothing)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            if sched is not None:
                sched.step()
            tot_loss += loss.item() * len(idx)
            tot_correct += (logits.argmax(1) == yb).sum().item()

        v_logits, v_pred = predict(model, x_va)
        v_loss = F.cross_entropy(v_logits, y_va).item()   # plain CE, comparable across models
        v_acc = accuracy_score(y_va, v_pred)
        v_f1 = f1_score(y_va, v_pred, average="macro")
        for k, v in zip(hist, (tot_loss / n, tot_correct / n, v_loss, v_acc, v_f1)):
            hist[k].append(float(v))
        if v_f1 > best["f1"]:
            best = {"f1": v_f1, "epoch": epoch, "state": copy.deepcopy(model.state_dict())}
        log(f"  epoch {epoch:>2}/{cfg.epochs}  train_loss {tot_loss / n:.4f}  "
            f"train_acc {tot_correct / n:.4f}  val_loss {v_loss:.4f}  "
            f"val_acc {v_acc:.4f}  val_f1 {v_f1:.4f}  [{time.perf_counter() - t0:.0f}s]")
    train_time = time.perf_counter() - t0

    model.load_state_dict(best["state"])
    t1 = time.perf_counter()
    logits, pred = predict(model, x_te)
    infer_ms = (time.perf_counter() - t1) / len(x_te) * 1000
    conf = logits.softmax(1).max(1).values
    y_true = y_te.numpy()
    p, r, f, sup = precision_recall_fscore_support(y_true, pred.numpy(), labels=range(10), zero_division=0)
    result = {
        "params": count_params(model),
        "config": asdict(cfg),
        "history": hist,
        "best_epoch": best["epoch"],
        "val_macro_f1": best["f1"],
        "test_accuracy": float(accuracy_score(y_true, pred.numpy())),
        "test_macro_f1": float(f1_score(y_true, pred.numpy(), average="macro")),
        "per_class": {
            CLASS_NAMES[i]: {"precision": float(p[i]), "recall": float(r[i]),
                             "f1": float(f[i]), "support": int(sup[i])}
            for i in range(10)
        },
        "confusion_matrix": confusion_matrix(y_true, pred.numpy(), labels=range(10)).tolist(),
        "train_time_s": train_time,
        "infer_ms_per_image": infer_ms,
    }
    return result, pred.numpy().astype(np.int8), conf.numpy().astype(np.float16), best["state"]
