"""Run the experiments.

    python -m src.run                       # all models, 3 seeds
    python -m src.run --models cnn resnet   # a subset
    python -m src.run --quick               # tiny smoke test (1 epoch, 1 seed, small subset)

Every (model, seed) run is saved to ``results/runs`` and skipped on re-run, so an
interrupted job can simply be restarted.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

from .data import get_data
from .models import MODEL_FACTORIES
from .train import TrainConfig, fit, set_seed

# Hyper-parameters (chosen once on the validation set / standard recipes; not tuned per seed).
CONFIGS = {
    "logreg":       TrainConfig(epochs=20, lr=1e-3, batch_size=256),
    "mlp":          TrainConfig(epochs=20, lr=1e-3, weight_decay=1e-4, batch_size=256),
    "cnn":          TrainConfig(epochs=10, lr=2e-3, weight_decay=5e-4, batch_size=128, scheduler="onecycle"),
    "resnet":       TrainConfig(epochs=12, lr=3e-3, weight_decay=5e-4, batch_size=128, scheduler="onecycle",
                                label_smoothing=0.1, augment=True),
    "resnet_noaug": TrainConfig(epochs=12, lr=3e-3, weight_decay=5e-4, batch_size=128, scheduler="onecycle",
                                label_smoothing=0.1, augment=False),
    "vit":          TrainConfig(epochs=15, lr=2e-3, weight_decay=0.05, batch_size=128, scheduler="onecycle",
                                label_smoothing=0.1, augment=True),
}


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", default=list(CONFIGS), choices=list(CONFIGS))
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--seed-start", type=int, default=0, help="first seed (lets several processes split the seeds)")
    ap.add_argument("--out", default="results")
    ap.add_argument("--data-root", default=None)
    ap.add_argument("--threads", type=int, default=None)
    ap.add_argument("--quick", action="store_true", help="smoke test: 1 epoch, 1 seed, 3000 train images")
    args = ap.parse_args(argv)

    if args.threads:
        torch.set_num_threads(args.threads)
    out = Path(args.out)
    (out / "runs").mkdir(parents=True, exist_ok=True)
    (out / "checkpoints").mkdir(parents=True, exist_ok=True)

    kw = {"root": args.data_root} if args.data_root else {}
    data = get_data(subset=3000 if args.quick else None, **kw)
    print(f"train {tuple(data['x_train'].shape)}  val {tuple(data['x_val'].shape)}  "
          f"test {tuple(data['x_test'].shape)}  mean {data['mean']:.4f}  std {data['std']:.4f}", flush=True)

    seeds = 1 if args.quick else args.seeds
    for name in args.models:
        cfg = CONFIGS[name]
        if args.quick:
            cfg = TrainConfig(**{**cfg.__dict__, "epochs": 1})
        for seed in range(args.seed_start, args.seed_start + seeds):
            tag = f"{name}_seed{seed}"
            path = out / "runs" / f"{tag}.json"
            if path.exists() and not args.quick:
                print(f"[skip] {tag} (already done)", flush=True)
                continue
            print(f"[run ] {tag}  config={cfg}", flush=True)
            set_seed(seed)
            model = MODEL_FACTORIES[name]()
            t = time.perf_counter()
            res, pred, conf, state = fit(model, data, cfg, seed, log=lambda s: print(s, flush=True))
            res.update(model=name, seed=seed)
            print(f"[done] {tag}  test_acc {res['test_accuracy']:.4f}  test_macro_f1 {res['test_macro_f1']:.4f}  "
                  f"best_epoch {res['best_epoch']}  params {res['params']:,}  "
                  f"{time.perf_counter() - t:.0f}s", flush=True)
            if args.quick:
                continue
            path.write_text(json.dumps(res))
            np.savez_compressed(out / "runs" / f"{tag}_preds.npz", pred=pred, conf=conf)
            if seed == 0:
                torch.save(state, out / "checkpoints" / f"{name}.pt")


if __name__ == "__main__":
    sys.exit(main())
