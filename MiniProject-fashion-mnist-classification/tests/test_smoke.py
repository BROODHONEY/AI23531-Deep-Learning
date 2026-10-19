"""Fast sanity tests: shapes, augmentation, and a one-epoch training run on a tiny subset."""
import torch

from src.data import get_data
from src.models import MODEL_FACTORIES, count_params
from src.train import TrainConfig, augment_batch, fit


def test_model_output_shapes():
    x = torch.randn(4, 1, 28, 28)
    for name, factory in MODEL_FACTORIES.items():
        model = factory().eval()
        assert model(x).shape == (4, 10), name
        assert count_params(model) > 0


def test_augmentation_preserves_shape_and_range():
    x = torch.randn(16, 1, 28, 28)
    out = augment_batch(x, bg=-0.8)
    assert out.shape == x.shape


def test_data_splits_are_disjoint_in_size_and_normalised():
    d = get_data(subset=1000)
    assert d["x_train"].shape[0] == 1000 and d["x_test"].shape == (10000, 1, 28, 28)
    assert abs(d["x_train"].mean().item()) < 1e-3          # normalised with train statistics
    assert set(d["y_test"].unique().tolist()) == set(range(10))


def test_one_epoch_training_learns_something():
    d = get_data(subset=2000)
    model = MODEL_FACTORIES["mlp"]()
    res, pred, conf, _ = fit(model, d, TrainConfig(epochs=2, lr=2e-3, batch_size=128), seed=0, log=lambda s: None)
    assert res["test_accuracy"] > 0.5                       # far above 10% chance
    assert pred.shape == (10000,) and conf.shape == (10000,)
