from pathlib import Path

from Visualization.prediction_plots import (
    plot_profile_predictions,
    plot_time_predictions,
    plot_predicted_vs_actual_profiles,
)
from Visualization.residual_plots import plot_residuals_profiles
from scripts.Visualization.metrics import compute_metrics
import random

import numpy as np
import torch


def set_reproducibility(seed: int = 42) -> None:
    """
    Configure random-number generators and PyTorch operations
    for reproducible model training.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    # Prevent cuDNN from benchmarking and selecting different
    # convolution algorithms between runs.
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

    # Raise an error if PyTorch encounters an operation for which
    # no deterministic implementation is available.
    torch.use_deterministic_algorithms(True)

def inverse_scale_sequence(
    data_scaled: np.ndarray,
    scaler,
) -> np.ndarray:
    """
    Inverse-transform a sequence array with shape (B, T, F).
    """
    batch_size, sequence_length, n_features = data_scaled.shape

    return scaler.inverse_transform(
        data_scaled.reshape(-1, n_features)
    ).reshape(batch_size, sequence_length, n_features)


def predict_sequence_model(
    model,
    x,
    y_scaler,
) -> np.ndarray:
    """
    Generate predictions and transform them back to physical units.

    Parameters
    ----------
    model:
        Trained PyTorch sequence model.
    x:
        Input tensor with shape (B, T, F).
    y_scaler:
        Scaler fitted to the target variables.

    Returns
    -------
    np.ndarray
        Predictions in physical units with shape (B, T, S).
    """
    model.eval()

    with torch.no_grad():
        pred_scaled = model(x).detach().cpu().numpy()

    return inverse_scale_sequence(
        data_scaled=pred_scaled,
        scaler=y_scaler,
    )


def evaluate_predictions(
    pred: np.ndarray,
    y_true_scaled: np.ndarray,
    mask: np.ndarray,
    y_scaler,
) -> dict[str, float]:
    """
    Evaluate predictions over valid, non-padded time steps.
    """
    y_true = inverse_scale_sequence(
        data_scaled=y_true_scaled,
        scaler=y_scaler,
    )

    mask = np.asarray(mask, dtype=bool)

    y_true_valid = y_true[mask]
    y_pred_valid = pred[mask]

    return compute_metrics(
        y_true_valid,
        y_pred_valid,
    )


def evaluate_sequence_model(
    model,
    x_test,
    y_test,
    mask_test,
    y_scaler,
) -> dict[str, float]:
    """
    Predict and evaluate a sequence model on a test dataset.
    """
    pred = predict_sequence_model(
        model=model,
        x=x_test,
        y_scaler=y_scaler,
    )

    return evaluate_predictions(
        pred=pred,
        y_true_scaled=y_test.detach().cpu().numpy(),
        mask=mask_test.detach().cpu().numpy(),
        y_scaler=y_scaler,
    )


def print_metrics(
    metrics: dict[str, float],
    prefix: str = "Final test metrics",
) -> None:
    print(
        f"{prefix}: "
        f"rmse={metrics['rmse']:.4f}, "
        f"mae={metrics['mae']:.4f}, "
        f"mse={metrics['mse']:.4f}, "
        f"max_abs={metrics['max_abs']:.4f}"
    )


def plot_sequence_model_results(
    model,
    x_test,
    mask_test,
    data,
    test_profiles,
    target_cols,
    y_scaler,
    show_threshold,
    ts_col: str = "Created",
) -> None:
    """
    Create prediction and residual plots for a sequence model.
    """
    pred = predict_sequence_model(
        model=model,
        x=x_test,
        y_scaler=y_scaler,
    )

    plot_profile_predictions(
        data=data,
        pred_c=pred,
        test_mask=mask_test,
        test_profiles=test_profiles,
        target_cols=target_cols,
        y_scaler=y_scaler,
    )

    plot_time_predictions(
        data=data,
        pred_c=pred,
        test_mask=mask_test,
        test_profiles=test_profiles,
        target_cols=target_cols,
        y_scaler=y_scaler,
        ts_col=ts_col,
    )

    plot_predicted_vs_actual_profiles(
        data=data,
        pred_c=pred,
        test_mask=mask_test,
        test_profiles=test_profiles,
        target_cols=target_cols,
        y_scaler=y_scaler,
        ts_col=ts_col,
    )

    plot_residuals_profiles(
        data=data,
        pred_c=pred,
        test_mask=mask_test,
        test_profiles=test_profiles,
        target_cols=target_cols,
        y_scaler=y_scaler,
        ts_col=ts_col,
    )


def save_sequence_model(
    model,
    x_scaler,
    y_scaler,
    input_cols,
    target_cols,
    model_type,
    model_config,
    training_config,
    dt_s,
    path,
    history=None,
) -> None:
    """
    Save a trained sequence model and its required metadata.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    state_dict = {
        name: value.detach().cpu()
        for name, value in model.state_dict().items()
    }

    checkpoint = {
        "model_state_dict": state_dict,
        "model_type": model_type,
        "model_class": model.__class__.__name__,
        "model_config": model_config,
        "training_config": training_config,
        "x_scaler": x_scaler,
        "y_scaler": y_scaler,
        "input_cols": list(input_cols),
        "target_cols": list(target_cols),
        "dt_s": dt_s,
    }

    if history is not None:
        checkpoint["history"] = history

    torch.save(checkpoint, path)

    print(f"{model_type} model saved to {path}")


from torch import nn
from typing import TypeAlias
ModelRegistry: TypeAlias = dict[str, type[nn.Module]]

def load_sequence_model(
    path: str | Path,
    model_registry: ModelRegistry,
    device: str | torch.device = "cpu",
    strict: bool = True,
) -> tuple[nn.Module, dict]:
    """
    Load a saved sequence model and its associated metadata.

    Parameters
    ----------
    path:
        Path to the saved PyTorch checkpoint.

    model_registry:
        Mapping from saved model identifiers to their corresponding
        PyTorch model classes. The loader first searches using
        ``model_type`` and then ``model_class``.

        Example
        -------
        {
            "TCN": TCNBaseline,
            "RNN": RNNBaseline,
            "TCNBaseline": TCNBaseline,
            "RNNBaseline": RNNBaseline,
        }

    device:
        Device on which the model should be loaded.

    strict:
        Whether the checkpoint state dictionary must exactly match
        the reconstructed model architecture.

    Returns
    -------
    model:
        Reconstructed model in evaluation mode.

    metadata:
        Dictionary containing the saved scalers, columns, model
        configuration, training configuration, sampling interval,
        training history, and other checkpoint information.

    Notes
    -----
    The checkpoint contains serialized scaler objects. Therefore,
    ``weights_only=False`` is required. Only load trusted checkpoint
    files.
    """
    path = Path(path)
    device = torch.device(device)

    if not path.exists():
        raise FileNotFoundError(f"Model checkpoint not found: {path}")

    checkpoint = torch.load(
        path,
        map_location=device,
        weights_only=False,
    )

    if not isinstance(checkpoint, dict):
        raise TypeError(
            "Expected the checkpoint to be a dictionary, "
            f"but received {type(checkpoint).__name__}."
        )

    required_keys = {
        "model_state_dict",
        "model_type",
        "model_class",
        "model_config",
        "x_scaler",
        "y_scaler",
        "input_cols",
        "target_cols",
        "dt_s",
    }

    missing_keys = required_keys.difference(checkpoint)

    if missing_keys:
        raise KeyError(
            "Checkpoint is missing required keys: "
            + ", ".join(sorted(missing_keys))
        )

    model_type = checkpoint["model_type"]
    model_class_name = checkpoint["model_class"]

    # Try the user-defined model type first, then the Python class name.
    model_class = model_registry.get(model_type)

    if model_class is None:
        model_class = model_registry.get(model_class_name)

    if model_class is None:
        available = ", ".join(sorted(model_registry))

        raise ValueError(
            f"No model class registered for model_type={model_type!r} "
            f"or model_class={model_class_name!r}. "
            f"Available registry entries: {available}"
        )

    model_config = checkpoint["model_config"]

    if not isinstance(model_config, dict):
        raise TypeError(
            "checkpoint['model_config'] must be a dictionary."
        )

    model = model_class(**model_config).to(device)

    model.load_state_dict(
        checkpoint["model_state_dict"],
        strict=strict,
    )

    model.eval()

    metadata = {
        key: value
        for key, value in checkpoint.items()
        if key != "model_state_dict"
    }

    metadata["path"] = path
    metadata["device"] = device

    print(
        f"{model_type} model loaded from {path} "
        f"on device {device}"
    )

    return model, metadata