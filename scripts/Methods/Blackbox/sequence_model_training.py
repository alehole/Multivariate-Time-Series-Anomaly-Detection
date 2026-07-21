import torch
from torch import Tensor
from torch.nn import Module

def masked_mse_loss_orig(y_hat, y_true, mask):
    """
    y_hat:  (B, T, F)
    y_true: (B, T, F)
    mask:   (B, T)
    """

    mask = mask.unsqueeze(-1)  # (B, T, 1)

    loss = (y_hat - y_true) ** 2
    loss = loss * mask

    return loss.sum() / mask.sum().clamp(min=1.0)




def masked_mse_loss(
    y_hat: Tensor,
    y_true: Tensor,
    mask: Tensor,
) -> Tensor:
    """
    Calculate mean squared error over valid sequence positions.

    Parameters
    ----------
    y_hat:
        Predicted values with shape (B, T, F).
    y_true:
        Target values with shape (B, T, F).
    mask:
        Validity mask with shape (B, T), where 1 indicates
        a valid time step and 0 indicates padding.

    Returns
    -------
    Tensor
        Scalar masked mean squared error.
    """
    if y_hat.shape != y_true.shape:
        raise ValueError(
            f"Prediction shape {y_hat.shape} does not match "
            f"target shape {y_true.shape}."
        )

    if mask.shape != y_true.shape[:2]:
        raise ValueError(
            f"Mask shape {mask.shape} does not match "
            f"batch and time dimensions {y_true.shape[:2]}."
        )

    # Convert from (B, T) to (B, T, 1).
    mask = mask.unsqueeze(-1).to(
        device=y_hat.device,
        dtype=y_hat.dtype,
    )

    # Expand the mask to include every output feature.
    mask = mask.expand_as(y_hat)

    squared_error = (y_hat - y_true).pow(2)
    masked_error = squared_error * mask

    return masked_error.sum() / mask.sum().clamp(min=1.0)


def train_sequence_model(
    model,
    x_train,
    y_train,
    mask_train,
    x_val=None,
    y_val=None,
    mask_val=None,
    *,
    n_epochs=200,
    lr=1e-3,
    weight_decay=1e-5,
    max_grad_norm=1.0,
):
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=lr,
        weight_decay=weight_decay,
    )

    best_val_loss = float("inf")
    best_state = None

    history = {
        "train_loss": [],
        "val_loss": [],
    }

    for epoch in range(1, n_epochs + 1):
        model.train()

        optimizer.zero_grad()

        y_hat = model(x_train)
        loss = masked_mse_loss(y_hat, y_train, mask_train)

        loss.backward()


        torch.nn.utils.clip_grad_norm_(
            model.parameters(),
            max_norm=max_grad_norm,
        )

        optimizer.step()

        train_loss = loss.item()
        history["train_loss"].append(train_loss)

        if x_val is not None:
            model.eval()
            with torch.no_grad():
                y_val_hat = model(x_val)
                val_loss = masked_mse_loss(y_val_hat, y_val, mask_val).item()

            history["val_loss"].append(val_loss)

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_state = {
                    k: v.detach().cpu().clone()
                    for k, v in model.state_dict().items()
                }

            print(
                f"Epoch {epoch:03d} | "
                f"train_loss={train_loss:.6f} | "
                f"val_loss={val_loss:.6f}"
            )

        else:
            print(
                f"Epoch {epoch:03d} | "
                f"train_loss={train_loss:.6f}"
            )

    if best_state is not None:
        model.load_state_dict(best_state)

    return model, history