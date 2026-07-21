import torch
import pandas as pd
from src.models.RNN.rnn_model import train_rnn_baseline
from src.preprocess.feature_engineering import ts_cols, feature_processing
from src.models.TCN.TCN import TCNBaseline
from src.models.profile_dataset import (
    train_val_test_split_profiles,
    scale_profile_data,
    tensorize_profiles,
    evaluate_predictions,
)
from src.config import paths
from pathlib import Path
from src.visualization.prediction_plots import (
    plot_profile_predictions,
    plot_time_predictions,
    plot_predicted_vs_actual_profiles,
)
from src.visualization.residual_plots import plot_residuals_profiles

def predict_rnn_test_set(model, x_test, y_scaler):
    model.eval()

    with torch.no_grad():
        pred_scaled = model(x_test).cpu().numpy()

    B, T, S = pred_scaled.shape

    pred_c = y_scaler.inverse_transform(
        pred_scaled.reshape(-1, S)
    ).reshape(B, T, S)

    return pred_c


def evaluate_rnn_model(
    model,
    x_test,
    y_test,
    mask_test,
    y_scaler,
):
    pred_c = predict_rnn_test_set(model, x_test, y_scaler)

    return evaluate_predictions(
        pred_c=pred_c,
        y_true_scaled=y_test.cpu().numpy(),
        mask=mask_test.cpu().numpy(),
        y_scaler=y_scaler,
    )
def eval_plot_rnn(
    model,
    x_test,
    mask_test,
    data,
    test_profiles,
    target_cols,
    y_scaler,
    ts_col="Created",
):
    pred_c = predict_rnn_test_set(
        model=model,
        x_test=x_test,
        y_scaler=y_scaler,
    )

    plot_profile_predictions(
        data=data,
        pred_c=pred_c,
        test_mask=mask_test,
        test_profiles=test_profiles,
        target_cols=target_cols,
        y_scaler=y_scaler,
    )

    plot_time_predictions(
        data=data,
        pred_c=pred_c,
        test_mask=mask_test,
        test_profiles=test_profiles,
        target_cols=target_cols,
        y_scaler=y_scaler,
        ts_col=ts_col,
    )

    plot_predicted_vs_actual_profiles(
        data=data,
        pred_c=pred_c,
        test_mask=mask_test,
        test_profiles=test_profiles,
        target_cols=target_cols,
        y_scaler=y_scaler,
        ts_col=ts_col,
    )

    plot_residuals_profiles(
        data=data,
        pred_c=pred_c,
        test_mask=mask_test,
        test_profiles=test_profiles,
        target_cols=target_cols,
        y_scaler=y_scaler,
        ts_col=ts_col,
    )

def save_rnn_model(
    model,
    x_scaler,
    y_scaler,
    input_cols,
    target_cols,
    model_type,
    hidden_size,
    num_layers,
    dropout,
    dt_s,
    path,
    history=None,
    params=None,
):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    save_dict = {
        "model_state_dict": model.state_dict(),
        "x_scaler": x_scaler,
        "y_scaler": y_scaler,
        "input_cols": input_cols,
        "target_cols": target_cols,
        "model_type": model_type,
        "hidden_size": hidden_size,
        "num_layers": num_layers,
        "dropout": dropout,
        "dt_s": dt_s,
    }

    if history is not None:
        save_dict["history"] = history

    if params is not None:
        save_dict["params"] = params

    torch.save(save_dict, path)
    print(f"RNN model saved to {path}")


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    TS_COL = "Created"
    WINDOW_STEPS = 300
    VAL_PROFILE_LEN = 3
    TEST_PROFILE_LEN = 3

    input_cols = [
        "AE_PS_EXH",
        "AE PORT END BRG.TEMP.",
        "AE PORT LUB.OIL TEMP.",
        "AE PORT HT FW OUTLET TEMP.",
        "AE PORT LUB.OIL PRESS.",
        "POWER_kW",
        "POWER_kW_sq",
    ]

    target_cols = [
        "AE PORT GEN.U-WINDING TEMP.",
        "AE PORT GEN.V-WINDING TEMP.",
        "AE PORT GEN.W-WINDING TEMP.",
    ]

    dataset = 1
    csv_path = paths.DATASET_PATH / "split" / f"Dataset_{dataset}" / "numeric" / "AE_PORT.csv"

    data = pd.read_csv(csv_path)

    data, dt_s = ts_cols(data, TS_COL)
    data = feature_processing(data)

    data, train_profiles, val_profiles, test_profiles, profile_sizes = train_val_test_split_profiles(
        data,
        ts_col=TS_COL,
        window_steps=WINDOW_STEPS,
        val_profile_len=VAL_PROFILE_LEN,
        test_profile_len=TEST_PROFILE_LEN,
        dt_s=dt_s,
    )

    data, x_scaler, y_scaler = scale_profile_data(
        data,
        train_profiles=train_profiles,
        input_cols=input_cols,
        target_cols=target_cols,
    )

    x_train, y_train, mask_train = tensorize_profiles(
        data, train_profiles, input_cols, target_cols, device=device
    )

    x_val, y_val, mask_val = tensorize_profiles(
        data, val_profiles, input_cols, target_cols, device=device
    )

    x_test, y_test, mask_test = tensorize_profiles(
        data, test_profiles, input_cols, target_cols, device=device
    )

    model_type = "TCN"  # or "LSTM"

    model = TCNBaseline(
        input_size=len(input_cols),
        output_size=len(target_cols),
        channels=(32, 64, 64),
        kernel_size=3,
        dropout=0.1,
    ).to(device)

    model, history = train_rnn_baseline(
        model,
        x_train,
        y_train,
        mask_train,
        x_val,
        y_val,
        mask_val,
        n_epochs=200,
        lr=1e-3,
        weight_decay=1e-5,
    )

    metrics = evaluate_rnn_model(
        model=model,
        x_test=x_test,
        y_test=y_test,
        mask_test=mask_test,
        y_scaler=y_scaler,
    )

    print(
        f"Final test metrics: "
        f"rmse={metrics['rmse']:.4f}, "
        f"mae={metrics['mae']:.4f}, "
        f"mse={metrics['mse']:.4f}, "
        f"max_abs={metrics['max_abs']:.4f}"
    )
    eval_plot_rnn(
        model=model,
        x_test=x_test,
        mask_test=mask_test,
        data=data,
        test_profiles=test_profiles,
        target_cols=target_cols,
        y_scaler=y_scaler,
        ts_col=TS_COL,
    )



    save_rnn_model(
        model=model,
        x_scaler=x_scaler,
        y_scaler=y_scaler,
        input_cols=input_cols,
        target_cols=target_cols,
        model_type=model_type,
        hidden_size=64,
        num_layers=2,
        dropout=0.1,
        dt_s=dt_s,
        path=paths.MODEL_DIR / f"{model_type}_winding_baseline.pt",
        history=history,
        params={
            "n_epochs": 200,
            "lr": 1e-3,
            "weight_decay": 1e-5,
            "hidden_size": 64,
            "num_layers": 2,
            "dropout": 0.1,
            "model_type": model_type,
        },
    )




if __name__ == "__main__":
    main()