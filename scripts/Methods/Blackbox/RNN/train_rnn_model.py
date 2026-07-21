import torch
import pandas as pd

from src.preprocess.feature_engineering import ts_cols, feature_processing
from src.models.RNN.rnn_model import RNNBaseline, train_rnn_baseline
from src.models.profile_dataset import (
    train_val_test_split_profiles,
    scale_profile_data,
    tensorize_profiles,
)

from src.models.model_utils import (
    evaluate_sequence_model,
    eval_plot_sequence,
    print_metrics,
    set_seed,
)

from src.config import paths
from pathlib import Path

#from src.models.RNN.experiment_configs import WINDING_CONFIG_1 as cfg
#from src.models.RNN.experiment_configs import CYL3_CONFIG as cfg
from src.models.RNN.experiment_configs import CYL_ALL_CONFIG as cfg

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
    set_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_type = "LSTM"  # or "GRU"
    TS_COL = cfg["ts_col"]
    WINDOW_STEPS = cfg["window_steps"]
    TEST_PROFILE_LEN = cfg["test_profile_len"]
    VAL_PROFILE_LEN = cfg["val_profile_len"]
    input_cols = cfg["input_cols"]
    target_cols = cfg["target_cols"]
    params = cfg["default_params"]

    dataset = 1
    #csv_path = paths.DATASET_PATH / "split" / f"Dataset_{dataset}" / "numeric" / "AE_PORT.csv"
    csv_path = paths.DATASET_PATH / "split" / "temp" / "AE_PORT_with_extra_TEMP.csv"
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

    model = RNNBaseline(
        input_size=len(input_cols),
        output_size=len(target_cols),
        hidden_size=params["hidden_size"],
        num_layers=params["num_layers"],
        dropout=params["dropout"],
        model_type=model_type,
    ).to(device)

    model, history = train_rnn_baseline(
        model,
        x_train,
        y_train,
        mask_train,
        x_val,
        y_val,
        mask_val,
        n_epochs=params["n_epochs"],
        lr=params["lr"],
        weight_decay=params["weight_decay"],
    )

    metrics = evaluate_sequence_model(
        model=model,
        x_test=x_test,
        y_test=y_test,
        mask_test=mask_test,
        y_scaler=y_scaler,
    )

    print_metrics(metrics)

    eval_plot_sequence(
        model=model,
        x_test=x_test,
        test_mask=mask_test,
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
        hidden_size=params["hidden_size"],
        num_layers=params["num_layers"],
        dropout=params["dropout"],
        dt_s=dt_s,
        path=paths.MODEL_DIR / f"{model_type}_cylinder_all.pt",
        history=history,
        params={
            "n_epochs": params["n_epochs"],
            "lr": params["lr"],
            "weight_decay": params["weight_decay"],
            "hidden_size": params["hidden_size"],
            "num_layers": params["num_layers"],
            "dropout": params["dropout"],
            "model_type": model_type,
        },
    )

if __name__ == "__main__":
    main()