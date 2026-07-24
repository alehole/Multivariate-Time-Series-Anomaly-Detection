import config as cfg

# ---------------------------------------------------------
# Shared training configuration
# ---------------------------------------------------------
TRAINING_CONFIG = {
    "n_epochs": 200,
    "lr": 1e-3,
    "weight_decay": 1e-5,
    "max_grad_norm": 1.0,
}

WINDOW_STEPS = 300
# ---------------------------------------------------------
# TCN configuration
# ---------------------------------------------------------
TCN_MODEL_TYPE = "TCN"

TCN_MODEL_CONFIG = {
    "input_size": len(cfg.INPUT_COLS),
    "output_size": len(cfg.TARGET_COLS),
    "channels": (32, 64, 64, 64, 64),
    "kernel_size": 3,
    "dropout": 0.1,
}

# ---------------------------------------------------------
# RNN configuration
# ---------------------------------------------------------
RNN_MODEL_TYPE = "LSTM" # "LSTM" or "GRU"

RNN_MODEL_CONFIG = {
    "input_size": len(cfg.INPUT_COLS),
    "output_size": len(cfg.TARGET_COLS),
    "hidden_size": 128,
    "num_layers": 2,
    "dropout": 0.1,
    "model_type": RNN_MODEL_TYPE,
}
