import config as cfg

# ---------------------------------------------------------
# Shared training configuration
# ---------------------------------------------------------
COMMON_TRAINING_CONFIG = {
    "n_epochs": 200,
    "max_grad_norm": 1.0,
}

WINDOW_STEPS = 300

# =========================================================
# TCN
# =========================================================
TCN_MODEL_TYPE = "TCN"

TCN_MODEL_CONFIG = {
    "input_size": len(cfg.INPUT_COLS),
    "output_size": len(cfg.TARGET_COLS),
    "channels": (32, 64, 64, 64),
    "kernel_size": 5,
    "dropout": 0.3,
}

TCN_TRAINING_CONFIG = {
    **COMMON_TRAINING_CONFIG,
    "lr": 0.003,
    "weight_decay": 1e-5,
}

TCN_SEARCH_SPACE = {
    "channels": [
        (32, 32, 32),
        (32, 64, 64, 64),
        (32, 64, 64, 64, 64),
        (64, 64, 128, 128),
    ],
    "kernel_size": [3, 5, 7],
    "dropout": [0.0, 0.1, 0.3],
    "lr": [
        1e-4,
        3e-4,
        1e-3,
        3e-3,
    ],
    "weight_decay": [
        0.0,
        1e-5,
        1e-4,
        1e-3,
    ],
}


# =========================================================
# LSTM
# =========================================================
RNN_MODEL_TYPE = "LSTM"

RNN_MODEL_CONFIG = {
    "input_size": len(cfg.INPUT_COLS),
    "output_size": len(cfg.TARGET_COLS),
    "hidden_size": 128,
    "num_layers": 3,
    "dropout": 0.1,
    "model_type": RNN_MODEL_TYPE,
}

RNN_TRAINING_CONFIG = {
    **COMMON_TRAINING_CONFIG,
    "lr": 0.003,
    "weight_decay": 0.0,
}

RNN_SEARCH_SPACE = {
    "hidden_size": [64, 128, 256],
    "num_layers": [1, 2, 3],
    "dropout": [0.0, 0.1, 0.3],
    "lr": [
        1e-4,
        3e-4,
        1e-3,
        3e-3,
    ],
    "weight_decay": [
        0.0,
        1e-5,
        1e-4,
        1e-3,
    ],
}