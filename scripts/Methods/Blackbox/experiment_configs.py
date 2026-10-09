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
TRAINING_SEARCH_SPACE = {
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
TCN_SEARCH_SPACE = {
    "channels": [
        (32, 32, 32),
        (32, 64, 64, 64),
        (32, 64, 64, 64, 64),
        (64, 64, 128, 128),
    ],
    "kernel_size": [3, 5, 7],
    "dropout": [0.0, 0.1, 0.3],
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

RNN_SEARCH_SPACE = {
    "hidden_size": [64, 128, 256],
    "num_layers": [1, 2, 3],
    "dropout": [0.0, 0.1, 0.3],
}

## Best configs
RNN_BEST_CONFIG = {
    "input_size": len(cfg.INPUT_COLS),
    "output_size": len(cfg.TARGET_COLS),
    "hidden_size": 256,
    "num_layers": 2,
    "dropout": 0.3,
    "lr": 0.001,
    "weight_decay": 0.0001,
    "model_type": RNN_MODEL_TYPE,
    'best_val_loss': 0.07345976680517197
}

TCN_BEST_CONFIG = {
    "input_size": len(cfg.INPUT_COLS),
    "output_size": len(cfg.TARGET_COLS),
    "channels": (64, 64, 128, 128),
    "kernel_size": 3,
    "dropout": 0.0,
    'lr': 0.003,
    'weight_decay': 0.0001,
    'best_val_loss': 0.0636480301618576,
}
