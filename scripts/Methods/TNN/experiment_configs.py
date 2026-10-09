WINDOW_STEPS = 300
TNN_MODEL_TYPE = "TNN"

TNN_TRAINING_CONFIG = {
    "n_epochs": 300,
    "lr": 0.1,
    "tbptt_size": 256,
    "weight_decay": 0.001,
    "smoothness_weight": 0.0,
    "n_neurons": 512,
}

TNN_SEARCH_SPACE = {
    "n_neurons": [128, 256, 512],
    "lr": [0.01, 0.03, 0.1],
    "weight_decay": [1e-4, 1e-3, 1e-2],
    "tbptt_size": [128, 256, 512],
    "smoothness_weight": [0.0, 1e-4, 1e-3],
}

TNN_BEST_CONFIG = {
    "n_epochs": 300,
    "lr": 0.03,
    "tbptt_size": 128,
    "weight_decay": 0.01,
    "smoothness_weight": 0.0001,
    "n_neurons": 512,
    "best_val_loss": 0.05405641347169876,
    "best_epoch": 238.0,
}
