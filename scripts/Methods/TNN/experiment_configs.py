WINDOW_STEPS = 300
TNN_MODEL_TYPE = "TNN"

TNN_SEARCH_SPACE = {
    "n_neurons": [128, 256, 512],
    "lr": [0.05, 0.01, 0.03],
    "weight_decay": [1e-4, 1e-3, 1e-2, 3e-2],
    "tbptt_size": [50, 100, 200, 300],
}

TNN_TRAINING_CONFIG = {
    "n_epochs": 200,
    "lr": 0.05,
    "tbptt_size": 128,
    "weight_decay": 0.001,
    "n_neurons": 512,
}

