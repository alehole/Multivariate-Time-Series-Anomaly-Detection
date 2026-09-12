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

TNN_GRID = {
    "n_epochs": [100, 200],
    "lr": [0.1, 0.01],
    "tbptt_size": [256, 512],
    "weight_decay": [0.0, 0.1, 0.01, 0.001, 0.0001, 0.00001],
    "smoothness_weight": [0.0, 0.01, 0.001],
    "n_neurons": [8, 16, 32, 64],
}
