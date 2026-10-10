WINDOW_STEPS = 300
TNN_MODEL_TYPE = "TNN"

'''
TNN_TRAINING_CONFIG = {
    "n_epochs": 300,
    "lr": 0.1,
    "tbptt_size": 256,
    "weight_decay": 0.001,
    "n_neurons": 512,
}
'''

TNN_SEARCH_SPACE = {
    "n_neurons": [128, 256, 512, 1024],
    "lr": [0.003, 0.01, 0.03, 0.1],
    "weight_decay": [1e-4, 1e-3, 1e-2, 3e-2],
    "tbptt_size": [64, 128, 256, 512],
}

TNN_TRAINING_CONFIG = {
    "n_epochs": 300,
    "lr": 0.03,
    "tbptt_size": 64,
    "weight_decay": 0.001,
    "n_neurons": 256,
}


