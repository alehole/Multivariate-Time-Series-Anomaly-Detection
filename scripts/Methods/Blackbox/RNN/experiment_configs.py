WINDING_CONFIG_1 = {
    "ts_col": "Created",
    "window_steps": 300,
    "val_profile_len": 3,
    "test_profile_len": 3,

    "target_cols": [
        "AE PORT GEN.U-WINDING TEMP.",
        "AE PORT GEN.V-WINDING TEMP.",
        "AE PORT GEN.W-WINDING TEMP.",
    ],

    "input_cols": [
        "AE_PS_EXH",
        'AE PORT END BRG.TEMP.',
        'AE PORT LUB.OIL TEMP.',
        "AE PORT TC EXH.GAS OUT.TEMP.",
        "AE PORT HT FW OUTLET TEMP.",
        "FAN 3 TEMPERATURE",
        "FAN 1 TEMPERATURE",
        "LT FW TEMP.",
        "POWER_kW",
    ],

    "default_params": {
        "hidden_size": 64,
        "num_layers": 2,
        "dropout": 0.1,
        "n_epochs": 200,
        "lr": 1e-3,
        "weight_decay": 1e-5,
    },
}

CYL3_CONFIG = {
    "ts_col": "Created",
    "window_steps": 300,
    "val_profile_len": 3,
    "test_profile_len": 3,

    "target_cols": [
        "AE PORT CYL.3 EXH.GAS TEMP.",
    ],

    "input_cols": [
        "AE PORT END BRG.TEMP.",
        "AE_PS_EXH",
        "AE PORT HT FW OUTLET TEMP.",
        "AE PORT TC EXH.GAS OUT.TEMP.",
        "AE PORT CYL.1 EXH.GAS TEMP.",
        "AE PORT CYL.2 EXH.GAS TEMP.",
        "AE PORT CYL.4 EXH.GAS TEMP.",
        "AE PORT CYL.5 EXH.GAS TEMP.",
        "AE PORT CYL.6 EXH.GAS TEMP.",
    ],

    "default_params": {
        "hidden_size": 64,
        "num_layers": 2,
        "dropout": 0.1,
        "n_epochs": 200,
        "lr": 1e-3,
        "weight_decay": 1e-5,
    },
}

CYL_ALL_CONFIG = {
    "ts_col": "Created",
    "window_steps": 300,
    "val_profile_len": 3,
    "test_profile_len": 3,

    "target_cols": [
        "AE PORT CYL.1 EXH.GAS TEMP.",
        "AE PORT CYL.2 EXH.GAS TEMP.",
        "AE PORT CYL.3 EXH.GAS TEMP.",
        "AE PORT CYL.4 EXH.GAS TEMP.",
        "AE PORT CYL.5 EXH.GAS TEMP.",
        "AE PORT CYL.6 EXH.GAS TEMP.",
    ],

    "input_cols": [
        "AE PORT END BRG.TEMP.",
        "AE_PS_EXH",
        "AE PORT HT FW OUTLET TEMP.",
        "AE PORT TC EXH.GAS OUT.TEMP.",
        "FAN 3 TEMPERATURE",
        "FAN 1 TEMPERATURE",
        "LT FW TEMP.",
    ],

    "default_params": {
        "hidden_size": 64,
        "num_layers": 2,
        "dropout": 0.1,
        "n_epochs": 200,
        "lr": 1e-3,
        "weight_decay": 1e-5,
    },
}