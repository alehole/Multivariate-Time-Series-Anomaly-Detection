# Multivariate-Time-Series-Anomaly-Detection
This repository contains the Python implementation developed for the master’s thesis:

### Unsupervised Machine Learning for Time-Series Anomaly Detection in Maritime Condition-Based Monitoring

#### Project overview

Repository structure
## Repository structure

```text
project-root/
├── config/
│   ├── config.py
│   └── config_example.py
│
├── data/
│   ├── raw/
│   ├── raw_categorized/
│   ├── subsystems/
│   ├── train_test_split/
│   ├── EDA/
│   └── results/
│
├── scripts/
│   ├── Subsystem_grouping/
│   │   ├── categorize_raw_dataset.py
│   │   └── group_categorized_dataset.py
│   │
│   ├── misc/
│   │   ├── combine_csv.py
│   │   ├── feature_engineering.py
│   │   └── split_csv.py
│   │
│   ├── EDA/
│   │   ├── temperature_API.py
│   │   ├── sea_state.py
│   │   ├── plot_sensor_distributions.py
│   │   ├── sensor_statistics.py
│   │   └── correlation_analysis.py
│   │
│   └── modelling/
│       ├── model.py
│       ├── ekf.py
│       ├── likelihood.py
│       ├── parameter_estimation.py
│       ├── PL1.py
│       ├── PL2.py
│       └── MCMC.py
│
├── run_pipeline.py
├── requirements.txt
├── README.md
└── .gitignore
```
Running the preprocessing and EDA pipeline

The complete preprocessing pipeline can be executed using:

python run_pipeline.py

The pipeline performs the following operations:
```text
Raw SCADA data
       ↓
Column categorisation
       ↓
Subsystem grouping
       ↓
CSV combination
       ↓
Feature engineering
       ↓
Ambient-temperature retrieval
       ↓
Vessel-state classification
       ↓
Sensor statistics
       ↓
Distribution plots
       ↓
Correlation analysis
```

### Data

The project uses operational vessel SCADA data provided by an industrial partner.

Due to confidentiality restrictions, the original datasets are not included in this repository.

The expected input structure is:
```text
data/
└── raw/
    ├── DS1/
    │   └── LiveData.csv
    └── DS2/
        └── LiveData.csv
```

### Author

#### Aleksander Holthe
Master’s programme in Industrial IT and Automation
University of South-Eastern Norway