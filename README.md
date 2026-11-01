# Multivariate-Time-Series-Anomaly-Detection
This repository contains the Python implementation for the master’s thesis:

**Predictive analysis based on time-series data**

### Project overview
This project investigates multivariate time-series anomaly detection (MTSAD) for condition-based monitoring (CBM) of maritime vessels.

The study uses real-world Supervisory Control and Data Acquisition (SCADA) data provided by an industry partner, focusing on the thermal behaviour of maritime power generators.

The investigated modelling approaches include:

- **Stochastic Differential Equations (SDE):** Physics-based grey-box thermal models using state-space estimation.
- **Thermal Neural Networks (TNN):** Hybrid models combining physical thermal relationships with neural networks.
- **Recurrent Neural Networks (RNN):** Data-driven sequence models, including LSTM and GRU architectures.
- **Temporal Convolutional Networks (TCN):** Data-driven convolutional models for time-series prediction.

Anomalies are identified through deviations between measured and predicted generator winding temperatures.

## Repository Structure

```text
project-root/
├── data/                          # SCADA datasets and generated results
│
├── scripts/
│   ├── Subsystem_grouping/        # Dataset categorisation and grouping
│   │   ├── categorize_raw_dataset.py
│   │   └── group_categorized_dataset.py
│   │
│   ├── misc/                      # Data preparation and feature engineering
│   │   ├── report_scripts/
│   │   ├── combine_csv.py
│   │   ├── data.py
│   │   ├── feature_engineering.py
│   │   └── train_val_test_split.py
│   │
│   ├── EDA/                       # Exploratory Data Analysis
│   │   ├── auto_correlation.py
│   │   ├── compare_sensors.py
│   │   ├── correlation_analysis.py
│   │   ├── cross_correlation.py
│   │   ├── plot_sensor_distributions.py
│   │   ├── plot_sensor_distributions_v2.py
│   │   ├── plot_sensors.py
│   │   ├── plot_sensors_v2.py
│   │   ├── sea_state.py
│   │   ├── sensor_statistics.py
│   │   └── temperature_API.py
│   │
│   ├── Methods/                   # Predictive modelling and anomaly detection
│   │   ├── Anomalies/
│   │   ├── Blackbox/
│   │   ├── hyperparams/
│   │   ├── SDE/
│   │   ├── TNN/
│   │   └── random_search.py
│   │
│   ├── Visualization/             # Model results and vessel visualisation
│   │   ├── metrics.py
│   │   ├── plot_vessel_loc.py
│   │   ├── plotly_plot.py
│   │   ├── prediction_plots.py
│   │   └── residual_plots.py
│   │
│   └── run_pipeline.py           # Full preprocessing and EDA pipeline
│
├── config.py                      # Project configuration
├── tag_definitions.json          # SCADA sensor tag definitions
├── .gitignore
└── README.md
```

## Running the Preprocessing and EDA Pipeline

The preprocessing and Exploratory Data Analysis (EDA) pipeline can be
executed from the project root using:

```bash
python run_pipeline.py
```

The pipeline consists of four main stages: dataset grouping, data
preprocessing and feature engineering, exploratory data analysis,
and visualization.

### Pipeline Structure

```text
Raw SCADA data
      |
      v
1. DATASET GROUPING
   |-- Column categorisation
   |-- Subsystem grouping
      |
      v
2. DATA PREPROCESSING
   |-- CSV combination
   |-- Feature engineering
   |-- Train/validation/test splitting
      |
      v
3. EXPLORATORY DATA ANALYSIS
   |-- Sensor statistics
   |-- Sensor comparison
   |-- Correlation analysis
   |-- Autocorrelation analysis
   |-- Cross-correlation analysis
   |-- Sensor distribution plots
   |-- Sensor distribution plots (v2)
   |-- Sensor time-series plots
   |-- Sensor time-series plots (v2)
   |-- Vessel-state classification
   |-- Ambient-temperature retrieval
      |
      v
4. VISUALIZATION
   |-- Interactive Plotly visualization
   |-- Vessel location visualization
```

### Pipeline Configuration

The execution order and enabled operations are defined in the
`PIPELINE` list in `run_pipeline.py`.

Each operation can be enabled or disabled independently using its Boolean flag (`True` or `False`).

```python
PIPELINE = [
    # 1. Dataset grouping
    ("Categorize raw dataset", categorize_raw_dataset, True),
    ("Group categorized dataset", group_categorized_dataset, True),

    # 2. Data preprocessing and feature engineering
    ("Combine CSV", combine_csv, True),
    ("Feature engineering", feature_engineering, True),
    ("Train/Val/Test split", train_val_test_split, True),

    # 3. Exploratory Data Analysis
    ("Sensor statistics", sensor_statistics, False),
    ("Compare sensors", compare_sensors, False),
    ("Correlation analysis", correlation_analysis, False),
    ("Auto correlation", auto_correlation, False),
    ("Cross correlation", cross_correlation, False),
    ("Plot distributions", plot_distributions, False),
    ("Plot distributions v2", plot_distributions_v2, True),
    ("Plot sensors", plot_sensors, False),
    ("Plot sensors v2", plot_sensors_v2, False),
    ("Sea state", sea_state, False),
    ("Temperature", temperature, False),

    # 4. Visualization
    ("Plotly plot", plotly_plot, False),
    ("Vessel location", vessel_loc, False),
]
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

#### Aleksander Wad Holthe
Master’s programme in Industrial IT and Automation
University of South-Eastern Norway