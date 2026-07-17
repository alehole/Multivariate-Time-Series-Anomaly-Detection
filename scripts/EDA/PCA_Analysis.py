import pandas as pd
import matplotlib.pyplot as plt

from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

import config as cfg
# ---------------------------------------------------------
# Configuration
# ---------------------------------------------------------

dataset = 1
dataset_name = f"DS{dataset}"
csv_path = (
        cfg.DATA_PATH
        / "subsystems"
        / dataset_name
        / "NUMERIC"
        / "AE_PORT.csv"
)


N_COMPONENTS = 2

if dataset == 1:
    drop_cols = [
        "POWER_kW_sq",
        "AE PS RUNNING",
        "AE PS POWER COUNTER",
        "AE PORT CYL.1 EXH.GAS TEMP. DEV",
        "AE PORT CYL.2 EXH.GAS TEMP. DEV",
        "AE PORT CYL.3 EXH.GAS TEMP. DEV",
        "AE PORT CYL.4 EXH.GAS TEMP. DEV",
        "AE PORT CYL.5 EXH.GAS TEMP. DEV",
        "AE PORT CYL.6 EXH.GAS TEMP. DEV",
    ]
else:
    drop_cols = [
        "POWER_kW_sq",
        "AE SB RUNNING",
        "AE SB POWER COUNTER",
        "AE STBD CYL.1 EXH.GAS TEMP. DEV",
        "AE STBD CYL.2 EXH.GAS TEMP. DEV",
        "AE STBD CYL.3 EXH.GAS TEMP. DEV",
        "AE STBD CYL.4 EXH.GAS TEMP. DEV",
        "AE STBD CYL.5 EXH.GAS TEMP. DEV",
        "AE STBD CYL.6 EXH.GAS TEMP. DEV",
    ]
    # Build the numeric analysis frame (same filtering as loop_folder).

# ---------------------------------------------------------
# Load data
# ---------------------------------------------------------
df = pd.read_csv(csv_path)

df = (
    df.select_dtypes(include="number")
    .drop(columns=[*drop_cols], errors="ignore")
    .copy()
)

# Keep only numeric columns
numeric_df = df.select_dtypes(include="number")

# Remove columns with no variation
numeric_df = numeric_df.loc[:, numeric_df.std() > 0]

# Remove rows containing missing values
numeric_df = numeric_df.dropna()

print(f"Rows used: {len(numeric_df)}")
print(f"Variables used: {numeric_df.shape[1]}")


# ---------------------------------------------------------
# Standardize data
# ---------------------------------------------------------
scaler = StandardScaler()
X_scaled = scaler.fit_transform(numeric_df)


# ---------------------------------------------------------
# PCA
# ---------------------------------------------------------
pca = PCA(n_components=N_COMPONENTS)
X_pca = pca.fit_transform(X_scaled)

pca_df = pd.DataFrame(
    X_pca,
    columns=[f"PC{i + 1}" for i in range(N_COMPONENTS)],
    index=numeric_df.index,
)

print("\nExplained variance ratio:")
for i, variance in enumerate(pca.explained_variance_ratio_):
    print(f"PC{i + 1}: {variance:.2%}")

print(
    f"Total explained variance: "
    f"{pca.explained_variance_ratio_.sum():.2%}"
)


# ---------------------------------------------------------
# Plot PC1 against PC2
# ---------------------------------------------------------
plt.figure(figsize=(8, 6))

plt.scatter(
    pca_df["PC1"],
    pca_df["PC2"],
    s=10,
    alpha=0.6,
)

plt.xlabel(
    f"PC1 ({pca.explained_variance_ratio_[0]:.1%})"
)
plt.ylabel(
    f"PC2 ({pca.explained_variance_ratio_[1]:.1%})"
)
plt.title("PCA of CSV Dataset")
plt.grid(alpha=0.3)
plt.tight_layout()
plt.show()


# ---------------------------------------------------------
# Loadings
# ---------------------------------------------------------
loadings = pd.DataFrame(
    pca.components_.T,
    index=numeric_df.columns,
    columns=["PC1", "PC2"],
)

print("\nLargest contributors to PC1:")
print(loadings["PC1"].abs().sort_values(ascending=False).head(20))

print("\nLargest contributors to PC2:")
print(loadings["PC2"].abs().sort_values(ascending=False).head(20))