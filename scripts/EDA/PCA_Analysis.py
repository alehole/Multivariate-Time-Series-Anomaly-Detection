import pandas as pd
import matplotlib.pyplot as plt

from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

import config as cfg



import pandas as pd
import matplotlib.pyplot as plt

from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler





def pca_analysis(
    numeric_df: pd.DataFrame,
    n_components: int = 2,
) -> tuple[PCA, pd.DataFrame]:

    if numeric_df.empty:
        raise ValueError("numeric_df is empty.")

    if numeric_df.isna().any().any():
        raise ValueError(
            "numeric_df contains missing values. "
            "Remove or impute them before PCA."
        )

    if n_components > numeric_df.shape[1]:
        raise ValueError(
            f"n_components={n_components}, but the dataset only "
            f"contains {numeric_df.shape[1]} variables."
        )

    # Standardize the variables
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(numeric_df)

    # Perform PCA
    pca = PCA(n_components=n_components)
    X_pca = pca.fit_transform(X_scaled)

    component_names = [
        f"PC{i + 1}"
        for i in range(n_components)
    ]

    pca_df = pd.DataFrame(
        X_pca,
        columns=component_names,
        index=numeric_df.index,
    )

    print("\nExplained variance ratio:")

    for component, variance in zip(
        component_names,
        pca.explained_variance_ratio_,
    ):
        print(f"{component}: {variance:.2%}")

    print(
        "Total explained variance: "
        f"{pca.explained_variance_ratio_.sum():.2%}"
    )

    return pca, pca_df


def plot_pca_components(
    pca_df: pd.DataFrame,
    pca: PCA,
) -> None:
    """
    Plot the scores of PC1 against PC2.
    """
    if pca.n_components_ < 2:
        raise ValueError(
            "At least two principal components are required "
            "for a PC1-PC2 score plot."
        )

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
    plt.title("PCA")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()

def print_loadings(
    pca: PCA,
    feature_names: pd.Index,
    n_loadings: int = 20,
) -> pd.DataFrame:
    """
    Print the variables with the largest absolute loading for
    each principal component.
    """
    component_names = [
        f"PC{i + 1}"
        for i in range(pca.n_components_)
    ]

    loadings = pd.DataFrame(
        pca.components_.T,
        index=feature_names,
        columns=component_names,
    )

    for component in component_names:
        order = (
            loadings[component]
            .abs()
            .sort_values(ascending=False)
            .head(n_loadings)
            .index
        )

        print(f"\nLargest contributors to {component}:")
        print(loadings.loc[order, component])

    return loadings

def main():


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

    N_COMPONENTS = 4

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



    # numeric_df must already be prepared before this point
    pca, pca_df = pca_analysis(
        numeric_df=numeric_df,
        n_components=N_COMPONENTS,
    )

    plot_pca_components(
        pca_df=pca_df,
        pca=pca,
    )

    loadings = print_loadings(
        pca=pca,
        feature_names=numeric_df.columns,
        n_loadings=20,
    )

    dataset = 2
    dataset_name = f"DS{dataset}"
    csv_path = (
            cfg.DATA_PATH
            / "subsystems"
            / dataset_name
            / "NUMERIC"
            / "AE_STBD.csv"
    )

    N_COMPONENTS = 4

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

    # numeric_df must already be prepared before this point
    pca, pca_df = pca_analysis(
        numeric_df=numeric_df,
        n_components=N_COMPONENTS,
    )

    plot_pca_components(
        pca_df=pca_df,
        pca=pca,
    )

    loadings = print_loadings(
        pca=pca,
        feature_names=numeric_df.columns,
        n_loadings=20,
    )





if __name__ == "__main__":
    main()