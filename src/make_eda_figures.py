import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

from src.config import PROCESSED_DATA_PATH, FIGURES_DIR, TARGET_COLUMN


def save_class_distribution(df: pd.DataFrame) -> None:
    crop_counts = df[TARGET_COLUMN].value_counts().sort_index()

    plt.figure(figsize=(12, 7))
    sns.countplot(
        data=df,
        y=TARGET_COLUMN,
        order=crop_counts.index
    )
    plt.title("Distribution of Crop Classes")
    plt.xlabel("Number of Samples")
    plt.ylabel("Crop")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "crop_class_distribution.png", dpi=300)
    plt.close()


def save_correlation_heatmap(df: pd.DataFrame, feature_cols: list[str]) -> None:
    corr_matrix = df[feature_cols].corr()

    plt.figure(figsize=(9, 7))
    sns.heatmap(
        corr_matrix,
        annot=True,
        cmap="coolwarm",
        fmt=".2f",
        square=True
    )
    plt.title("Correlation Matrix of Numeric Features")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "correlation_heatmap.png", dpi=300)
    plt.close()


def save_crop_feature_heatmap(df: pd.DataFrame, feature_cols: list[str]) -> None:
    crop_feature_means = df.groupby(TARGET_COLUMN)[feature_cols].mean().round(2)

    plt.figure(figsize=(12, 8))
    sns.heatmap(
        crop_feature_means,
        annot=True,
        fmt=".1f",
        cmap="viridis"
    )
    plt.title("Average Soil and Climate Conditions by Crop")
    plt.xlabel("Feature")
    plt.ylabel("Crop")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "crop_feature_profile_heatmap.png", dpi=300)
    plt.close()


def save_pca_plot(df: pd.DataFrame, feature_cols: list[str]) -> None:
    X = df[feature_cols]
    y = df[TARGET_COLUMN]

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    pca = PCA(n_components=2, random_state=42)
    X_pca = pca.fit_transform(X_scaled)

    pca_df = pd.DataFrame(X_pca, columns=["PC1", "PC2"])
    pca_df[TARGET_COLUMN] = y.values

    plt.figure(figsize=(12, 8))
    sns.scatterplot(
        data=pca_df,
        x="PC1",
        y="PC2",
        hue=TARGET_COLUMN,
        palette="tab20",
        alpha=0.8,
        s=50
    )
    plt.title("PCA Visualization of Crop Classes")
    plt.xlabel(f"PC1 ({pca.explained_variance_ratio_[0]:.2%} variance)")
    plt.ylabel(f"PC2 ({pca.explained_variance_ratio_[1]:.2%} variance)")
    plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "pca_crop_classes.png", dpi=300)
    plt.close()


def main() -> None:
    sns.set_theme(style="whitegrid")

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(PROCESSED_DATA_PATH)
    feature_cols = df.drop(columns=TARGET_COLUMN).columns.tolist()

    save_class_distribution(df)
    save_correlation_heatmap(df, feature_cols)
    save_crop_feature_heatmap(df, feature_cols)
    save_pca_plot(df, feature_cols)

    print("EDA figures saved successfully.")


if __name__ == "__main__":
    main()