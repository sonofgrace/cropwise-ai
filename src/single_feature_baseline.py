import json

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

from src.config import PROCESSED_DATA_PATH, TARGET_COLUMN, RANDOM_STATE, FIGURES_DIR


def evaluate_single_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Evaluate each feature individually using Logistic Regression and macro-F1.

    Parameters
    ----------
    df : pd.DataFrame
        Clean crop recommendation dataset.

    Returns
    -------
    pd.DataFrame
        Feature ranking by mean macro-F1 score.
    """
    X = df.drop(columns=TARGET_COLUMN)
    y = df[TARGET_COLUMN]

    cv = StratifiedKFold(
        n_splits=5,
        shuffle=True,
        random_state=RANDOM_STATE
    )

    feature_scores = {}

    for feature in X.columns:
        pipeline = Pipeline(
            steps=[
                ("scaler", StandardScaler()),
                ("model", LogisticRegression(max_iter=5000))
            ]
        )

        scores = cross_val_score(
            pipeline,
            X[[feature]],
            y,
            cv=cv,
            scoring="f1_macro"
        )

        feature_scores[feature] = scores.mean()

    feature_score_df = (
        pd.DataFrame({
            "feature": feature_scores.keys(),
            "macro_f1": feature_scores.values()
        })
        .sort_values("macro_f1", ascending=False)
        .reset_index(drop=True)
    )

    return feature_score_df


def save_single_feature_plot(feature_score_df: pd.DataFrame) -> None:
    """
    Save a bar chart of single-feature predictive performance.
    """
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(10, 6))
    sns.barplot(
        data=feature_score_df,
        x="macro_f1",
        y="feature"
    )
    plt.title("Single-Feature Predictive Performance")
    plt.xlabel("Mean Macro-F1 Score")
    plt.ylabel("Feature")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "single_feature_performance.png", dpi=300)
    plt.close()


def main() -> None:
    df = pd.read_csv(PROCESSED_DATA_PATH)

    feature_score_df = evaluate_single_features(df)

    best_feature = feature_score_df.loc[0, "feature"]
    best_score = feature_score_df.loc[0, "macro_f1"]

    best_predictive_feature = {
        best_feature: round(float(best_score), 4)
    }

    print("Single-feature baseline results:")
    print(feature_score_df)

    print("\nBest predictive feature:")
    print(best_predictive_feature)

    save_single_feature_plot(feature_score_df)

    output_path = FIGURES_DIR.parent / "single_feature_scores.json"
    with open(output_path, "w") as f:
        json.dump(
            {
                "best_predictive_feature": best_predictive_feature,
                "all_feature_scores": {
                    row["feature"]: round(float(row["macro_f1"]), 4)
                    for _, row in feature_score_df.iterrows()
                },
            },
            f,
            indent=4
        )

    print(f"\nResults saved to: {output_path}")


if __name__ == "__main__":
    main()