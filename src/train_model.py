from __future__ import annotations

import json
from pathlib import Path

import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


DATA_PATH = Path("data/customer_churn.csv")
MODEL_PATH = Path("models/churn_model.joblib")
REPORTS_DIR = Path("reports")
TARGET = "churn"
DROP_COLUMNS = ["customer_id"]


def load_data(path: Path = DATA_PATH) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(
            f"{path} does not exist. Run `python3 src/generate_data.py` first."
        )
    return pd.read_csv(path)


def build_pipeline(df: pd.DataFrame) -> Pipeline:
    feature_columns = [column for column in df.columns if column not in DROP_COLUMNS + [TARGET]]
    numeric_features = df[feature_columns].select_dtypes(include=["int64", "float64"]).columns.tolist()
    categorical_features = [column for column in feature_columns if column not in numeric_features]

    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore")),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("numeric", numeric_pipeline, numeric_features),
            ("categorical", categorical_pipeline, categorical_features),
        ]
    )

    classifier = RandomForestClassifier(
        n_estimators=350,
        max_depth=9,
        min_samples_leaf=8,
        class_weight="balanced",
        random_state=42,
        n_jobs=1,
    )

    return Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("classifier", classifier),
        ]
    )


def get_feature_importance(model: Pipeline) -> pd.DataFrame:
    preprocessor = model.named_steps["preprocessor"]
    classifier = model.named_steps["classifier"]
    feature_names = preprocessor.get_feature_names_out()

    return (
        pd.DataFrame(
            {
                "feature": feature_names,
                "importance": classifier.feature_importances_,
            }
        )
        .sort_values("importance", ascending=False)
        .reset_index(drop=True)
    )


def save_confusion_matrix(y_test: pd.Series, y_pred: pd.Series) -> None:
    matrix = confusion_matrix(y_test, y_pred)
    pd.DataFrame(
        matrix,
        index=["actual_stay", "actual_churn"],
        columns=["predicted_stay", "predicted_churn"],
    ).to_csv(REPORTS_DIR / "confusion_matrix.csv")


def train() -> dict[str, float]:
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)

    df = load_data()
    X = df.drop(columns=DROP_COLUMNS + [TARGET])
    y = df[TARGET]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        stratify=y,
        random_state=42,
    )

    model = build_pipeline(df)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    metrics = {
        "accuracy": round(accuracy_score(y_test, y_pred), 4),
        "precision": round(precision_score(y_test, y_pred), 4),
        "recall": round(recall_score(y_test, y_pred), 4),
        "f1": round(f1_score(y_test, y_pred), 4),
        "roc_auc": round(roc_auc_score(y_test, y_proba), 4),
        "test_rows": int(len(y_test)),
        "training_rows": int(len(y_train)),
    }

    joblib.dump(model, MODEL_PATH)
    (REPORTS_DIR / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    (REPORTS_DIR / "classification_report.txt").write_text(
        classification_report(y_test, y_pred, target_names=["Stay", "Churn"]),
        encoding="utf-8",
    )

    feature_importance = get_feature_importance(model)
    feature_importance.to_csv(REPORTS_DIR / "feature_importance.csv", index=False)
    save_confusion_matrix(y_test, y_pred)

    return metrics


def main() -> None:
    metrics = train()
    print(f"Saved model to {MODEL_PATH}")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
