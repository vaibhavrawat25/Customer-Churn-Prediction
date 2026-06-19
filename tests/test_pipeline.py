import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.generate_data import generate_customer_churn_data
from src.train_model import TARGET, build_pipeline


def test_generated_data_contains_target() -> None:
    df = generate_customer_churn_data(rows=100, seed=7)
    assert TARGET in df.columns
    assert df[TARGET].isin([0, 1]).all()


def test_model_pipeline_can_fit_and_predict() -> None:
    df = generate_customer_churn_data(rows=250, seed=8)
    X = df.drop(columns=["customer_id", TARGET])
    y = df[TARGET]

    model = build_pipeline(df)
    model.fit(X, y)
    predictions = model.predict(X.head(5))

    assert len(predictions) == 5
    assert set(predictions).issubset({0, 1})
