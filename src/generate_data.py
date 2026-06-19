from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


DATA_PATH = Path("data/customer_churn.csv")
RANDOM_SEED = 42


def sigmoid(values: np.ndarray) -> np.ndarray:
    return 1 / (1 + np.exp(-values))


def generate_customer_churn_data(rows: int = 5000, seed: int = RANDOM_SEED) -> pd.DataFrame:
    rng = np.random.default_rng(seed)

    tenure_months = rng.integers(1, 73, rows)
    senior_citizen = rng.binomial(1, 0.17, rows)
    partner = rng.choice(["Yes", "No"], rows, p=[0.48, 0.52])
    dependents = rng.choice(["Yes", "No"], rows, p=[0.31, 0.69])
    phone_service = rng.choice(["Yes", "No"], rows, p=[0.9, 0.1])
    multiple_lines = np.where(
        phone_service == "Yes",
        rng.choice(["Yes", "No"], rows, p=[0.46, 0.54]),
        "No phone service",
    )
    internet_service = rng.choice(["DSL", "Fiber optic", "No"], rows, p=[0.35, 0.45, 0.2])

    online_security = np.where(
        internet_service == "No",
        "No internet service",
        rng.choice(["Yes", "No"], rows, p=[0.38, 0.62]),
    )
    tech_support = np.where(
        internet_service == "No",
        "No internet service",
        rng.choice(["Yes", "No"], rows, p=[0.36, 0.64]),
    )
    contract = rng.choice(["Month-to-month", "One year", "Two year"], rows, p=[0.56, 0.24, 0.2])
    paperless_billing = rng.choice(["Yes", "No"], rows, p=[0.59, 0.41])
    payment_method = rng.choice(
        ["Electronic check", "Mailed check", "Bank transfer", "Credit card"],
        rows,
        p=[0.34, 0.22, 0.22, 0.22],
    )

    base_charge = rng.normal(62, 18, rows)
    fiber_addon = np.where(internet_service == "Fiber optic", 24, 0)
    no_internet_discount = np.where(internet_service == "No", -30, 0)
    support_addon = np.where(tech_support == "Yes", 8, 0)
    security_addon = np.where(online_security == "Yes", 7, 0)
    monthly_charges = np.clip(
        base_charge + fiber_addon + no_internet_discount + support_addon + security_addon,
        18,
        125,
    ).round(2)
    total_charges = (monthly_charges * tenure_months * rng.normal(1.0, 0.04, rows)).round(2)

    risk_score = (
        -2.2
        + 1.35 * (contract == "Month-to-month")
        - 0.85 * (contract == "Two year")
        + 0.95 * (internet_service == "Fiber optic")
        + 0.65 * (payment_method == "Electronic check")
        + 0.55 * (paperless_billing == "Yes")
        + 0.5 * (tech_support == "No")
        + 0.45 * (online_security == "No")
        + 0.35 * senior_citizen
        - 0.025 * tenure_months
        + 0.012 * (monthly_charges - 65)
        - 0.25 * (partner == "Yes")
        - 0.18 * (dependents == "Yes")
    )
    churn_probability = sigmoid(risk_score)
    churn = rng.binomial(1, churn_probability)

    return pd.DataFrame(
        {
            "customer_id": [f"CUST-{index:05d}" for index in range(1, rows + 1)],
            "tenure_months": tenure_months,
            "monthly_charges": monthly_charges,
            "total_charges": total_charges,
            "senior_citizen": senior_citizen,
            "partner": partner,
            "dependents": dependents,
            "phone_service": phone_service,
            "multiple_lines": multiple_lines,
            "internet_service": internet_service,
            "online_security": online_security,
            "tech_support": tech_support,
            "contract": contract,
            "paperless_billing": paperless_billing,
            "payment_method": payment_method,
            "churn": churn,
        }
    )


def main() -> None:
    DATA_PATH.parent.mkdir(parents=True, exist_ok=True)
    df = generate_customer_churn_data()
    df.to_csv(DATA_PATH, index=False)
    churn_rate = df["churn"].mean()
    print(f"Created {DATA_PATH} with {len(df):,} rows. Churn rate: {churn_rate:.1%}")


if __name__ == "__main__":
    main()
