from __future__ import annotations

import argparse
from pathlib import Path

import joblib
import pandas as pd


MODEL_PATH = Path("models/churn_model.joblib")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Predict customer churn risk.")
    parser.add_argument("--tenure-months", type=int, required=True)
    parser.add_argument("--monthly-charges", type=float, required=True)
    parser.add_argument("--contract", choices=["Month-to-month", "One year", "Two year"], required=True)
    parser.add_argument("--internet-service", choices=["DSL", "Fiber optic", "No"], required=True)
    parser.add_argument(
        "--tech-support",
        choices=["Yes", "No", "No internet service"],
        required=True,
    )
    parser.add_argument(
        "--online-security",
        choices=["Yes", "No", "No internet service"],
        required=True,
    )
    parser.add_argument(
        "--payment-method",
        choices=["Electronic check", "Mailed check", "Bank transfer", "Credit card"],
        required=True,
    )
    parser.add_argument("--paperless-billing", choices=["Yes", "No"], required=True)
    parser.add_argument("--senior-citizen", type=int, choices=[0, 1], required=True)
    parser.add_argument("--partner", choices=["Yes", "No"], required=True)
    parser.add_argument("--dependents", choices=["Yes", "No"], required=True)
    parser.add_argument("--phone-service", choices=["Yes", "No"], required=True)
    parser.add_argument("--multiple-lines", choices=["Yes", "No", "No phone service"], required=True)
    return parser.parse_args()


def build_customer_record(args: argparse.Namespace) -> pd.DataFrame:
    total_charges = round(args.monthly_charges * args.tenure_months, 2)
    return pd.DataFrame(
        [
            {
                "tenure_months": args.tenure_months,
                "monthly_charges": args.monthly_charges,
                "total_charges": total_charges,
                "senior_citizen": args.senior_citizen,
                "partner": args.partner,
                "dependents": args.dependents,
                "phone_service": args.phone_service,
                "multiple_lines": args.multiple_lines,
                "internet_service": args.internet_service,
                "online_security": args.online_security,
                "tech_support": args.tech_support,
                "contract": args.contract,
                "paperless_billing": args.paperless_billing,
                "payment_method": args.payment_method,
            }
        ]
    )


def main() -> None:
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"{MODEL_PATH} does not exist. Run `python3 src/train_model.py` first."
        )

    args = parse_args()
    model = joblib.load(MODEL_PATH)
    customer = build_customer_record(args)
    churn_probability = model.predict_proba(customer)[0, 1]
    prediction = "Churn" if churn_probability >= 0.5 else "Stay"

    print(f"Prediction: {prediction}")
    print(f"Churn probability: {churn_probability:.1%}")


if __name__ == "__main__":
    main()
