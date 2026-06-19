from __future__ import annotations

import json
from pathlib import Path

import joblib
import pandas as pd
import streamlit as st


DATA_PATH = Path("data/customer_churn.csv")
MODEL_PATH = Path("models/churn_model.joblib")
METRICS_PATH = Path("reports/metrics.json")
FEATURES_PATH = Path("reports/feature_importance.csv")


@st.cache_data
def load_data() -> pd.DataFrame:
    return pd.read_csv(DATA_PATH)


@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)


def require_project_outputs() -> None:
    missing = [path for path in [DATA_PATH, MODEL_PATH, METRICS_PATH] if not path.exists()]
    if missing:
        st.error("Project outputs are missing. Run `python3 src/generate_data.py` and `python3 src/train_model.py` first.")
        st.stop()


def customer_input_form() -> pd.DataFrame:
    with st.sidebar:
        st.header("Customer Profile")
        tenure_months = st.slider("Tenure months", 1, 72, 10)
        monthly_charges = st.slider("Monthly charges", 18.0, 125.0, 75.0, 1.0)
        contract = st.selectbox("Contract", ["Month-to-month", "One year", "Two year"])
        internet_service = st.selectbox("Internet service", ["DSL", "Fiber optic", "No"])

        no_internet = internet_service == "No"
        online_security = st.selectbox(
            "Online security",
            ["No internet service"] if no_internet else ["No", "Yes"],
        )
        tech_support = st.selectbox(
            "Tech support",
            ["No internet service"] if no_internet else ["No", "Yes"],
        )
        payment_method = st.selectbox(
            "Payment method",
            ["Electronic check", "Mailed check", "Bank transfer", "Credit card"],
        )
        paperless_billing = st.selectbox("Paperless billing", ["Yes", "No"])
        senior_citizen = st.toggle("Senior citizen")
        partner = st.selectbox("Partner", ["No", "Yes"])
        dependents = st.selectbox("Dependents", ["No", "Yes"])
        phone_service = st.selectbox("Phone service", ["Yes", "No"])
        multiple_lines = st.selectbox(
            "Multiple lines",
            ["No phone service"] if phone_service == "No" else ["No", "Yes"],
        )

    return pd.DataFrame(
        [
            {
                "tenure_months": tenure_months,
                "monthly_charges": monthly_charges,
                "total_charges": round(monthly_charges * tenure_months, 2),
                "senior_citizen": int(senior_citizen),
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
            }
        ]
    )


def main() -> None:
    st.set_page_config(page_title="Customer Churn Prediction", page_icon="chart_with_downwards_trend", layout="wide")
    require_project_outputs()

    df = load_data()
    model = load_model()
    customer = customer_input_form()
    churn_probability = model.predict_proba(customer)[0, 1]
    prediction = "Likely to Churn" if churn_probability >= 0.5 else "Likely to Stay"

    st.title("Customer Churn Prediction")
    st.caption("Classification model for identifying customers at risk of leaving.")

    metric_columns = st.columns(4)
    metrics = json.loads(METRICS_PATH.read_text(encoding="utf-8"))
    metric_columns[0].metric("Accuracy", f"{metrics['accuracy']:.1%}")
    metric_columns[1].metric("Precision", f"{metrics['precision']:.1%}")
    metric_columns[2].metric("Recall", f"{metrics['recall']:.1%}")
    metric_columns[3].metric("ROC AUC", f"{metrics['roc_auc']:.1%}")

    left, right = st.columns([1, 1])
    with left:
        st.subheader("Live Prediction")
        st.metric(prediction, f"{churn_probability:.1%}")
        st.progress(float(churn_probability))
        st.dataframe(customer, use_container_width=True, hide_index=True)

    with right:
        st.subheader("Dataset Overview")
        st.metric("Customers", f"{len(df):,}")
        st.metric("Observed churn rate", f"{df['churn'].mean():.1%}")
        st.bar_chart(df["churn"].map({0: "Stay", 1: "Churn"}).value_counts())

    st.subheader("Key Churn Drivers")
    feature_importance = pd.read_csv(FEATURES_PATH).head(12)
    feature_importance["feature"] = (
        feature_importance["feature"]
        .str.replace("numeric__", "", regex=False)
        .str.replace("categorical__", "", regex=False)
    )
    st.bar_chart(feature_importance.set_index("feature")["importance"])

    st.subheader("Sample Customers")
    st.dataframe(df.head(25), use_container_width=True, hide_index=True)


if __name__ == "__main__":
    main()
