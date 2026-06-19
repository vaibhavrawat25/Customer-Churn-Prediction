# Customer Churn Prediction

A working machine learning project that predicts whether a customer is likely to leave a subscription service. It includes sample data generation, model training, evaluation reports, command-line prediction, and an interactive Streamlit dashboard.

## Project Structure

```text
.
├── app.py                    # Streamlit dashboard
├── data/                     # Generated sample dataset
├── models/                   # Trained model artifact
├── reports/                  # Evaluation metrics and charts
├── src/
│   ├── generate_data.py      # Creates realistic synthetic churn data
│   ├── predict.py            # CLI prediction for one customer
│   └── train_model.py        # Trains and evaluates the classifier
└── tests/
    └── test_pipeline.py      # Smoke tests for the ML pipeline
```

## Quick Start

Install dependencies:

```bash
python3 -m pip install -r requirements.txt
```

Generate the sample dataset:

```bash
python3 src/generate_data.py
```

Train the model:

```bash
python3 src/train_model.py
```

Run the dashboard:

```bash
streamlit run app.py
```

Make a single prediction from the terminal:

```bash
python3 src/predict.py \
  --tenure-months 8 \
  --monthly-charges 92 \
  --contract Month-to-month \
  --internet-service Fiber optic \
  --tech-support No \
  --online-security No \
  --payment-method "Electronic check" \
  --paperless-billing Yes \
  --senior-citizen 1 \
  --partner No \
  --dependents No \
  --phone-service Yes \
  --multiple-lines Yes
```

## What the Model Learns

The classifier analyzes behavior and account attributes such as tenure, monthly charges, contract type, support services, security add-ons, payment method, and paperless billing. The training script saves:

- `models/churn_model.joblib`
- `reports/metrics.json`
- `reports/confusion_matrix.csv`
- `reports/feature_importance.csv`

## Notes

The included dataset is synthetic and meant for learning or portfolio use. Replace `data/customer_churn.csv` with a real churn dataset when available, keeping the same target column name: `churn`.
