import os
import pandas as pd

from src.utils import (
    load_model,
    calculate_trust_score,
    get_risk_level
)

MODEL_PATH = "models/trust_xgb_model.pkl"
TEST_PATH = "data/processed/processed_test.csv"
OUTPUT_PATH = "data/predictions.csv"

TARGET = "isFraud"
ID_COL = "TransactionID"


def main():
    print("=" * 60)
    print("PREDICT USER TRUST SCORE")
    print("31. Ứng dụng Dự đoán độ tin cậy người dùng")
    print("=" * 60)

    print("\n[1] Loading model...")
    model = load_model(MODEL_PATH)

    print("\n[2] Loading test data...")
    df = pd.read_csv(TEST_PATH)

    print("Dataset:", df.shape)

    print("\n[3] Preparing features...")

    feature_cols = [
        col for col in df.columns
        if col not in [TARGET, ID_COL]
    ]

    X = df[feature_cols]

    print("Features:", X.shape)

    print("\n[4] Predicting fraud probability...")

    fraud_prob = model.predict_proba(X)[:, 1]

    print("\n[5] Creating trust score...")

    results = pd.DataFrame()

    if ID_COL in df.columns:
        results[ID_COL] = df[ID_COL]

    if TARGET in df.columns:
        results["Actual_isFraud"] = df[TARGET]

    results["Fraud_Probability"] = fraud_prob

    results["Trust_Score"] = results["Fraud_Probability"].apply(
        calculate_trust_score
    )

    results["Risk_Level"] = results["Trust_Score"].apply(
        get_risk_level
    )

    print("\n[6] Sample Results:\n")
    print(results.head(10))

    print("\n[7] Saving prediction results...")

    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    results.to_csv(OUTPUT_PATH, index=False)

    print(f"Saved: {OUTPUT_PATH}")
    print("\n✅ PREDICTION COMPLETED SUCCESSFULLY!")


if __name__ == "__main__":
    main()