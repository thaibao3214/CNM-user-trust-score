import os
import gc
import json
import pandas as pd
from xgboost import XGBClassifier
from sklearn.metrics import classification_report, roc_auc_score, average_precision_score
from src.utils import save_model, save_json

PROCESSED_DATA_DIR = "data/processed"
MODEL_DIR = "models"

TRAIN_PATH = os.path.join(PROCESSED_DATA_DIR, "processed_train.csv")
TEST_PATH = os.path.join(PROCESSED_DATA_DIR, "processed_test.csv")

TARGET = "isFraud"
ID_COL = "TransactionID"


def main():
    print("=" * 60)
    print("TRAIN MODEL - 31. Ứng dụng Dự đoán độ tin cậy người dùng")
    print("=" * 60)

    print("\n[1] Loading processed data...")
    train_df = pd.read_csv(TRAIN_PATH)
    test_df = pd.read_csv(TEST_PATH)

    print("Train:", train_df.shape)
    print("Test:", test_df.shape)

    drop_cols = [TARGET]
    if ID_COL in train_df.columns:
        drop_cols.append(ID_COL)

    X_train = train_df.drop(columns=drop_cols)
    y_train = train_df[TARGET]

    X_test = test_df.drop(columns=drop_cols)
    y_test = test_df[TARGET]

    del train_df, test_df
    gc.collect()

    print("\n[2] Class imbalance...")
    neg = (y_train == 0).sum()
    pos = (y_train == 1).sum()
    scale_pos_weight = neg / pos

    print("Non-fraud:", neg)
    print("Fraud:", pos)
    print("scale_pos_weight:", round(scale_pos_weight, 2))

    print("\n[3] Training XGBoost model...")

    model = XGBClassifier(
        n_estimators=150,
        max_depth=5,
        learning_rate=0.08,
        subsample=0.8,
        colsample_bytree=0.8,
        scale_pos_weight=scale_pos_weight,
        objective="binary:logistic",
        eval_metric="aucpr",
        tree_method="hist",
        random_state=42,
        n_jobs=2
    )

    model.fit(X_train, y_train)

    print("\n[4] Evaluating model...")

    y_prob = model.predict_proba(X_test)[:, 1]
    y_pred = (y_prob >= 0.5).astype(int)

    roc_auc = roc_auc_score(y_test, y_prob)
    pr_auc = average_precision_score(y_test, y_prob)

    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))

    print("ROC-AUC:", roc_auc)
    print("PR-AUC:", pr_auc)

    print("\n[5] Saving model and metadata...")

    os.makedirs(MODEL_DIR, exist_ok=True)

    model_path = os.path.join(MODEL_DIR, "trust_xgb_model.pkl")
    save_model(model, model_path)

    feature_columns = X_train.columns.tolist()

    save_json(
        feature_columns,
        os.path.join(MODEL_DIR, "feature_columns.json")
    )

    metrics = {
        "project": "31. Ứng dụng Dự đoán độ tin cậy người dùng",
        "model": "XGBoost",
        "roc_auc": float(roc_auc),
        "pr_auc": float(pr_auc),
        "scale_pos_weight": float(scale_pos_weight),
        "threshold": 0.5,
        "trust_score_formula": "Trust Score = (1 - fraud_probability) * 100"
    }

    save_json(
        metrics,
        os.path.join(MODEL_DIR, "training_metrics.json")
    )

    print("\n✅ TRAINING COMPLETED SUCCESSFULLY!")


if __name__ == "__main__":
    main()