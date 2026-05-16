import os
import gc
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

from utils import reduce_mem_usage, save_json, ensure_dir

PROJECT_NAME = "31. Ứng dụng Dự đoán độ tin cậy người dùng"
COURSE_NAME = "CÔNG NGHỆ MỚI TRONG PHÁT TRIỂN ỨNG DỤNG"

RAW_DATA_DIR = "data/raw"
PROCESSED_DATA_DIR = "data/processed"

TRAIN_TRANSACTION_PATH = os.path.join(RAW_DATA_DIR, "train_transaction.csv")
TRAIN_IDENTITY_PATH = os.path.join(RAW_DATA_DIR, "train_identity.csv")

TARGET = "isFraud"
ID_COL = "TransactionID"

MISSING_THRESHOLD = 0.90
TEST_SIZE = 0.2
RANDOM_STATE = 42


def main():
    print("=" * 60)
    print(PROJECT_NAME)
    print(COURSE_NAME)
    print("=" * 60)

    ensure_dir(PROCESSED_DATA_DIR)

    print("\n[1] Loading raw data...")
    train_transaction = pd.read_csv(TRAIN_TRANSACTION_PATH)
    train_identity = pd.read_csv(TRAIN_IDENTITY_PATH)

    print("train_transaction:", train_transaction.shape)
    print("train_identity:", train_identity.shape)

    print("\n[2] Reducing memory before merge...")
    train_transaction = reduce_mem_usage(train_transaction)
    train_identity = reduce_mem_usage(train_identity)

    print("\n[3] Merging data...")
    df = train_transaction.merge(
        train_identity,
        on=ID_COL,
        how="left"
    )

    del train_transaction, train_identity
    gc.collect()

    print("Merged dataset:", df.shape)

    print("\n[4] Dropping columns missing > 90%...")
    missing_ratio = df.isnull().mean()

    drop_cols = missing_ratio[
        missing_ratio > MISSING_THRESHOLD
    ].index.tolist()

    drop_cols = [
        col for col in drop_cols
        if col not in [TARGET, ID_COL]
    ]

    df.drop(columns=drop_cols, inplace=True)
    
    print("Dropped columns:", len(drop_cols))
    print("After dropping:", df.shape)

    print("\n[5] Encoding categorical columns...")
    categorical_cols = df.select_dtypes(include=["object"]).columns.tolist()
    label_encoders = {}

    for col in categorical_cols:
        df[col] = df[col].astype(str)

        encoder = LabelEncoder()
        df[col] = encoder.fit_transform(df[col])

        label_encoders[col] = encoder.classes_.tolist()

    print("Categorical columns:", len(categorical_cols))

    print("\n[6] Filling missing values...")
    feature_cols = [col for col in df.columns if col != TARGET]
    fill_values = {}

    for col in feature_cols:
        if df[col].isnull().sum() > 0:
            median_value = df[col].median()
            df[col] = df[col].fillna(median_value)
            fill_values[col] = float(median_value)

    print("Remaining missing:", df.isnull().sum().sum())

    print("\n[7] Reducing memory after processing...")
    before_mem = df.memory_usage().sum() / 1024**2
    df = reduce_mem_usage(df)
    after_mem = df.memory_usage().sum() / 1024**2

    print(f"Memory: {before_mem:.2f} MB -> {after_mem:.2f} MB")

    print("\n[8] Splitting train/test...")
    train_df, test_df = train_test_split(
        df,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=df[TARGET]
    )

    del df
    gc.collect()

    print("Train:", train_df.shape)
    print("Test:", test_df.shape)

    print("\n[9] Saving processed files...")
    train_path = os.path.join(PROCESSED_DATA_DIR, "processed_train.csv")
    test_path = os.path.join(PROCESSED_DATA_DIR, "processed_test.csv")

    train_df.to_csv(train_path, index=False)
    test_df.to_csv(test_path, index=False)

    print("Saved:", train_path)
    print("Saved:", test_path)

    print("\n[10] Saving metadata...")
    metadata = {
        "project_name": PROJECT_NAME,
        "course_name": COURSE_NAME,
        "dataset": "IEEE-CIS Fraud Detection",
        "target": TARGET,
        "id_column": ID_COL,
        "missing_threshold": MISSING_THRESHOLD,
        "test_size": TEST_SIZE,
        "random_state": RANDOM_STATE,
        "dropped_columns": drop_cols,
        "categorical_columns": categorical_cols,
        "fill_values": fill_values,
        "train_shape": list(train_df.shape),
        "test_shape": list(test_df.shape),
        "note": "Optimized for local machine with 8GB RAM. processed_full.csv is not saved."
    }

    save_json(
        metadata,
        os.path.join(PROCESSED_DATA_DIR, "preprocessing_metadata.json")
    )

    save_json(
        label_encoders,
        os.path.join(PROCESSED_DATA_DIR, "label_encoders.json")
    )

    print("\n✅ PREPROCESSING COMPLETED SUCCESSFULLY!")


if __name__ == "__main__":
    main()