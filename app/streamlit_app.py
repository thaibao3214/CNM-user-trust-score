import os
import sys
import json
import pandas as pd
import streamlit as st

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.utils import load_model, calculate_trust_score, get_risk_level


MODEL_PATH = "models/trust_xgb_model.pkl"
FEATURE_COLUMNS_PATH = "models/feature_columns.json"
TEST_DATA_PATH = "data/processed/processed_test.csv"

TARGET = "isFraud"
ID_COL = "TransactionID"


st.set_page_config(
    page_title="Dự đoán độ tin cậy người dùng",
    page_icon="🛡️",
    layout="wide"
)


@st.cache_resource
def load_trust_model():
    return load_model(MODEL_PATH)


@st.cache_data
def load_feature_columns():
    with open(FEATURE_COLUMNS_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


@st.cache_data
def load_sample_data():
    df = pd.read_csv(TEST_DATA_PATH)
    return df


def main():
    st.title("🛡️ Ứng dụng Dự đoán độ tin cậy người dùng")

    st.caption(
        "Đồ án 31 - Môn CÔNG NGHỆ MỚI TRONG PHÁT TRIỂN ỨNG DỤNG"
    )

    st.markdown(
        """
        Hệ thống sử dụng mô hình Machine Learning để dự đoán xác suất rủi ro/gian lận
        của một giao dịch, sau đó chuyển đổi thành **Trust Score**.

        **Công thức:**

        `Trust Score = (1 - Fraud Probability) × 100`
        """
    )

    model = load_trust_model()
    feature_columns = load_feature_columns()
    df = load_sample_data()

    st.sidebar.header("⚙️ Tùy chọn đầu vào")

    mode = st.sidebar.radio(
        "Chọn cách demo",
        [
            "Dự đoán từ giao dịch mẫu",
            "Dự đoán ngẫu nhiên"
        ]
    )

    if mode == "Dự đoán từ giao dịch mẫu":
        max_index = len(df) - 1

        selected_index = st.sidebar.number_input(
            "Chọn index giao dịch",
            min_value=0,
            max_value=max_index,
            value=0,
            step=1
        )

        sample = df.iloc[[selected_index]]

    else:
        sample = df.sample(1, random_state=None)

    transaction_id = (
        int(sample[ID_COL].iloc[0])
        if ID_COL in sample.columns
        else "N/A"
    )

    actual_label = (
        int(sample[TARGET].iloc[0])
        if TARGET in sample.columns
        else "N/A"
    )

    X = sample[feature_columns]

    fraud_probability = model.predict_proba(X)[0][1]
    trust_score = calculate_trust_score(fraud_probability)
    risk_level = get_risk_level(trust_score)

    st.subheader("📌 Thông tin giao dịch")

    col1, col2, col3 = st.columns(3)

    col1.metric("Transaction ID", transaction_id)
    col2.metric("Actual isFraud", actual_label)
    col3.metric("Số lượng feature", len(feature_columns))

    st.subheader("📊 Kết quả dự đoán")

    col1, col2, col3 = st.columns(3)

    col1.metric(
        "Fraud Probability",
        f"{fraud_probability:.4f}"
    )

    col2.metric(
        "Trust Score",
        f"{trust_score}/100"
    )

    col3.metric(
        "Risk Level",
        risk_level
    )

    if trust_score >= 80:
        st.success("✅ Người dùng/giao dịch có độ tin cậy cao.")
    elif trust_score >= 50:
        st.warning("⚠️ Người dùng/giao dịch có độ tin cậy trung bình.")
    else:
        st.error("🚨 Người dùng/giao dịch có độ tin cậy thấp.")

    st.subheader("🔎 Dữ liệu đầu vào của giao dịch")

    display_cols = [
        col for col in [
            ID_COL,
            TARGET,
            "TransactionAmt",
            "ProductCD",
            "card1",
            "card4",
            "card6",
            "DeviceType",
            "DeviceInfo"
        ]
        if col in sample.columns
    ]

    st.dataframe(sample[display_cols], use_container_width=True)

    with st.expander("Xem toàn bộ feature đã đưa vào model"):
        st.dataframe(X, use_container_width=True)


if __name__ == "__main__":
    main()