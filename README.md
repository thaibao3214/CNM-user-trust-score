# 🛡️ 31. Ứng dụng Dự đoán độ tin cậy người dùng

## 📚 Môn học
CÔNG NGHỆ MỚI TRONG PHÁT TRIỂN ỨNG DỤNG

---

# 📌 Giới thiệu

Dự án xây dựng hệ thống Machine Learning nhằm dự đoán độ tin cậy của người dùng dựa trên dữ liệu giao dịch trực tuyến.

Mô hình sử dụng XGBoost để dự đoán xác suất gian lận (Fraud Probability), sau đó chuyển đổi thành:

```text
Trust Score = (1 - Fraud Probability) × 100
```

Dựa trên Trust Score, hệ thống phân loại:

- Độ tin cậy cao
- Độ tin cậy trung bình
- Độ tin cậy thấp

---

# 📂 Dataset

Dataset sử dụng:

```text
IEEE-CIS Fraud Detection
```

Nguồn:

```text
https://www.kaggle.com/competitions/ieee-fraud-detection
```

Các file chính sử dụng:

```text
train_transaction.csv
train_identity.csv
```

---

# 🧠 Công nghệ sử dụng

- Python
- Pandas
- NumPy
- Scikit-learn
- XGBoost
- Streamlit
- Git/GitHub

---

# 📁 Cấu trúc project

```text
CNM_project/
│
├── app/
│   └── streamlit_app.py
│
├── data/
│   ├── raw/
│   └── processed/
│
├── models/
│   ├── feature_columns.json
│   ├── training_metrics.json
│   └── trust_xgb_model.pkl
│
├── notebooks/
│   └── eda.ipynb
│
├── src/
│   ├── preprocess.py
│   ├── train.py
│   ├── predict.py
│   ├── evaluate.py
│   ├── test_utils.py
│   └── utils.py
│
├── requirements.txt
├── README.md
└── .gitignore
```

---

# 🚀 Hướng dẫn chạy project

## 1. Clone repository

```bash
git clone https://github.com/thaibao3214/CNM-user-trust-score.git
```

```bash
cd CNM-user-trust-score
```

---

## 2. Tạo virtual environment

### Windows

```bash
python -m venv venv
```

```bash
venv\Scripts\activate
```

### Linux / MacOS

```bash
python3 -m venv venv
```

```bash
source venv/bin/activate
```

---

## 3. Cài đặt thư viện

```bash
pip install -r requirements.txt
```

---

# 📥 Chuẩn bị dữ liệu

Tải dataset tại:

```text
https://www.kaggle.com/competitions/ieee-fraud-detection
```

Sau đó đặt các file:

```text
train_transaction.csv
train_identity.csv
```

vào thư mục:

```text
data/raw/
```

---

# ⚙️ Tiền xử lý dữ liệu

Chạy:

```bash
python src/preprocess.py
```

Kết quả:

```text
data/processed/
│
├── processed_train.csv
├── processed_test.csv
├── preprocessing_metadata.json
└── label_encoders.json
```

---

# 🧠 Train model

Chạy:

```bash
python -m src.train
```

Kết quả:

```text
models/
│
├── trust_xgb_model.pkl
├── feature_columns.json
└── training_metrics.json
```

---

# 🔍 Predict Trust Score

Chạy:

```bash
python -m src.predict
```

Kết quả được lưu tại:

```text
data/predictions.csv
```

Ví dụ output:

| TransactionID | Fraud Probability | Trust Score | Risk Level |
|---|---|---|---|
| 3303078 | 0.0846 | 91.54 | Độ tin cậy cao |

---

# 🌐 Chạy ứng dụng Streamlit

Chạy:

```bash
streamlit run app/streamlit_app.py
```

Sau đó mở trình duyệt:

```text
http://localhost:8501
```

Ứng dụng sẽ:
- load model đã train
- chọn giao dịch mẫu
- dự đoán Fraud Probability
- tính Trust Score
- phân loại Risk Level

---

# 📊 Kết quả mô hình

| Metric | Giá trị |
|---|---|
| ROC-AUC | ~0.92 |
| PR-AUC | ~0.60 |
| Recall Fraud | ~0.80 |

Mô hình ưu tiên phát hiện các giao dịch có rủi ro cao, phù hợp với bài toán đánh giá độ tin cậy người dùng.

---

# 🛠️ MLOps trong dự án

Dự án áp dụng các nguyên tắc MLOps cơ bản:

- Structured ML Pipeline
- Data Processing Pipeline
- Model Artifact Management
- Prediction Pipeline
- Virtual Environment
- Requirements Management
- Streamlit Deployment
- Git/GitHub Version Control

Pipeline được tách thành các bước:

```text
Raw Data
→ Preprocessing
→ Training
→ Prediction
→ Streamlit Application
```

---

# 🖥️ Cấu hình tối ưu

Project được tối ưu để chạy trên:

```text
Windows 10
Intel Core i5-6200U
RAM 8GB
64-bit OS
```

Các chiến lược tối ưu:
- giảm memory usage
- không lưu processed_full.csv
- drop cột missing nhiều
- downcast float/int
- XGBoost cấu hình vừa phải

---

# 👨‍💻 Tác giả

```text

```
