# 📊 Hệ Thống Dự Đoán Customer Churn

Hệ thống dự đoán khách hàng rời bỏ dịch vụ (Customer Churn Prediction) sử dụng XGBoost và SHAP Explainable AI.

---

## 🚀 Demo

Ứng dụng được xây dựng bằng Streamlit với khả năng:

- Dự đoán churn theo thời gian thực
- Giải thích dự đoán bằng SHAP Explainable AI
- Batch prediction bằng file CSV
- Dashboard trực quan
- Phân tích business insights

---

## 🛠️ Công Nghệ Sử Dụng

### Machine Learning
- Python
- Scikit-learn
- XGBoost
- SHAP

### Data Processing
- Pandas
- NumPy

### Visualization
- Matplotlib
- Plotly

### Deployment
- Streamlit

---

## 📂 Cấu Trúc Project

```text
customer-churn-ai/
│
├── data/
├── processed/
├── models/
├── notebooks/
├── dashboard/
│   └── app.py
│
├── requirements.txt
└── README.md
```

---

## 📈 Quy Trình Machine Learning

### 1. Data Cleaning
- Xử lý missing values
- Chuyển đổi kiểu dữ liệu
- Chuẩn hóa dữ liệu

### 2. Exploratory Data Analysis (EDA)
- Phân tích churn distribution
- Correlation analysis
- Feature importance exploration

### 3. Feature Engineering
- One-hot encoding
- Tạo feature mới
- Feature selection

### 4. Model Training
Đã thử nghiệm nhiều mô hình:

- Logistic Regression
- Random Forest
- XGBoost

Trong đó XGBoost cho kết quả tốt nhất.

### 5. Explainable AI
Sử dụng SHAP để:

- Giải thích prediction
- Hiển thị feature importance
- Phân tích nguyên nhân churn

### 6. Dashboard Development
Xây dựng dashboard với:

- Single prediction
- Batch prediction
- SHAP visualization
- Risk analytics
- Download prediction results

---

## 📊 Kết Quả Mô Hình

| Metric | Score |
|---|---|
| Accuracy | 0.xx |
| Precision | 0.xx |
| Recall | 0.xx |
| F1 Score | 0.xx |

---

## 🧠 Business Insights

### Các yếu tố làm tăng churn risk
- Khách hàng có tenure thấp
- Hợp đồng ngắn hạn
- Monthly charges cao
- Thanh toán bằng Electronic Check

### Các yếu tố giúp giảm churn risk
- Hợp đồng dài hạn
- Khách hàng sử dụng dịch vụ lâu năm
- Chi phí hàng tháng ổn định

---

## 💡 Tính Năng Chính

### ✅ Single Prediction
Dự đoán churn cho từng khách hàng riêng lẻ.

### ✅ Batch Prediction
Upload file CSV để dự đoán hàng loạt khách hàng.

### ✅ Explainable AI
Giải thích trực quan bằng SHAP:

- Waterfall Plot
- Feature Importance
- Business Explanation

### ✅ Modern Dashboard
Dashboard hiện đại với:

- KPI Cards
- Gauge Chart
- Risk Score
- Interactive Charts

---

## ▶️ Chạy Project

### 1. Clone repository

```bash
git clone https://github.com/yourname/customer-churn-ai.git
```

### 2. Cài thư viện

```bash
pip install -r requirements.txt
```

### 3. Chạy Streamlit App

```bash
streamlit run dashboard/app.py
```

---

## 📌 Hướng Phát Triển Trong Tương Lai

- Hyperparameter tuning
- SMOTE balancing
- FastAPI backend
- Docker deployment
- CI/CD pipeline
- Database integration
- Real-time prediction API

---

## 👨‍💻 Tác Giả

Developed by Nguyễn Phước An