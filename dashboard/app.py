import streamlit as st
import pandas as pd
import joblib
import shap
import matplotlib.pyplot as plt
import plotly.graph_objects as go

# ==========================================
# PAGE CONFIG
# ==========================================

st.set_page_config(
    page_title="Customer Churn Prediction",
    layout="wide"
)

# ==========================================
# CUSTOM CSS
# ==========================================

st.markdown("""
<style>

.main {
    background-color: #0E1117;
}

.stMetric {
    background-color: #1E1E1E;
    padding: 15px;
    border-radius: 10px;
}

h1, h2, h3 {
    color: white;
}

</style>
""", unsafe_allow_html=True)

# ==========================================
# LOAD MODEL
# ==========================================

model = joblib.load(
    "D:\\customer-churn-ai\\models\\xgboost.pkl"
)

# ==========================================
# SHAP EXPLAINER
# ==========================================

explainer = shap.Explainer(model)

# ==========================================
# HERO SECTION
# ==========================================

st.markdown("""
# 📊 Customer Churn Prediction Dashboard

AI-powered churn prediction system using XGBoost and SHAP Explainable AI.
""")

# ==========================================
# SIDEBAR
# ==========================================

st.sidebar.markdown(
    "## ⚙️ Customer Parameters"
    
)

st.sidebar.markdown("---")

uploaded_file = st.sidebar.file_uploader(
    "Upload CSV File",
    type=["csv"]
)
# ==========================================
# INPUTS
# ==========================================

tenure = st.sidebar.slider(
    "Tenure (Months)",
    0,
    72,
    12
)

monthly_charges = st.sidebar.slider(
    "Monthly Charges",
    0,
    200,
    70
)

total_charges = st.sidebar.slider(
    "Total Charges",
    0,
    10000,
    1000
)

contract = st.sidebar.selectbox(
    "Contract Type",
    [
        "Month-to-month",
        "One year",
        "Two year"
    ]
)

internet_service = st.sidebar.selectbox(
    "Internet Service",
    [
        "DSL",
        "Fiber optic",
        "No"
    ]
)

payment_method = st.sidebar.selectbox(
    "Payment Method",
    [
        "Electronic check",
        "Mailed check",
        "Bank transfer",
        "Credit card"
    ]
)

paperless_billing = st.sidebar.selectbox(
    "Paperless Billing",
    [
        "Yes",
        "No"
    ]
)

online_security = st.sidebar.selectbox(
    "Online Security",
    [
        "Yes",
        "No"
    ]
)

tech_support = st.sidebar.selectbox(
    "Tech Support",
    [
        "Yes",
        "No"
    ]
)

# ==========================================
# CREATE INPUT DATA
# ==========================================

input_data = {
    "tenure": tenure,
    "MonthlyCharges": monthly_charges,
    "TotalCharges": total_charges,

    "Contract_One year":
        1 if contract == "One year" else 0,

    "Contract_Two year":
        1 if contract == "Two year" else 0,

    "InternetService_Fiber optic":
        1 if internet_service == "Fiber optic" else 0,

    "InternetService_No":
        1 if internet_service == "No" else 0,

    "PaymentMethod_Electronic check":
        1 if payment_method == "Electronic check" else 0,

    "PaymentMethod_Mailed check":
        1 if payment_method == "Mailed check" else 0,

    "PaperlessBilling_Yes":
        1 if paperless_billing == "Yes" else 0,

    "OnlineSecurity_Yes":
        1 if online_security == "Yes" else 0,

    "TechSupport_Yes":
        1 if tech_support == "Yes" else 0
}

# ==========================================
# CREATE DATAFRAME
# ==========================================

input_df = pd.DataFrame([input_data])

# ==========================================
# ADD MISSING COLUMNS
# ==========================================

model_columns = model.get_booster().feature_names

for col in model_columns:

    if col not in input_df.columns:
        input_df[col] = 0

# ==========================================
# REORDER COLUMNS
# ==========================================

input_df = input_df[model_columns]

# ==========================================
# CUSTOMER SUMMARY
# ==========================================

st.subheader("Customer Summary")

summary = f"""
- Tenure: {tenure} months
- Monthly Charges: ${monthly_charges}
- Total Charges: ${total_charges}
- Contract: {contract}
- Internet Service: {internet_service}
"""

st.info(summary)

# ==========================================
# SHOW INPUT DATA
# ==========================================

st.subheader("Input Data")

st.dataframe(input_df)

# ==========================================
# PREDICT BUTTON
# ==========================================

if st.button("Predict Churn"):

    # ==========================================
    # PREDICTION
    # ==========================================

    prediction = model.predict(input_df)[0]

    probability = model.predict_proba(
        input_df
    )[0][1]

    # ==========================================
    # KPI SECTION
    # ==========================================

    st.subheader("Prediction Dashboard")

    kpi1, kpi2, kpi3 = st.columns(3)

    # ==========================================
    # KPI 1
    # ==========================================

    with kpi1:

        st.metric(
            "Churn Probability",
            f"{probability:.2%}"
        )

    # ==========================================
    # KPI 2
    # ==========================================

    with kpi2:

        st.metric(
            "Prediction",
            "Churn" if prediction == 1 else "Stay"
        )

    # ==========================================
    # KPI 3
    # ==========================================

    with kpi3:

        if probability < 0.3:
            risk = "Low"

        elif probability < 0.7:
            risk = "Medium"

        else:
            risk = "High"

        st.metric(
            "Risk Level",
            risk
        )

    # ==========================================
    # RESULT MESSAGE
    # ==========================================

    if prediction == 1:

        st.error(
            "⚠️ Customer is likely to churn"
        )

    else:

        st.success(
            "✅ Customer is likely to stay"
        )

    # ==========================================
    # RISK SCORE
    # ==========================================

    st.subheader("Risk Score")

    st.progress(float(probability))

    # ==========================================
    # GAUGE CHART
    # ==========================================

    fig_gauge = go.Figure(go.Indicator(
        mode="gauge+number",
        value=probability * 100,

        title={
            "text": "Churn Risk"
        },

        gauge={
            "axis": {
                "range": [0, 100]
            },

            "bar": {
                "color": "red"
            },

            "steps": [
                {
                    "range": [0, 30],
                    "color": "green"
                },
                {
                    "range": [30, 70],
                    "color": "yellow"
                },
                {
                    "range": [70, 100],
                    "color": "red"
                }
            ]
        }
    ))

    st.plotly_chart(
        fig_gauge,
        use_container_width=True
    )

    # ==========================================
    # SHAP VALUES
    # ==========================================

    shap_values = explainer(input_df)

    # ==========================================
    # SHAP DATAFRAME
    # ==========================================

    shap_df = pd.DataFrame({
        "Feature": input_df.columns,
        "SHAP Value": shap_values.values[0]
    })

    shap_df["ABS"] = (
        shap_df["SHAP Value"].abs()
    )

    shap_df = shap_df.sort_values(
        by="ABS",
        ascending=False
    )

    # ==========================================
    # TABS
    # ==========================================

    tab1, tab2, tab3 = st.tabs([
        "Prediction",
        "SHAP Analysis",
        "Business Insights"
    ])

    # ==========================================
    # TAB 1
    # ==========================================

    with tab1:

        st.subheader(
            "Top Factors Affecting Prediction"
        )

        top10 = shap_df.head(10)

        fig_bar = go.Figure()

        fig_bar.add_bar(
            x=top10["SHAP Value"],
            y=top10["Feature"],
            orientation="h"
        )

        st.plotly_chart(
            fig_bar,
            use_container_width=True
        )

    # ==========================================
    # TAB 2
    # ==========================================

    with tab2:

        st.subheader("SHAP Waterfall Plot")

        fig, ax = plt.subplots(
            figsize=(10, 5)
        )

        shap.plots.waterfall(
            shap_values[0],
            show=False
        )

        st.pyplot(fig)

    # ==========================================
    # TAB 3
    # ==========================================

    with tab3:

        st.subheader(
            "Business Explanation"
        )

        top_features = shap_df.head(3)

        for _, row in top_features.iterrows():

            feature = row["Feature"]

            value = row["SHAP Value"]

            if value > 0:

                st.write(
                    f"🔺 {feature} increases churn risk"
                )

            else:

                st.write(
                    f"🔻 {feature} decreases churn risk"
                )

        st.markdown("""
        ### 📌 Business Insights

        - Customers with short tenure are more likely to churn.
        - Long-term contracts significantly reduce churn risk.
        - High monthly charges increase customer dissatisfaction.
        - Electronic check payment users show higher churn tendency.
        - SHAP Explainable AI helps identify the most influential features.
        """)

# ==========================================
# BATCH PREDICTION
# ==========================================

st.markdown("---")

st.header("📂 Batch Prediction")

if uploaded_file is not None:

    # ==========================================
    # READ CSV
    # ==========================================

    batch_df = pd.read_csv(uploaded_file)

    st.subheader("Uploaded Dataset")

    st.dataframe(batch_df.head())

    # ==========================================
    # PREPROCESSING
    # ==========================================

    if "TotalCharges" in batch_df.columns:

        batch_df["TotalCharges"] = pd.to_numeric(
            batch_df["TotalCharges"],
            errors="coerce"
        )

        batch_df["TotalCharges"] = (
            batch_df["TotalCharges"]
            .fillna(
                batch_df["TotalCharges"].median()
            )
        )

    # ==========================================
    # ENCODING
    # ==========================================

    batch_encoded = pd.get_dummies(batch_df)

    # ==========================================
    # ALIGN COLUMNS
    # ==========================================

    for col in model_columns:

        if col not in batch_encoded.columns:
            batch_encoded[col] = 0

    batch_encoded = batch_encoded[
        model_columns
    ]

    # ==========================================
    # PREDICTIONS
    # ==========================================

    batch_predictions = model.predict(
        batch_encoded
    )

    batch_probabilities = (
        model.predict_proba(batch_encoded)[:, 1]
    )

    # ==========================================
    # RESULTS
    # ==========================================

    batch_df["Prediction"] = (
        batch_predictions
    )

    batch_df["Churn Probability"] = (
        batch_probabilities
    )

    batch_df["Prediction"] = (
        batch_df["Prediction"]
        .map({
            1: "Churn",
            0: "Stay"
        })
    )

    # ==========================================
    # SHOW RESULTS
    # ==========================================

    st.subheader(
        "Batch Prediction Results"
    )

    st.dataframe(batch_df.head(20))

    # ==========================================
    # PIE CHART
    # ==========================================

    churn_count = (
        batch_df["Prediction"]
        .value_counts()
    )

    fig_pie = go.Figure(
        data=[
            go.Pie(
                labels=churn_count.index,
                values=churn_count.values,
                hole=0.4
            )
        ]
    )

    st.plotly_chart(
        fig_pie,
        use_container_width=True
    )

    # ==========================================
    # HIGH RISK CUSTOMERS
    # ==========================================

    high_risk = batch_df[
        batch_df["Churn Probability"] > 0.7
    ]

    st.subheader(
        "🚨 High Risk Customers"
    )

    st.dataframe(high_risk)

    # ==========================================
    # DOWNLOAD BUTTON
    # ==========================================

    csv = batch_df.to_csv(
        index=False
    ).encode("utf-8")

    st.download_button(
        label="📥 Download Results",
        data=csv,
        file_name="batch_predictions.csv",
        mime="text/csv"
    )