# ⚡ Telco Customer Churn Prediction & Retention Intelligence

A production-ready **Streamlit** web application and Machine Learning decision platform built to predict telecom customer churn risk, identify churn drivers, and generate prescriptive retention actions.

---

## 🚀 Quick Start

### 1. Launch with One Click (Windows)
Double-click `run_app.bat` or run:
```bash
streamlit run app.py
```
Then open your browser at **`http://localhost:8501`**.

---

## 🛠️ Key Features

### 🎯 1. Single Customer Risk Predictor & What-If Simulator
- **Interactive Form:** Demographics, Contract types, Financial profiles, and 8 Subscribed Services.
- **Dynamic Feature Engineering:** Auto-calculates `num_services`, `avg_monthly_spend`, `has_support`, and `tenure_group`.
- **Plotly Churn Probability Gauge:** Visual gauge indicating Safe (Green), Elevated Risk (Amber), and Critical Risk (Red).
- **Personalized Risk Driver Breakdown:** Pinpoints specific high-risk factors for the customer (e.g. Month-to-month contract, Fiber optic without tech support, Electronic check).
- **AI-Powered Prescriptive Retention Playbook:** Provides actionable interventions to retain the customer before they churn.
- **1-Click Demo Presets:** Test scenarios instantly (🚨 High Risk Churner, 🛡️ Loyal Advocate, ⚖️ At-Risk Borderline).

### 📁 2. Batch Portfolio Risk Scoring
- Upload any custom customer CSV or score the full 7,043 customer Telco portfolio.
- Executive KPI Metrics: Flagged Churners Count, Churn Rate %, Monthly Revenue at Risk ($ MRR), and Annual Revenue at Risk ($ ARR).
- Interactive Customer Risk Distribution Charts (Donut chart & Probability histogram).
- Filterable high-priority customer worklist sorted by attrition risk.
- **1-Click Download:** Export the scored portfolio with churn probabilities and risk categories to CSV.

### 📊 3. Exploratory Data Analysis & Churn Drivers
- Empirical churn rates by Contract type (Month-to-month: 42.7% vs Two-year: 2.8%).
- Internet service analysis (Fiber Optic churn paradox).
- Tenure distribution analysis & the First-Year Danger Zone.
- Monthly charges vs Churn boxplots.

### 🧠 4. Model Architecture & Decision Threshold Economics
- Explains the complete pipeline: **ColumnTransformer (StandardScaler + OneHotEncoder) + XGBoost Classifier**.
- **Business Cost Economics:** Explains why the threshold was tuned to **0.19** instead of default 0.50 (cost of missed churner = $500 vs cost of retention offer = $50).
- Interactive Cost Ratio Simulator to see how threshold changes with business unit economics.

---

## 📦 Project Structure
- [`app.py`](file:///c:/Users/Sivan/OneDrive/college/churn%20prediction/app.py): The main Streamlit interactive web application.
- [`churn_model.joblib`](file:///c:/Users/Sivan/OneDrive/college/churn%20prediction/churn_model.joblib): Pre-trained XGBoost pipeline bundle (pipeline, threshold, feature columns).
- [`customer_churn_prediction.ipynb`](file:///c:/Users/Sivan/OneDrive/college/churn%20prediction/customer_churn_prediction.ipynb): End-to-end model exploration and training notebook.
- [`WA_Fn-UseC_-Telco-Customer-Churn.csv`](file:///c:/Users/Sivan/OneDrive/college/churn%20prediction/WA_Fn-UseC_-Telco-Customer-Churn.csv): IBM Telco customer dataset.
- [`requirements.txt`](file:///c:/Users/Sivan/OneDrive/college/churn%20prediction/requirements.txt): Python dependencies.
- [`run_app.bat`](file:///c:/Users/Sivan/OneDrive/college/churn%20prediction/run_app.bat): Windows double-click launcher.
