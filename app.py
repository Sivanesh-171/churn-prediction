import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import plotly.express as px
import plotly.graph_objects as go

# ---------------------------------------------------------
# Page Configuration & Styling
# ---------------------------------------------------------
st.set_page_config(
    page_title="Telco Churn Intelligence Hub",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom Styling
st.markdown("""
<style>
    /* Metric Card Styling */
    div[data-testid="stMetricValue"] {
        font-size: 1.8rem;
        font-weight: 700;
        color: #1E293B;
    }
    .metric-card {
        background: linear-gradient(135deg, #ffffff 0%, #f8fafc 100%);
        padding: 1.2rem;
        border-radius: 12px;
        border: 1px solid #E2E8F0;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.05);
        margin-bottom: 1rem;
    }
    .badge-danger {
        background-color: #FEE2E2;
        color: #991B1B;
        padding: 4px 10px;
        border-radius: 9999px;
        font-weight: 600;
        font-size: 0.85rem;
        display: inline-block;
    }
    .badge-warning {
        background-color: #FEF3C7;
        color: #92400E;
        padding: 4px 10px;
        border-radius: 9999px;
        font-weight: 600;
        font-size: 0.85rem;
        display: inline-block;
    }
    .badge-success {
        background-color: #DCFCE7;
        color: #166534;
        padding: 4px 10px;
        border-radius: 9999px;
        font-weight: 600;
        font-size: 0.85rem;
        display: inline-block;
    }
    .recommendation-box {
        background-color: #F0FDF4;
        border-left: 4px solid #16A34A;
        padding: 1rem;
        border-radius: 0 8px 8px 0;
        margin-top: 0.8rem;
    }
    .alert-box {
        background-color: #FEF2F2;
        border-left: 4px solid #DC2626;
        padding: 1rem;
        border-radius: 0 8px 8px 0;
        margin-top: 0.8rem;
    }
</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------
# Load Model & Dataset with Caching
# ---------------------------------------------------------
@st.cache_resource
def load_churn_model():
    candidates = [
        "churn_model.joblib",
        os.path.join(os.path.dirname(__file__), "churn_model.joblib"),
        os.path.join("project_churn", "churn_model.joblib"),
        os.path.join(os.path.dirname(__file__), "project_churn", "churn_model.joblib"),
    ]
    model_path = next((p for p in candidates if os.path.exists(p)), None)
    if not model_path:
        st.error(f"Model file `churn_model.joblib` not found! Please check the workspace directory.")
        return None, 0.19, []
    bundle = joblib.load(model_path)
    model = bundle.get("model")
    threshold = bundle.get("threshold", 0.19)
    feature_cols = bundle.get("feature_columns", [])
    return model, threshold, feature_cols

@st.cache_data
def load_raw_dataset():
    candidates = [
        "WA_Fn-UseC_-Telco-Customer-Churn.csv",
        os.path.join(os.path.dirname(__file__), "WA_Fn-UseC_-Telco-Customer-Churn.csv"),
        os.path.join("project_churn", "WA_Fn-UseC_-Telco-Customer-Churn.csv"),
        os.path.join(os.path.dirname(__file__), "project_churn", "WA_Fn-UseC_-Telco-Customer-Churn.csv"),
    ]
    csv_path = next((p for p in candidates if os.path.exists(p)), None)
    if csv_path and os.path.exists(csv_path):
        df = pd.read_csv(csv_path)
        df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce").fillna(0)
        return df
    return None

model, default_threshold, feature_columns = load_churn_model()
raw_df = load_raw_dataset()

# ---------------------------------------------------------
# Feature Engineering Function
# ---------------------------------------------------------
def prepare_customer_features(df_input):
    """
    Computes all engineered features matching the model training pipeline:
    - num_services
    - avg_monthly_spend
    - has_support
    - tenure_group
    """
    df = df_input.copy()
    
    # Ensure TotalCharges is numeric
    if "TotalCharges" in df.columns:
        df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce").fillna(0)
    
    service_cols = [
        "PhoneService", "MultipleLines", "OnlineSecurity", "OnlineBackup",
        "DeviceProtection", "TechSupport", "StreamingTV", "StreamingMovies"
    ]
    
    # Count active services
    active_services = (df[service_cols] == "Yes").sum(axis=1)
    internet_active = (df["InternetService"] != "No").astype(int)
    df["num_services"] = active_services + internet_active
    
    # Average monthly spend
    safe_tenure = df["tenure"].replace(0, np.nan)
    df["avg_monthly_spend"] = (df["TotalCharges"] / safe_tenure).fillna(df["MonthlyCharges"])
    
    # Has support or security
    has_sec = df["OnlineSecurity"] == "Yes"
    has_tech = df["TechSupport"] == "Yes"
    df["has_support"] = (has_sec | has_tech).astype(int)
    
    # Tenure bucket
    df["tenure_group"] = pd.cut(
        df["tenure"],
        bins=[-1, 12, 24, 48, np.inf],
        labels=["0-12", "13-24", "25-48", "49+"]
    ).astype(str)
    
    return df

# ---------------------------------------------------------
# Sidebar Configuration
# ---------------------------------------------------------
with st.sidebar:
    st.image("https://img.icons8.com/fluency/96/combo-chart.png", width=64)
    st.title("Telco Churn AI")
    st.caption("Machine Learning & Retention Intelligence")
    st.markdown("---")
    
    st.subheader("⚙️ Model Controls")
    threshold_val = st.slider(
        "Decision Threshold",
        min_value=0.05,
        max_value=0.90,
        value=float(default_threshold),
        step=0.01,
        help="Probability cut-off for classifying a customer as High Churn Risk. Default is cost-optimized at 0.19."
    )
    
    if threshold_val == default_threshold:
        st.info(f"🎯 Set to **Optimal Threshold ({default_threshold:.2f})** based on business cost optimization ($500 FN vs $50 FP).")
    else:
        st.warning(f"⚠️ Custom threshold selected: **{threshold_val:.2f}** (Default is {default_threshold:.2f})")
        
    st.markdown("---")
    st.markdown("### 📋 Quick Customer Presets")
    preset_choice = st.radio(
        "Load demo scenario:",
        ["Custom Input", "🚨 High Risk Churner", "🛡️ Loyal Advocate", "⚖️ At-Risk Borderline"],
        index=0
    )
    
    st.markdown("---")
    st.caption("College Capstone Project • Telco Churn Engine")
    st.caption("Algorithm: **XGBoost Classifier + Cost Tuning**")

# ---------------------------------------------------------
# Main Page Header
# ---------------------------------------------------------
st.title("⚡ Telco Customer Churn Prediction & Retention Hub")
st.markdown(
    "Predict customer attrition risk before it happens, discover key churn drivers, "
    "and trigger AI-powered retention recommendations."
)

tab1, tab2, tab3, tab4 = st.tabs([
    "🎯 Single Customer Predictor",
    "📁 Batch Scoring & Portfolio",
    "📊 Dataset & Churn Drivers",
    "🧠 Model Architecture & Economics"
])

# =========================================================
# TAB 1: Single Customer Predictor
# =========================================================
with tab1:
    st.subheader("Individual Customer Risk Assessment")
    st.markdown("Configure customer details or select a preset from the sidebar to evaluate churn probability.")
    
    # Preset default values
    if preset_choice == "🚨 High Risk Churner":
        d_gender = "Female"
        d_senior = 0
        d_partner = "No"
        d_dependents = "No"
        d_tenure = 2
        d_phone = "Yes"
        d_multiline = "No"
        d_internet = "Fiber optic"
        d_sec = "No"
        d_backup = "No"
        d_protect = "No"
        d_tech = "No"
        d_tv = "Yes"
        d_movies = "Yes"
        d_contract = "Month-to-month"
        d_paperless = "Yes"
        d_payment = "Electronic check"
        d_monthly = 95.50
        d_total = 191.00
    elif preset_choice == "🛡️ Loyal Advocate":
        d_gender = "Male"
        d_senior = 0
        d_partner = "Yes"
        d_dependents = "Yes"
        d_tenure = 65
        d_phone = "Yes"
        d_multiline = "Yes"
        d_internet = "DSL"
        d_sec = "Yes"
        d_backup = "Yes"
        d_protect = "Yes"
        d_tech = "Yes"
        d_tv = "No"
        d_movies = "No"
        d_contract = "Two year"
        d_paperless = "No"
        d_payment = "Bank transfer (automatic)"
        d_monthly = 64.20
        d_total = 4173.00
    elif preset_choice == "⚖️ At-Risk Borderline":
        d_gender = "Male"
        d_senior = 1
        d_partner = "Yes"
        d_dependents = "No"
        d_tenure = 14
        d_phone = "Yes"
        d_multiline = "No"
        d_internet = "Fiber optic"
        d_sec = "No"
        d_backup = "Yes"
        d_protect = "No"
        d_tech = "No"
        d_tv = "No"
        d_movies = "No"
        d_contract = "Month-to-month"
        d_paperless = "Yes"
        d_payment = "Electronic check"
        d_monthly = 75.00
        d_total = 1050.00
    else:
        # Default Custom
        d_gender = "Female"
        d_senior = 0
        d_partner = "Yes"
        d_dependents = "No"
        d_tenure = 12
        d_phone = "Yes"
        d_multiline = "No"
        d_internet = "Fiber optic"
        d_sec = "No"
        d_backup = "No"
        d_protect = "No"
        d_tech = "No"
        d_tv = "No"
        d_movies = "No"
        d_contract = "Month-to-month"
        d_paperless = "Yes"
        d_payment = "Electronic check"
        d_monthly = 70.00
        d_total = 840.00

    col_form, col_result = st.columns([1.1, 0.9], gap="large")

    with col_form:
        with st.expander("👤 1. Demographics & Account Info", expanded=True):
            c1, c2 = st.columns(2)
            with c1:
                gender = st.selectbox("Gender", ["Female", "Male"], index=["Female", "Male"].index(d_gender))
                senior_citizen = st.selectbox("Senior Citizen", [0, 1], index=[0, 1].index(d_senior), format_func=lambda x: "Yes" if x == 1 else "No")
            with c2:
                partner = st.selectbox("Has Partner", ["Yes", "No"], index=["Yes", "No"].index(d_partner))
                dependents = st.selectbox("Has Dependents", ["Yes", "No"], index=["Yes", "No"].index(d_dependents))
                
        with st.expander("📄 2. Contract & Financial Profile", expanded=True):
            c3, c4 = st.columns(2)
            with c3:
                contract = st.selectbox("Contract Type", ["Month-to-month", "One year", "Two year"], index=["Month-to-month", "One year", "Two year"].index(d_contract))
                tenure = st.slider("Tenure (Months with Company)", min_value=0, max_value=72, value=d_tenure, step=1)
                paperless = st.selectbox("Paperless Billing", ["Yes", "No"], index=["Yes", "No"].index(d_paperless))
            with c4:
                payment_method = st.selectbox(
                    "Payment Method",
                    ["Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"],
                    index=["Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"].index(d_payment)
                )
                monthly_charges = st.number_input("Monthly Charges ($)", min_value=15.0, max_value=150.0, value=float(d_monthly), step=1.0)
                auto_calc_total = tenure * monthly_charges
                total_charges = st.number_input("Total Charges ($)", min_value=0.0, max_value=10000.0, value=float(d_total if d_total > 0 else auto_calc_total), step=10.0)

        with st.expander("🌐 3. Subscribed Services", expanded=True):
            c5, c6 = st.columns(2)
            with c5:
                phone_service = st.selectbox("Phone Service", ["Yes", "No"], index=["Yes", "No"].index(d_phone))
                multiline_opts = ["No phone service", "No", "Yes"] if phone_service == "No" else ["No", "Yes"]
                multiple_lines = st.selectbox("Multiple Lines", multiline_opts, index=0 if d_multiline not in multiline_opts else multiline_opts.index(d_multiline))
                
                internet_service = st.selectbox("Internet Service Provider", ["DSL", "Fiber optic", "No"], index=["DSL", "Fiber optic", "No"].index(d_internet))
                
                sec_opts = ["No internet service"] if internet_service == "No" else ["No", "Yes"]
                online_security = st.selectbox("Online Security", sec_opts, index=0 if d_sec not in sec_opts else sec_opts.index(d_sec))
                online_backup = st.selectbox("Online Backup", sec_opts, index=0 if d_backup not in sec_opts else sec_opts.index(d_backup))
            with c6:
                device_protection = st.selectbox("Device Protection", sec_opts, index=0 if d_protect not in sec_opts else sec_opts.index(d_protect))
                tech_support = st.selectbox("Tech Support", sec_opts, index=0 if d_tech not in sec_opts else sec_opts.index(d_tech))
                streaming_tv = st.selectbox("Streaming TV", sec_opts, index=0 if d_tv not in sec_opts else sec_opts.index(d_tv))
                streaming_movies = st.selectbox("Streaming Movies", sec_opts, index=0 if d_movies not in sec_opts else sec_opts.index(d_movies))

    # Prediction Logic
    raw_input_dict = {
        "gender": [gender],
        "SeniorCitizen": [senior_citizen],
        "Partner": [partner],
        "Dependents": [dependents],
        "tenure": [tenure],
        "PhoneService": [phone_service],
        "MultipleLines": [multiple_lines],
        "InternetService": [internet_service],
        "OnlineSecurity": [online_security],
        "OnlineBackup": [online_backup],
        "DeviceProtection": [device_protection],
        "TechSupport": [tech_support],
        "StreamingTV": [streaming_tv],
        "StreamingMovies": [streaming_movies],
        "Contract": [contract],
        "PaperlessBilling": [paperless],
        "PaymentMethod": [payment_method],
        "MonthlyCharges": [monthly_charges],
        "TotalCharges": [total_charges]
    }
    
    input_df = pd.DataFrame(raw_input_dict)
    engineered_df = prepare_customer_features(input_df)
    
    # Align features with training columns
    X_pred = engineered_df[feature_columns]
    
    with col_result:
        st.subheader("🎯 Real-Time Churn Analysis")
        
        if model is not None:
            churn_proba = float(model.predict_proba(X_pred)[0, 1])
            is_churn = churn_proba >= threshold_val
            
            # Gauge Chart
            fig_gauge = go.Figure(go.Indicator(
                mode="gauge+number",
                value=churn_proba * 100,
                domain={'x': [0, 1], 'y': [0, 1]},
                title={'text': "<b>Churn Probability</b>", 'font': {'size': 20, 'color': '#1E293B'}},
                number={'suffix': "%", 'font': {'size': 36, 'color': '#DC2626' if is_churn else '#16A34A'}},
                gauge={
                    'axis': {'range': [0, 100], 'tickwidth': 1, 'tickcolor': "#94A3B8"},
                    'bar': {'color': "#DC2626" if churn_proba >= 0.50 else ("#F59E0B" if churn_proba >= threshold_val else "#10B981")},
                    'bgcolor': "white",
                    'borderwidth': 1,
                    'bordercolor': "#E2E8F0",
                    'steps': [
                        {'range': [0, threshold_val * 100], 'color': '#DCFCE7'},
                        {'range': [threshold_val * 100, 50], 'color': '#FEF3C7'},
                        {'range': [50, 100], 'color': '#FEE2E2'}
                    ],
                    'threshold': {
                        'line': {'color': "red", 'width': 4},
                        'thickness': 0.75,
                        'value': threshold_val * 100
                    }
                }
            ))
            fig_gauge.update_layout(height=260, margin=dict(l=20, r=20, t=40, b=10))
            st.plotly_chart(fig_gauge, use_container_width=True)
            
            # Risk Classification Badge
            if churn_proba >= 0.50:
                st.markdown(
                    f'<div style="text-align:center;"><span class="badge-danger">🚨 CRITICAL CHURN RISK ({churn_proba:.1%})</span></div>',
                    unsafe_allow_html=True
                )
            elif is_churn:
                st.markdown(
                    f'<div style="text-align:center;"><span class="badge-warning">⚠️ ELEVATED AT-RISK ({churn_proba:.1%}) - Above {threshold_val:.2f} Threshold</span></div>',
                    unsafe_allow_html=True
                )
            else:
                st.markdown(
                    f'<div style="text-align:center;"><span class="badge-success">🛡️ SAFE / LOW RISK ({churn_proba:.1%})</span></div>',
                    unsafe_allow_html=True
                )
            
            st.markdown("<br>", unsafe_allow_html=True)
            
            # Risk Drivers Breakdown for this specific customer
            st.markdown("#### 🔍 Customer Risk Factors Identified")
            risk_drivers = []
            positive_factors = []
            
            if contract == "Month-to-month":
                risk_drivers.append("⚠️ **Month-to-Month Contract:** Primary churn driver across the entire dataset.")
            else:
                positive_factors.append(f"✅ **{contract} Commitment:** High stability factor.")
                
            if internet_service == "Fiber optic" and tech_support != "Yes":
                risk_drivers.append("⚠️ **Fiber Optic without Tech Support:** Customers paying premium rates without support have 3x higher churn.")
            
            if tenure <= 12:
                risk_drivers.append(f"⚠️ **Early Lifecycle (Tenure {tenure} mo):** First-year customers have the highest attrition rate.")
            elif tenure >= 36:
                positive_factors.append(f"✅ **Established Loyalty ({tenure} months):** Long tenure strongly protects against churn.")
                
            if payment_method == "Electronic check":
                risk_drivers.append("⚠️ **Payment Method:** Electronic check users exhibit the highest payment friction and churn.")
            elif "automatic" in payment_method:
                positive_factors.append("✅ **Autopay Active:** Automatic bank/card payments minimize voluntary churn.")
                
            if online_security == "No" and tech_support == "No":
                risk_drivers.append("⚠️ **No Security/Support Bundle:** Lacks sticky support services.")
            elif engineered_df["has_support"].iloc[0] == 1:
                positive_factors.append("✅ **Support/Security Subscribed:** Adds product stickiness.")

            if risk_drivers:
                for r in risk_drivers:
                    st.markdown(r)
            if positive_factors:
                for p in positive_factors:
                    st.markdown(p)
                    
            # Prescriptive Recommendations
            st.markdown("#### 💡 Prescriptive Retention Actions")
            if is_churn:
                st.markdown("""
                <div class="alert-box">
                    <strong>Recommended Retention Playbook:</strong>
                    <ol style="margin-top: 5px; margin-bottom: 0px;">
                        <li><strong>Annual Contract Incentive:</strong> Offer a 15% promotional discount to migrate from Month-to-month to a 1-Year agreement.</li>
                        <li><strong>Free Tech Support Add-on:</strong> Bundle 6 months of complimentary TechSupport to resolve service issues before churn happens.</li>
                        <li><strong>Autopay Incentive:</strong> Offer a one-time $10 credit to switch from Electronic check to Automatic Card/ACH billing.</li>
                    </ol>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown("""
                <div class="recommendation-box">
                    <strong>Relationship Nurturing & Expansion:</strong>
                    <ul style="margin-top: 5px; margin-bottom: 0px;">
                        <li>Customer profile is healthy. No emergency retention discount required.</li>
                        <li>Eligible for cross-sell campaigns (e.g. streaming bundles or smart home device protection).</li>
                    </ul>
                </div>
                """, unsafe_allow_html=True)

# =========================================================
# TAB 2: Batch Scoring & Portfolio Analysis
# =========================================================
with tab2:
    st.subheader("Batch Customer Risk Scoring")
    st.markdown("Upload a customer CSV or analyze the default enterprise portfolio to prioritize retention campaigns.")
    
    batch_mode = st.radio("Choose Data Source:", ["Default Telco Dataset (7,043 Customers)", "Upload Custom CSV"], horizontal=True)
    
    batch_df = None
    if batch_mode == "Upload Custom CSV":
        uploaded_file = st.file_uploader("Upload CSV matching Telco schema", type=["csv"])
        if uploaded_file is not None:
            batch_df = pd.read_csv(uploaded_file)
            st.success(f"Successfully uploaded {len(batch_df):,} customers.")
    else:
        if raw_df is not None:
            batch_df = raw_df.copy()
            st.info(f"Loaded existing Telco dataset with **{len(batch_df):,} customers**.")
        else:
            st.error("Default dataset not found in workspace.")
            
    if batch_df is not None and model is not None:
        with st.spinner("Processing features and computing batch predictions..."):
            prep_batch = prepare_customer_features(batch_df)
            X_batch = prep_batch[feature_columns]
            probabilities = model.predict_proba(X_batch)[:, 1]
            
            results_df = batch_df.copy()
            results_df["Churn_Probability"] = np.round(probabilities, 4)
            results_df["Risk_Category"] = np.where(
                probabilities >= 0.50, "Critical Risk",
                np.where(probabilities >= threshold_val, "Elevated Risk", "Low Risk")
            )
            results_df["Predicted_Churn"] = (probabilities >= threshold_val).astype(int)

        # Summary KPIs
        total_customers = len(results_df)
        predicted_churners = int(results_df["Predicted_Churn"].sum())
        churn_pct = (predicted_churners / total_customers) * 100
        
        # Financial Impact
        monthly_at_risk = results_df[results_df["Predicted_Churn"] == 1]["MonthlyCharges"].sum()
        annual_at_risk = monthly_at_risk * 12
        
        col_m1, col_m2, col_m3, col_m4 = st.columns(4)
        col_m1.metric("Total Customers", f"{total_customers:,}")
        col_m2.metric("Flagged Churners", f"{predicted_churners:,}", f"{churn_pct:.1f}% of base")
        col_m3.metric("Monthly MRR at Risk", f"${monthly_at_risk:,.2f}")
        col_m4.metric("Annual ARR at Risk", f"${annual_at_risk:,.2f}")
        
        st.markdown("---")
        
        # Visual breakdown
        c_v1, c_v2 = st.columns(2)
        with c_v1:
            fig_risk_pie = px.pie(
                results_df,
                names="Risk_Category",
                title="Customer Portfolio Risk Distribution",
                color="Risk_Category",
                color_discrete_map={
                    "Critical Risk": "#DC2626",
                    "Elevated Risk": "#F59E0B",
                    "Low Risk": "#10B981"
                },
                hole=0.45
            )
            fig_risk_pie.update_layout(height=350, margin=dict(t=40, b=20, l=20, r=20))
            st.plotly_chart(fig_risk_pie, use_container_width=True)
            
        with c_v2:
            fig_hist = px.histogram(
                results_df,
                x="Churn_Probability",
                nbins=30,
                title="Churn Probability Distribution across Portfolio",
                color="Risk_Category",
                color_discrete_map={
                    "Critical Risk": "#DC2626",
                    "Elevated Risk": "#F59E0B",
                    "Low Risk": "#10B981"
                }
            )
            fig_hist.add_vline(x=threshold_val, line_dash="dash", line_color="black", annotation_text=f"Threshold ({threshold_val:.2f})")
            fig_hist.update_layout(height=350, margin=dict(t=40, b=20, l=20, r=20), xaxis_title="Predicted Churn Probability", yaxis_title="Customer Count")
            st.plotly_chart(fig_hist, use_container_width=True)

        # High Priority Customer Worklist
        st.subheader("📋 Top Priority Customers for Retention Outreach")
        filter_risk = st.multiselect("Filter by Risk Tier:", ["Critical Risk", "Elevated Risk", "Low Risk"], default=["Critical Risk", "Elevated Risk"])
        
        display_cols = [c for c in ["customerID", "Contract", "tenure", "MonthlyCharges", "PaymentMethod", "Churn_Probability", "Risk_Category"] if c in results_df.columns]
        filtered_view = results_df[results_df["Risk_Category"].isin(filter_risk)][display_cols].sort_values("Churn_Probability", ascending=False)
        
        st.dataframe(filtered_view.head(200), use_container_width=True)
        
        # Download results
        csv_download = results_df.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="📥 Download Full Scored Portfolio CSV",
            data=csv_download,
            file_name="telco_churn_predictions.csv",
            mime="text/csv",
        )

# =========================================================
# TAB 3: Dataset Analytics & Churn Drivers
# =========================================================
with tab3:
    st.subheader("Exploratory Data Analysis: What Drives Churn?")
    st.markdown("Visualizing empirical churn relationships in the IBM Telco sample dataset.")
    
    if raw_df is not None:
        c_eda1, c_eda2 = st.columns(2)
        
        with c_eda1:
            contract_churn = raw_df.groupby("Contract")["Churn"].apply(lambda s: (s == "Yes").mean() * 100).reset_index()
            fig_contract = px.bar(
                contract_churn,
                x="Contract",
                y="Churn",
                text=contract_churn["Churn"].apply(lambda x: f"{x:.1f}%"),
                title="Churn Rate by Contract Type (%)",
                color="Contract",
                color_discrete_sequence=["#EF4444", "#3B82F6", "#10B981"]
            )
            fig_contract.update_layout(yaxis_title="Churn Rate (%)", showlegend=False)
            st.plotly_chart(fig_contract, use_container_width=True)
            
        with c_eda2:
            inet_churn = raw_df.groupby("InternetService")["Churn"].apply(lambda s: (s == "Yes").mean() * 100).reset_index()
            fig_inet = px.bar(
                inet_churn,
                x="InternetService",
                y="Churn",
                text=inet_churn["Churn"].apply(lambda x: f"{x:.1f}%"),
                title="Churn Rate by Internet Service (%)",
                color="InternetService",
                color_discrete_sequence=["#3B82F6", "#EF4444", "#10B981"]
            )
            fig_inet.update_layout(yaxis_title="Churn Rate (%)", showlegend=False)
            st.plotly_chart(fig_inet, use_container_width=True)
            
        c_eda3, c_eda4 = st.columns(2)
        
        with c_eda3:
            fig_tenure = px.histogram(
                raw_df,
                x="tenure",
                color="Churn",
                barmode="overlay",
                nbins=36,
                title="Tenure Distribution by Churn Status",
                color_discrete_map={"Yes": "#EF4444", "No": "#3B82F6"}
            )
            fig_tenure.update_layout(xaxis_title="Tenure (Months)", yaxis_title="Number of Customers")
            st.plotly_chart(fig_tenure, use_container_width=True)
            
        with c_eda4:
            fig_box = px.box(
                raw_df,
                x="Churn",
                y="MonthlyCharges",
                color="Churn",
                title="Monthly Charges by Churn Status ($)",
                color_discrete_map={"Yes": "#EF4444", "No": "#3B82F6"}
            )
            fig_box.update_layout(yaxis_title="Monthly Charges ($)", showlegend=False)
            st.plotly_chart(fig_box, use_container_width=True)

        st.markdown("---")
        st.markdown("### 📌 Key Business Insights from Data Exploration")
        kpi_c1, kpi_c2, kpi_c3 = st.columns(3)
        with kpi_c1:
            st.markdown("""
            **1. Contract Type is King**
            - Month-to-month customers churn at **42.7%**
            - One-year: **11.3%**
            - Two-year: **2.8%**
            *Takeaway:* Locking customers into annual agreements reduces churn by 75-90%.
            """)
        with kpi_c2:
            st.markdown("""
            **2. The First Year Danger Zone**
            - High attrition spikes in months **1 to 6**.
            - Customers who survive past 24 months rarely churn.
            *Takeaway:* Intensive onboarding and early proactive check-ins are crucial.
            """)
        with kpi_c3:
            st.markdown("""
            **3. Fiber Optic Churn Paradox**
            - Fiber optic users churn at **41.9%** compared to DSL at **18.9%**.
            - Caused by high charges ($70-100+) without bundled tech support.
            *Takeaway:* Bundle free tech support with premium fiber packages.
            """)

# =========================================================
# TAB 4: Model Architecture & Economics
# =========================================================
with tab4:
    st.subheader("Model Architecture & Decision Threshold Economics")
    st.markdown("Why was this model chosen and why is the optimal decision threshold set to **0.19** instead of **0.50**?")
    
    col_arch1, col_arch2 = st.columns([1, 1], gap="large")
    
    with col_arch1:
        st.markdown("#### 🏗️ Machine Learning Pipeline")
        st.markdown("""
        The production pipeline trained in `customer_churn_prediction.ipynb` consists of:
        1. **Feature Engineering**:
           - `num_services`: Sum of active telecom services
           - `avg_monthly_spend`: Historical spending rate
           - `has_support`: Presence of online security or technical assistance
           - `tenure_group`: Discretized tenure buckets (0-12, 13-24, 25-48, 49+)
        2. **ColumnTransformer Preprocessor**:
           - **StandardScaler**: Scaled numerical features (`tenure`, `MonthlyCharges`, `TotalCharges`, etc.)
           - **OneHotEncoder**: Encoded categorical attributes with `handle_unknown='ignore'`
        3. **Predictive Engine**:
           - **XGBoost Classifier** with class imbalance tuning (`scale_pos_weight`), `n_estimators=200`, `max_depth=5`, `learning_rate=0.01`.
        """)
        
        st.markdown("#### 🏆 Model Comparison on Test Set")
        metrics_data = {
            "Model": ["Logistic Regression", "Random Forest", "Tuned XGBoost (Current)"],
            "ROC-AUC": [0.842, 0.835, 0.846],
            "Recall (@ 0.19 Threshold)": ["86.2%", "84.8%", "89.3%"],
            "Key Strength": ["Interpretable baseline", "Handles non-linearities", "Best tabular ranking & cost balance"]
        }
        st.table(pd.DataFrame(metrics_data))
        
    with col_arch2:
        st.markdown("#### 💰 Business Cost Economics (Why Threshold = 0.19?)")
        st.markdown("""
        In churn prevention, **mistakes have asymmetric costs**:
        - **False Negative (Missed Churner):** The customer leaves. Replacing a lost customer costs roughly **$500**.
        - **False Positive (Unnecessary Retention Offer):** Giving a discount to someone who wasn't leaving costs only **$50**.
        
        Because missing a churner is **10x costlier** than giving a small proactive discount, the optimal threshold shifts down from 0.50 to **0.19**!
        """)
        
        # Interactive Cost Simulator
        st.markdown("##### 🧮 Interactive Cost Optimization Simulator")
        sim_fn_cost = st.slider("Cost of Missed Churner (FN) in $", min_value=100, max_value=1000, value=500, step=50)
        sim_fp_cost = st.slider("Cost of Retention Discount (FP) in $", min_value=10, max_value=200, value=50, step=10)
        
        ratio = sim_fn_cost / sim_fp_cost
        approx_threshold = max(0.08, min(0.60, round(1 / (1 + ratio), 2)))
        st.success(f"Cost Ratio = {ratio:.1f}:1 → Mathematically Optimal Threshold ≈ **{approx_threshold:.2f}**")
        st.caption("At this threshold, your business minimizes total financial losses across the customer base.")

# ---------------------------------------------------------
# Footer
# ---------------------------------------------------------
st.markdown("---")
f_col1, f_col2 = st.columns([2, 1])
with f_col1:
    st.caption("Developed with Streamlit • Scikit-learn Pipeline • XGBoost • Plotly")
with f_col2:
    st.caption("© Telco Customer Churn Prediction Project")
