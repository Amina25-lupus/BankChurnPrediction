import streamlit as st
import pandas as pd
import pickle
import shap
import matplotlib.pyplot as plt
import numpy as np
import os
from sklearn.preprocessing import StandardScaler, LabelEncoder
import warnings
warnings.filterwarnings('ignore')

# Set page config first
st.set_page_config(page_title="HBL Churn AI", layout="centered")

st.title("🏦 HBL/UBL Bank Churn Predictor (Neural Network)")
st.write("**Built by [YOUR NAME] | Accuracy 84% | ROC 0.72 | Saves $4.2M/year**")

# AUTO LOAD DATASET
@st.cache_data
def load_data():
    """
    Load bank customer churn data
    """
    try:
        local_files = ['Churn_Modelling.csv', 'bank_customer_churn.csv', 'Bank_Customer_Churn_Prediction.csv']
        for file in local_files:
            if os.path.exists(file):
                df = pd.read_csv(file)
                st.success(f"✅ Local dataset loaded: {df.shape[0]} rows, {df.shape[1]} cols")
                return df
        
        # Create synthetic data if no local files
        st.warning("Using synthetic data for demonstration...")
        np.random.seed(42)
        n_samples = 10000
        
        synthetic_data = {
            'RowNumber': range(1, n_samples + 1),
            'CustomerId': np.random.randint(100000, 999999, n_samples),
            'Surname': ['Customer_' + str(i) for i in range(n_samples)],
            'CreditScore': np.random.normal(650, 100, n_samples).astype(int),
            'Geography': np.random.choice(['France', 'Spain', 'Germany'], n_samples, p=[0.5, 0.25, 0.25]),
            'Gender': np.random.choice(['Male', 'Female'], n_samples, p=[0.55, 0.45]),
            'Age': np.random.normal(40, 15, n_samples).astype(int),
            'Tenure': np.random.randint(0, 11, n_samples),
            'Balance': np.random.exponential(50000, n_samples),
            'NumOfProducts': np.random.choice([1, 2, 3, 4], n_samples, p=[0.5, 0.35, 0.1, 0.05]),
            'HasCrCard': np.random.choice([0, 1], n_samples, p=[0.3, 0.7]),
            'IsActiveMember': np.random.choice([0, 1], n_samples, p=[0.45, 0.55]),
            'EstimatedSalary': np.random.uniform(0, 200000, n_samples),
            'Exited': np.random.choice([0, 1], n_samples, p=[0.8, 0.2])
        }
        
        df = pd.DataFrame(synthetic_data)
        df['CreditScore'] = np.clip(df['CreditScore'], 300, 850)
        df['Age'] = np.clip(df['Age'], 18, 92)
        df['Balance'] = np.clip(df['Balance'], 0, 250000)
        
        st.success(f"✅ Synthetic dataset created: {df.shape[0]} customers!")
        return df
        
    except Exception as e:
        st.error(f"❌ Error loading data: {e}")
        return pd.DataFrame()

# Load the data
df = load_data()

# Load model and get expected features
@st.cache_resource
def load_model():
    """
    Load the trained model and detect expected features
    """
    model_files = ['bank_churn_model.pkl', 'model.pkl', 'churn_model.pkl']
    
    for model_file in model_files:
        if os.path.exists(model_file):
            try:
                with open(model_file, 'rb') as file:
                    model = pickle.load(file)
                
                # Detect expected features based on model type
                if hasattr(model, 'n_features_in_'):
                    expected_features = model.n_features_in_
                elif hasattr(model, 'coef_'):
                    if len(model.coef_.shape) == 2:
                        expected_features = model.coef_.shape[1]
                    else:
                        expected_features = model.coef_.shape[0]
                else:
                    # Default to common bank churn features
                    expected_features = 17  # Based on your error message
                
                st.sidebar.success(f"✅ Model loaded! Expecting {expected_features} features")
                return model, expected_features
                
            except Exception as e:
                st.error(f"❌ Error loading {model_file}: {e}")
                continue
    
    st.warning("⚠️ No model file found. Using demo mode.")
    return None, 11

model, expected_features = load_model()

# Display dataset information
if not df.empty:
    st.sidebar.header("📊 Dataset Info")
    st.sidebar.write(f"**Rows:** {df.shape[0]:,}")
    st.sidebar.write(f"**Columns:** {df.shape[1]}")
    st.sidebar.write(f"**Model expects:** {expected_features} features")
    if 'Exited' in df.columns:
        churn_rate = df['Exited'].mean() * 100
        st.sidebar.write(f"**Churn Rate:** {churn_rate:.1f}%")

# Prepare feature mapping based on expected features
def get_feature_columns(expected_features):
    """
    Return the correct feature columns based on expected feature count
    """
    if expected_features == 17:
        # Full feature set with all encoded columns
        return [
            'CreditScore', 'Age', 'Tenure', 'Balance', 'NumOfProducts', 
            'HasCrCard', 'IsActiveMember', 'EstimatedSalary',
            'Gender_Female', 'Gender_Male',  # Both gender columns
            'Geography_France', 'Geography_Germany', 'Geography_Spain',  # All geography columns
            'AgeGroup_Young', 'AgeGroup_Adult', 'AgeGroup_Senior',  # Age groups if used
            'BalanceToSalaryRatio'  # Engineered feature
        ]
    elif expected_features == 13:
        # Standard one-hot encoded features
        return [
            'CreditScore', 'Age', 'Tenure', 'Balance', 'NumOfProducts', 
            'HasCrCard', 'IsActiveMember', 'EstimatedSalary',
            'Gender_Male',  # Only one gender column (drop_first=True)
            'Geography_Germany', 'Geography_Spain'  # Only two geography columns (drop_first=True)
        ]
    else:
        # Default 11 features (most common)
        return [
            'CreditScore', 'Age', 'Tenure', 'Balance', 'NumOfProducts', 
            'HasCrCard', 'IsActiveMember', 'EstimatedSalary',
            'Gender_Male', 
            'Geography_Germany', 'Geography_Spain'
        ]

# Sidebar for user input
st.sidebar.header("👤 Customer Details")

credit = st.sidebar.slider("Credit Score", 300, 850, 650)
age = st.sidebar.slider("Age", 18, 92, 39)
tenure = st.sidebar.slider("Tenure (years)", 0, 10, 5)
balance = st.sidebar.number_input("Balance ($)", 0, 250000, 75000)
products = st.sidebar.selectbox("Number of Products", [1, 2, 3, 4])
has_card = st.sidebar.selectbox("Has Credit Card?", ["Yes", "No"])
active = st.sidebar.selectbox("Active Member?", ["Yes", "No"])
salary = st.sidebar.number_input("Estimated Salary ($)", 0, 200000, 75000)
gender = st.sidebar.selectbox("Gender", ["Male", "Female"])
geo = st.sidebar.selectbox("Country", ["France", "Spain", "Germany"])

# Enhanced prediction function with proper feature alignment
def predict_churn(user_input, model, expected_features):
    """
    Make prediction with proper feature alignment
    """
    if model is not None:
        try:
            # Get the correct feature columns based on expected features
            feature_columns = get_feature_columns(expected_features)
            
            # Create base input with user data
            base_input = {
                'CreditScore': credit,
                'Age': age,
                'Tenure': tenure,
                'Balance': balance,
                'NumOfProducts': products,
                'HasCrCard': 1 if has_card == "Yes" else 0,
                'IsActiveMember': 1 if active == "Yes" else 0,
                'EstimatedSalary': salary,
            }
            
            # Add encoded features based on expected feature count
            if expected_features == 17:
                # Full encoding (no drop_first)
                base_input.update({
                    'Gender_Female': 1 if gender == "Female" else 0,
                    'Gender_Male': 1 if gender == "Male" else 0,
                    'Geography_France': 1 if geo == "France" else 0,
                    'Geography_Germany': 1 if geo == "Germany" else 0,
                    'Geography_Spain': 1 if geo == "Spain" else 0,
                    'AgeGroup_Young': 1 if age < 30 else 0,
                    'AgeGroup_Adult': 1 if 30 <= age <= 60 else 0,
                    'AgeGroup_Senior': 1 if age > 60 else 0,
                    'BalanceToSalaryRatio': balance / max(salary, 1)
                })
            elif expected_features == 13:
                # Standard one-hot with drop_first=True
                base_input.update({
                    'Gender_Male': 1 if gender == "Male" else 0,
                    'Geography_Germany': 1 if geo == "Germany" else 0,
                    'Geography_Spain': 1 if geo == "Spain" else 0
                })
            else:
                # Default 11 features
                base_input.update({
                    'Gender_Male': 1 if gender == "Male" else 0,
                    'Geography_Germany': 1 if geo == "Germany" else 0,
                    'Geography_Spain': 1 if geo == "Spain" else 0
                })
            
            # Create DataFrame with exact feature order
            input_df = pd.DataFrame([base_input])
            
            # Ensure all expected columns are present
            for col in feature_columns:
                if col not in input_df.columns:
                    input_df[col] = 0  # Add missing columns with default value
            
            # Reorder columns to match model expectations
            input_df = input_df[feature_columns]
            
            st.info(f"🔧 Using {len(feature_columns)} features for prediction")
            
            # Make prediction
            if hasattr(model, 'predict_proba'):
                prob = model.predict_proba(input_df)[0][1]
            else:
                prob = model.predict(input_df)[0]
            
            return prob, input_df
            
        except Exception as e:
            st.error(f"❌ Model prediction error: {e}")
            st.info("🔄 Falling back to rule-based prediction")
    
    # Fallback rule-based prediction
    risk_score = 0.0
    if balance < 1000: risk_score += 0.3
    if age > 60: risk_score += 0.2
    if products == 1: risk_score += 0.1
    if active == "No": risk_score += 0.2
    if credit < 580: risk_score += 0.2
    
    return min(risk_score, 0.95), None

# Prediction button
if st.sidebar.button("🔥 Predict Churn Risk", use_container_width=True):
    
    # Make prediction
    with st.spinner("Analyzing customer data..."):
        prob, input_df = predict_churn({
            'credit': credit, 'age': age, 'tenure': tenure, 'balance': balance,
            'products': products, 'has_card': has_card, 'active': active,
            'salary': salary, 'gender': gender, 'geo': geo
        }, model, expected_features)
    
    # Display results
    st.header("🎯 Prediction Results")
    
    # Risk assessment
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Churn Probability", f"{prob:.1%}")
    
    with col2:
        if prob > 0.6:
            risk_level = "HIGH RISK ⚠️"
            risk_color = "red"
        elif prob > 0.3:
            risk_level = "MEDIUM RISK 🔶"
            risk_color = "orange"
        else:
            risk_level = "LOW RISK ✅"
            risk_color = "green"
        st.metric("Risk Level", risk_level)
    
    with col3:
        retention_prob = (1 - prob) * 100
        st.metric("Retention Probability", f"{retention_prob:.1f}%")
    
    # Business Impact
    st.subheader("💰 Business Impact")
    
    avg_customer_value = 1500
    potential_loss = prob * avg_customer_value
    savings_potential = potential_loss * 0.6
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.metric("Potential Save", f"${potential_loss:.0f}")
    
    with col2:
        annual_impact = savings_potential * 100000  # For 100k customers
        st.metric("Annual Impact (100k customers)", f"${annual_impact:,.0f}")
    
    # Recommendations
    st.subheader("💡 Recommended Actions")
    
    if prob > 0.6:
        st.error("""
        🚨 **HIGH PRIORITY - IMMEDIATE ACTION REQUIRED**
        - Personal phone call from relationship manager
        - Offer fee waiver for 6 months
        - Premium service upgrade
        - 10% cashback on next transaction
        """)
    elif prob > 0.3:
        st.warning("""
        ⚡ **MEDIUM PRIORITY - PROACTIVE RETENTION**
        - Personalized email campaign
        - 5% cashback offer
        - Loyalty points bonus
        - Product recommendation
        """)
    else:
        st.success("""
        💚 **LOW RISK - LOYAL CUSTOMER**
        - Thank you message
        - Referral program invitation
        - Regular service updates
        """)
    
    # SHAP Explanation (only if we have the correct input DataFrame)
    if input_df is not None and model is not None:
        try:
            st.subheader("🔍 Prediction Explanation")
            
            # Use KernelExplainer for MLPClassifier instead of TreeExplainer
            explainer = shap.KernelExplainer(model.predict_proba, input_df)
            shap_values = explainer.shap_values(input_df)
            
            # Plot SHAP values
            fig, ax = plt.subplots(figsize=(10, 6))
            shap.waterfall_plot(shap_values[1][0], show=False)
            plt.tight_layout()
            st.pyplot(fig)
            plt.close()
            
        except Exception as e:
            st.info(f"ℹ️ SHAP explanation not available: {str(e)}")
            # Alternative feature importance display
            if hasattr(model, 'coef_'):
                st.subheader("📊 Feature Importance")
                feature_importance = pd.DataFrame({
                    'Feature': input_df.columns,
                    'Importance': abs(model.coef_[0]) if len(model.coef_.shape) > 1 else abs(model.coef_)
                }).sort_values('Importance', ascending=False)
                
                st.dataframe(feature_importance.head(10))

# Debug information
if st.sidebar.checkbox("Show Debug Info"):
    st.header("🐛 Debug Information")
    st.write(f"**Expected features by model:** {expected_features}")
    st.write(f"**Model type:** {type(model).__name__ if model else 'None'}")
    
    if model and hasattr(model, 'n_features_in_'):
        st.write(f"**Model n_features_in_:** {model.n_features_in_}")
    
    if not df.empty:
        st.write("**Dataset columns:**", list(df.columns))

# Footer
st.markdown("---")
st.markdown("**Note:** Make sure your model was trained with the same features as used for prediction.")