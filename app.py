"""
Fraud Detection Dashboard
Streamlit Application for Fraud Detection Analysis
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import warnings
import os
from datetime import datetime

# Suppress warnings
warnings.filterwarnings('ignore')

# Page configuration
st.set_page_config(
    page_title="Fraud Detection Dashboard",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .sub-header {
        font-size: 1.5rem;
        font-weight: bold;
        color: #2c3e50;
        margin-top: 1rem;
        margin-bottom: 1rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 10px;
        text-align: center;
    }
    .fraud-highlight {
        color: #e74c3c;
        font-weight: bold;
    }
    .safe-highlight {
        color: #2ecc71;
        font-weight: bold;
    }
    .stButton button {
        background-color: #1f77b4;
        color: white;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'data_loaded' not in st.session_state:
    st.session_state.data_loaded = False
if 'df' not in st.session_state:
    st.session_state.df = None
if 'df_clean' not in st.session_state:
    st.session_state.df_clean = None

# Title
st.markdown('<div class="main-header">🔍 Fraud Detection Analytics Dashboard</div>', unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    st.image("https://img.icons8.com/color/96/000000/security-checked--v1.png", width=80)
    st.markdown("## Navigation")
    
    page = st.radio(
        "Select Page",
        ["🏠 Home", "📊 Data Overview", "🔍 Fraud Analysis", "⚠️ Risk Scoring", "💡 Recommendations"],
        index=0
    )
    
    st.markdown("---")
    st.markdown("### About")
    st.info(
        "This dashboard analyzes transaction data to detect fraudulent patterns. "
        "It provides insights into fraud risk factors, visualizations, and actionable recommendations."
    )
    
    st.markdown("---")
    st.markdown("### Data Source")
    st.caption("Fraud Detection Dataset with 25,000+ transactions")

# Data loading function
@st.cache_data
def load_and_clean_data():
    """Load and clean the fraud detection dataset"""
    
    # Load data
    df = pd.read_csv('data/Fraud_Detection_Dataset.csv')
    
    # Create a copy for cleaning
    df_clean = df.copy()
    
    # Handle missing values
    categorical_cols = ['Transaction_Type', 'Device_Used', 'Location', 'Payment_Method']
    for col in categorical_cols:
        mode_value = df_clean[col].mode()[0] if not df_clean[col].mode().empty else 'Unknown'
        df_clean[col] = df_clean[col].fillna(mode_value)
    
    # Fill missing numeric values
    numeric_cols = ['Time_of_Transaction', 'Transaction_Amount']
    for col in numeric_cols:
        median_value = df_clean[col].median()
        df_clean[col] = df_clean[col].fillna(median_value)
    
    # Clean categorical variables
    df_clean['Payment_Method'] = df_clean['Payment_Method'].replace('', 'Unknown')
    df_clean['Payment_Method'] = df_clean['Payment_Method'].replace('Invalid Method', 'Other')
    df_clean['Device_Used'] = df_clean['Device_Used'].replace('', 'Unknown')
    df_clean['Device_Used'] = df_clean['Device_Used'].replace('Unknown Device', 'Other')
    df_clean['Location'] = df_clean['Location'].replace('', 'Unknown')
    
    # Handle outliers in Transaction Amount
    Q1 = df_clean['Transaction_Amount'].quantile(0.25)
    Q3 = df_clean['Transaction_Amount'].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    df_clean['Transaction_Amount'] = df_clean['Transaction_Amount'].clip(lower=lower_bound, upper=upper_bound)
    
    # Calculate Risk Score
    def calculate_risk_score(row):
        risk = 0
        high_risk_types = ['Online Purchase', 'Bank Transfer']
        if row['Transaction_Type'] in high_risk_types:
            risk += 2
        high_risk_devices = ['Unknown', 'Other']
        if row['Device_Used'] in high_risk_devices:
            risk += 2
        high_risk_payments = ['Net Banking', 'UPI']
        if row['Payment_Method'] in high_risk_payments:
            risk += 1
        if row['Previous_Fraudulent_Transactions'] > 0:
            risk += min(row['Previous_Fraudulent_Transactions'], 3)
        if row['Transaction_Amount'] > df_clean['Transaction_Amount'].quantile(0.95):
            risk += 2
        if row['Number_of_Transactions_Last_24H'] > 10:
            risk += 1
        if row['Account_Age'] < 30:
            risk += 1
        return risk
    
    df_clean['Risk_Score'] = df_clean.apply(calculate_risk_score, axis=1)
    
    return df, df_clean

# Load data
if not st.session_state.data_loaded:
    with st.spinner("Loading and processing data..."):
        st.session_state.df, st.session_state.df_clean = load_and_clean_data()
        st.session_state.data_loaded = True

df = st.session_state.df
df_clean = st.session_state.df_clean

# Page routing
if page == "🏠 Home":
    # Home Page
    st.markdown("## Welcome to the Fraud Detection Dashboard")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Transactions", f"{len(df_clean):,}")
    with col2:
        fraud_count = df_clean['Fraudulent'].sum()
        st.metric("Fraudulent Transactions", f"{fraud_count:,}", delta=f"{fraud_count/len(df_clean)*100:.1f}%")
    with col3:
        avg_amount = df_clean['Transaction_Amount'].mean()
        st.metric("Average Transaction", f"${avg_amount:.2f}")
    with col4:
        high_risk = len(df_clean[df_clean['Risk_Score'] >= 5])
        st.metric("High Risk Transactions", f"{high_risk:,}", delta=f"{high_risk/len(df_clean)*100:.1f}%")
    
    st.markdown("---")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### 📊 Quick Statistics")
        st.dataframe(df_clean.describe(), use_container_width=True)
    
    with col2:
        st.markdown("### 📋 Data Quality Report")
        
        # Missing values report
        missing_df = pd.DataFrame({
            'Column': df.columns,
            'Missing Values': df.isnull().sum(),
            'Missing %': (df.isnull().sum() / len(df)) * 100
        })
        missing_df = missing_df[missing_df['Missing Values'] > 0]
        
        if len(missing_df) > 0:
            st.write("**Missing Values Handled:**")
            st.dataframe(missing_df, use_container_width=True)
        else:
            st.success("No missing values found in the dataset!")
        
        st.write("**Data Quality Metrics:**")
        st.write(f"- Total records: {len(df_clean):,}")
        st.write(f"- Features: {len(df_clean.columns)}")
        st.write(f"- Duplicate rows: {df_clean.duplicated().sum():,}")
    
    st.markdown("---")
    st.markdown("### 🔍 Quick Preview of Cleaned Data")
    st.dataframe(df_clean.head(100), use_container_width=True)

elif page == "📊 Data Overview":
    # Data Overview Page
    st.markdown('<div class="sub-header">📊 Data Overview & Exploration</div>', unsafe_allow_html=True)
    
    tab1, tab2, tab3 = st.tabs(["📈 Distributions", "📉 Statistics", "🔗 Correlations"])
    
    with tab1:
        st.markdown("### Feature Distributions")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Transaction Amount Distribution
            fig = px.histogram(
                df_clean, x='Transaction_Amount', nbins=50,
                title='Transaction Amount Distribution',
                color_discrete_sequence=['#1f77b4']
            )
            fig.update_layout(height=400)
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # Account Age Distribution
            fig = px.histogram(
                df_clean, x='Account_Age', nbins=30,
                title='Account Age Distribution',
                color_discrete_sequence=['#2ecc71']
            )
            fig.update_layout(height=400)
            st.plotly_chart(fig, use_container_width=True)
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Transaction Types
            type_counts = df_clean['Transaction_Type'].value_counts()
            fig = px.pie(
                values=type_counts.values, names=type_counts.index,
                title='Transaction Types Distribution',
                color_discrete_sequence=px.colors.qualitative.Set3
            )
            fig.update_layout(height=400)
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # Payment Methods
            payment_counts = df_clean['Payment_Method'].value_counts().head(10)
            fig = px.bar(
                x=payment_counts.values, y=payment_counts.index,
                orientation='h', title='Top Payment Methods',
                color_discrete_sequence=['#e74c3c']
            )
            fig.update_layout(height=400)
            st.plotly_chart(fig, use_container_width=True)
    
    with tab2:
        st.markdown("### Statistical Summary")
        
        # Numeric features statistics
        st.subheader("Numeric Features")
        st.dataframe(df_clean.describe(), use_container_width=True)
        
        # Categorical features
        st.subheader("Categorical Features")
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("**Transaction Types**")
            st.dataframe(df_clean['Transaction_Type'].value_counts(), use_container_width=True)
        
        with col2:
            st.write("**Devices Used**")
            st.dataframe(df_clean['Device_Used'].value_counts().head(10), use_container_width=True)
    
    with tab3:
        st.markdown("### Correlation Analysis")
        
        # Correlation heatmap
        numeric_cols = ['Transaction_Amount', 'Time_of_Transaction', 'Previous_Fraudulent_Transactions',
                        'Account_Age', 'Number_of_Transactions_Last_24H', 'Fraudulent']
        
        corr_matrix = df_clean[numeric_cols].corr()
        
        fig = px.imshow(
            corr_matrix,
            text_auto=True,
            aspect="auto",
            color_continuous_scale='RdBu',
            title='Feature Correlation Matrix'
        )
        fig.update_layout(height=600)
        st.plotly_chart(fig, use_container_width=True)
        
        st.markdown("### Correlation Insights")
        
        corr_with_fraud = corr_matrix['Fraudulent'].sort_values(ascending=False)
        for col, corr in corr_with_fraud.items():
            if col != 'Fraudulent':
                if abs(corr) > 0.1:
                    st.info(f"**{col}** has {abs(corr):.3f} {'positive' if corr > 0 else 'negative'} correlation with fraud")

elif page == "🔍 Fraud Analysis":
    # Fraud Analysis Page
    st.markdown('<div class="sub-header">🔍 Fraud Pattern Analysis</div>', unsafe_allow_html=True)
    
    # Filters
    st.sidebar.markdown("### Filters")
    selected_type = st.sidebar.multiselect(
        "Transaction Type",
        options=df_clean['Transaction_Type'].unique(),
        default=df_clean['Transaction_Type'].unique()[:5]
    )
    
    selected_device = st.sidebar.multiselect(
        "Device Used",
        options=df_clean['Device_Used'].unique(),
        default=df_clean['Device_Used'].unique()[:5]
    )
    
    # Filter data
    filtered_df = df_clean.copy()
    if selected_type:
        filtered_df = filtered_df[filtered_df['Transaction_Type'].isin(selected_type)]
    if selected_device:
        filtered_df = filtered_df[filtered_df['Device_Used'].isin(selected_device)]
    
    # Metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Filtered Transactions", f"{len(filtered_df):,}")
    with col2:
        fraud_count = filtered_df['Fraudulent'].sum()
        st.metric("Fraudulent", f"{fraud_count:,}")
    with col3:
        fraud_rate = fraud_count / len(filtered_df) * 100 if len(filtered_df) > 0 else 0
        st.metric("Fraud Rate", f"{fraud_rate:.2f}%")
    with col4:
        avg_amount = filtered_df['Transaction_Amount'].mean()
        st.metric("Avg Amount", f"${avg_amount:.2f}")
    
    st.markdown("---")
    
    tab1, tab2, tab3 = st.tabs(["📊 Fraud Patterns", "📍 Location Analysis", "⏰ Time Analysis"])
    
    with tab1:
        col1, col2 = st.columns(2)
        
        with col1:
            # Fraud by Transaction Type
            fraud_by_type = filtered_df.groupby('Transaction_Type')['Fraudulent'].mean() * 100
            fraud_by_type = fraud_by_type.sort_values(ascending=False)
            
            fig = px.bar(
                x=fraud_by_type.values, y=fraud_by_type.index,
                orientation='h', title='Fraud Rate by Transaction Type',
                color=fraud_by_type.values, color_continuous_scale='RdYlGn_r',
                labels={'x': 'Fraud Rate (%)', 'y': 'Transaction Type'}
            )
            fig.update_layout(height=400)
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # Fraud by Device
            fraud_by_device = filtered_df.groupby('Device_Used')['Fraudulent'].mean() * 100
            fraud_by_device = fraud_by_device.sort_values(ascending=False).head(10)
            
            fig = px.bar(
                x=fraud_by_device.values, y=fraud_by_device.index,
                orientation='h', title='Fraud Rate by Device (Top 10)',
                color=fraud_by_device.values, color_continuous_scale='RdYlGn_r',
                labels={'x': 'Fraud Rate (%)', 'y': 'Device'}
            )
            fig.update_layout(height=400)
            st.plotly_chart(fig, use_container_width=True)
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Fraud by Payment Method
            fraud_by_payment = filtered_df.groupby('Payment_Method')['Fraudulent'].mean() * 100
            fraud_by_payment = fraud_by_payment.sort_values(ascending=False).head(10)
            
            fig = px.bar(
                x=fraud_by_payment.values, y=fraud_by_payment.index,
                orientation='h', title='Fraud Rate by Payment Method',
                color=fraud_by_payment.values, color_continuous_scale='RdYlGn_r',
                labels={'x': 'Fraud Rate (%)', 'y': 'Payment Method'}
            )
            fig.update_layout(height=400)
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # Box plot - Transaction Amount by Fraud Status
            fig = px.box(
                filtered_df, x='Fraudulent', y='Transaction_Amount',
                title='Transaction Amount Distribution by Fraud Status',
                color='Fraudulent', color_discrete_map={0: '#2ecc71', 1: '#e74c3c'},
                labels={'Fraudulent': 'Is Fraudulent?', 'Transaction_Amount': 'Amount ($)'}
            )
            fig.update_layout(height=400)
            st.plotly_chart(fig, use_container_width=True)
    
    with tab2:
        st.markdown("### Geographic Fraud Analysis")
        
        # Location risk analysis
        location_risk = filtered_df.groupby('Location').agg({
            'Fraudulent': ['mean', 'count']
        }).round(4)
        location_risk.columns = ['Fraud_Rate', 'Transaction_Count']
        location_risk['Fraud_Rate_Pct'] = location_risk['Fraud_Rate'] * 100
        location_risk = location_risk.sort_values('Fraud_Rate', ascending=False).head(20)
        
        col1, col2 = st.columns(2)
        
        with col1:
            fig = px.bar(
                x=location_risk['Fraud_Rate_Pct'].values,
                y=location_risk.index,
                orientation='h',
                title='Top 20 High-Risk Locations',
                color=location_risk['Fraud_Rate_Pct'].values,
                color_continuous_scale='Reds',
                labels={'x': 'Fraud Rate (%)', 'y': 'Location'}
            )
            fig.update_layout(height=600)
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            fig = px.scatter(
                location_risk.head(20),
                x='Transaction_Count',
                y='Fraud_Rate_Pct',
                size='Transaction_Count',
                text=location_risk.head(20).index,
                title='Location Risk Matrix',
                labels={'Transaction_Count': 'Number of Transactions', 'Fraud_Rate_Pct': 'Fraud Rate (%)'}
            )
            fig.update_traces(textposition='top center')
            fig.update_layout(height=600)
            st.plotly_chart(fig, use_container_width=True)
        
        st.dataframe(location_risk, use_container_width=True)
    
    with tab3:
        st.markdown("### Temporal Fraud Analysis")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Fraud rate by hour
            hourly_fraud = filtered_df.groupby('Time_of_Transaction')['Fraudulent'].mean() * 100
            
            fig = px.line(
                x=hourly_fraud.index, y=hourly_fraud.values,
                title='Fraud Rate by Hour of Day',
                markers=True,
                labels={'x': 'Hour of Day', 'y': 'Fraud Rate (%)'}
            )
            fig.add_hline(y=filtered_df['Fraudulent'].mean() * 100, 
                          line_dash="dash", line_color="red",
                          annotation_text="Average Fraud Rate")
            fig.update_layout(height=400)
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # Transaction volume by hour
            hourly_volume = filtered_df.groupby('Time_of_Transaction').size()
            
            fig = px.line(
                x=hourly_volume.index, y=hourly_volume.values,
                title='Transaction Volume by Hour',
                markers=True,
                labels={'x': 'Hour of Day', 'y': 'Number of Transactions'}
            )
            fig.update_layout(height=400)
            st.plotly_chart(fig, use_container_width=True)
        
        # Previous fraud analysis
        fraud_by_prev = filtered_df.groupby('Previous_Fraudulent_Transactions')['Fraudulent'].mean() * 100
        
        fig = px.line(
            x=fraud_by_prev.index, y=fraud_by_prev.values,
            title='Fraud Rate by Previous Fraudulent Transactions',
            markers=True,
            labels={'x': 'Previous Fraudulent Transactions', 'y': 'Fraud Rate (%)'}
        )
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)

elif page == "⚠️ Risk Scoring":
    # Risk Scoring Page
    st.markdown('<div class="sub-header">⚠️ Transaction Risk Scoring</div>', unsafe_allow_html=True)
    
    st.markdown("""
    The risk scoring model evaluates each transaction based on multiple factors:
    - Transaction type (Online Purchase, Bank Transfer = higher risk)
    - Device used (Unknown/Other devices = higher risk)
    - Payment method (Net Banking, UPI = higher risk)
    - Previous fraud history
    - Transaction amount (high amounts = higher risk)
    - Transaction velocity (>10 in 24h = higher risk)
    - Account age (<30 days = higher risk)
    """)
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Average Risk Score", f"{df_clean['Risk_Score'].mean():.2f}")
    with col2:
        high_risk = len(df_clean[df_clean['Risk_Score'] >= 5])
        st.metric("High Risk Transactions (Score ≥5)", f"{high_risk:,}")
    with col3:
        critical_risk = len(df_clean[df_clean['Risk_Score'] >= 8])
        st.metric("Critical Risk Transactions (Score ≥8)", f"{critical_risk:,}")
    
    st.markdown("---")
    
    tab1, tab2, tab3 = st.tabs(["📊 Risk Distribution", "🎯 Risk Factors", "🔍 Transaction Lookup"])
    
    with tab1:
        col1, col2 = st.columns(2)
        
        with col1:
            # Risk score distribution
            risk_counts = df_clean['Risk_Score'].value_counts().sort_index()
            
            fig = px.bar(
                x=risk_counts.index, y=risk_counts.values,
                title='Risk Score Distribution',
                labels={'x': 'Risk Score', 'y': 'Number of Transactions'},
                color=risk_counts.values,
                color_continuous_scale='RdYlGn_r'
            )
            fig.update_layout(height=500)
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # Fraud rate by risk score
            fraud_by_risk = df_clean.groupby('Risk_Score')['Fraudulent'].mean() * 100
            
            fig = px.line(
                x=fraud_by_risk.index, y=fraud_by_risk.values,
                title='Fraud Rate by Risk Score',
                markers=True,
                labels={'x': 'Risk Score', 'y': 'Fraud Rate (%)'}
            )
            fig.add_hline(y=df_clean['Fraudulent'].mean() * 100, 
                          line_dash="dash", line_color="red",
                          annotation_text="Average Fraud Rate")
            fig.update_layout(height=500)
            st.plotly_chart(fig, use_container_width=True)
        
        # Risk score heatmap by category
        st.markdown("### Risk Score Heatmap by Categories")
        
        risk_heatmap = df_clean.groupby(['Transaction_Type', 'Payment_Method'])['Risk_Score'].mean().unstack()
        
        fig = px.imshow(
            risk_heatmap,
            text_auto=True,
            aspect="auto",
            color_continuous_scale='RdYlGn_r',
            title='Average Risk Score by Transaction Type and Payment Method'
        )
        fig.update_layout(height=500)
        st.plotly_chart(fig, use_container_width=True)
    
    with tab2:
        st.markdown("### Top Risk Factors Analysis")
        
        # Calculate risk factor importance
        risk_factors = {
            'Unknown/Other Device': df_clean[df_clean['Device_Used'].isin(['Unknown', 'Other'])]['Fraudulent'].mean() * 100,
            'Online Purchase': df_clean[df_clean['Transaction_Type'] == 'Online Purchase']['Fraudulent'].mean() * 100,
            'Bank Transfer': df_clean[df_clean['Transaction_Type'] == 'Bank Transfer']['Fraudulent'].mean() * 100,
            'Previous Fraud History': df_clean[df_clean['Previous_Fraudulent_Transactions'] > 0]['Fraudulent'].mean() * 100,
            'New Account (<30 days)': df_clean[df_clean['Account_Age'] < 30]['Fraudulent'].mean() * 100,
            'High Transaction Volume (>10/24h)': df_clean[df_clean['Number_of_Transactions_Last_24H'] > 10]['Fraudulent'].mean() * 100,
            'High Amount (>95th percentile)': df_clean[df_clean['Transaction_Amount'] > df_clean['Transaction_Amount'].quantile(0.95)]['Fraudulent'].mean() * 100
        }
        
        risk_df = pd.DataFrame(list(risk_factors.items()), columns=['Risk Factor', 'Fraud Rate (%)'])
        risk_df = risk_df.sort_values('Fraud Rate (%)', ascending=True)
        
        fig = px.bar(
            risk_df,
            x='Fraud Rate (%)',
            y='Risk Factor',
            orientation='h',
            title='Fraud Rate by Risk Factor',
            color='Fraud Rate (%)',
            color_continuous_scale='Reds'
        )
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)
        
        st.markdown("### Risk Score Breakdown")
        
        # Show risk score components
        risk_components = df_clean[['Risk_Score', 'Fraudulent']].copy()
        risk_components['Risk Category'] = pd.cut(
            risk_components['Risk_Score'],
            bins=[0, 2, 4, 6, 8, 100],
            labels=['Very Low', 'Low', 'Medium', 'High', 'Critical']
        )
        
        risk_category_summary = risk_components.groupby('Risk Category').agg({
            'Fraudulent': ['count', 'mean']
        }).round(4)
        risk_category_summary.columns = ['Count', 'Fraud Rate']
        risk_category_summary['Fraud Rate %'] = risk_category_summary['Fraud Rate'] * 100
        
        st.dataframe(risk_category_summary, use_container_width=True)
    
    with tab3:
        st.markdown("### Transaction Risk Lookup")
        
        # Transaction search
        transaction_id = st.text_input("Enter Transaction ID:", placeholder="e.g., T1, T100, T5000")
        
        if transaction_id:
            transaction = df_clean[df_clean['Transaction_ID'] == transaction_id]
            
            if len(transaction) > 0:
                trans = transaction.iloc[0]
                
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.metric("Risk Score", trans['Risk_Score'])
                    st.metric("Fraudulent", "Yes" if trans['Fraudulent'] == 1 else "No")
                
                with col2:
                    st.metric("Transaction Amount", f"${trans['Transaction_Amount']:.2f}")
                    st.metric("Transaction Type", trans['Transaction_Type'])
                
                with col3:
                    st.metric("Device Used", trans['Device_Used'])
                    st.metric("Payment Method", trans['Payment_Method'])
                
                st.markdown("**Risk Factors:**")
                risk_factors_found = []
                
                if trans['Transaction_Type'] in ['Online Purchase', 'Bank Transfer']:
                    risk_factors_found.append("• High-risk transaction type")
                if trans['Device_Used'] in ['Unknown', 'Other']:
                    risk_factors_found.append("• Unknown/Other device")
                if trans['Previous_Fraudulent_Transactions'] > 0:
                    risk_factors_found.append(f"• Previous fraud history ({trans['Previous_Fraudulent_Transactions']} times)")
                if trans['Account_Age'] < 30:
                    risk_factors_found.append("• New account (<30 days old)")
                if trans['Number_of_Transactions_Last_24H'] > 10:
                    risk_factors_found.append("• High transaction velocity")
                
                if risk_factors_found:
                    for factor in risk_factors_found:
                        st.write(factor)
                else:
                    st.success("No major risk factors detected")
            else:
                st.error(f"Transaction ID '{transaction_id}' not found")
        
        # Show high-risk transactions table
        st.markdown("### High-Risk Transactions (Risk Score ≥ 7)")
        high_risk_transactions = df_clean[df_clean['Risk_Score'] >= 7].sort_values('Risk_Score', ascending=False).head(20)
        st.dataframe(high_risk_transactions[['Transaction_ID', 'Transaction_Amount', 'Transaction_Type', 
                                              'Device_Used', 'Payment_Method', 'Risk_Score', 'Fraudulent']], 
                     use_container_width=True)

elif page == "💡 Recommendations":
    # Recommendations Page
    st.markdown('<div class="sub-header">💡 Actionable Recommendations</div>', unsafe_allow_html=True)
    
    st.markdown("""
    Based on the comprehensive analysis of fraud patterns, the following recommendations are proposed to reduce fraudulent transactions:
    """)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### 🚨 Immediate Actions")
        st.info("""
        **1. Implement Real-time Risk Scoring**
        - Deploy the risk scoring model for all incoming transactions
        - Flag transactions with risk score ≥7 for manual review
        - Automatically block transactions with risk score ≥9
        
        **2. Enhanced Verification for High-Risk Categories**
        - Require 2FA for Online Purchase and Bank Transfer transactions
        - Additional verification for Unknown/Other devices
        - Implement OTP verification for Net Banking and UPI payments
        
        **3. Monitor New Accounts**
        - Apply stricter limits for accounts <30 days old
        - Cap transaction amount at $500 for new accounts
        - Limit to 5 transactions per day for new accounts
        """)
        
        st.markdown("### 📊 Monitoring & Alerts")
        st.info("""
        **4. Real-time Alert System**
        - Alert on transactions from high-risk locations (fraud rate >5%)
        - Notify for accounts with >10 transactions in 24 hours
        - Flag transactions with amount >95th percentile
        
        **5. Regular Review Schedule**
        - Daily review of high-risk transactions
        - Weekly pattern analysis updates
        - Monthly model retraining with new data
        """)
    
    with col2:
        st.markdown("### 🔧 System Improvements")
        st.info("""
        **6. Machine Learning Integration**
        - Train ML model on cleaned dataset
        - Implement real-time scoring API
        - Use ensemble methods for better accuracy
        
        **7. Data Quality Enhancements**
        - Standardize device tracking across all platforms
        - Implement required fields for payment methods
        - Add geolocation verification
        
        **8. Customer Education**
        - Send alerts for suspicious activities
        - Provide security tips for online transactions
        - Implement transaction confirmation for high-risk cases
        """)
        
        st.markdown("### 📈 Success Metrics")
        st.info("""
        **Target KPIs:**
        - Reduce fraud rate by 50% within 3 months
        - Decrease false positives by 30%
        - Achieve 95% fraud detection accuracy
        - Reduce manual review time by 40%
        """)
    
    st.markdown("---")
    st.markdown("### 📋 Implementation Roadmap")
    
    roadmap = pd.DataFrame({
        'Phase': ['Phase 1', 'Phase 2', 'Phase 3', 'Phase 4'],
        'Timeline': ['Week 1-2', 'Week 3-4', 'Week 5-6', 'Week 7-8'],
        'Actions': [
            'Deploy risk scoring model, Set up monitoring dashboard',
            'Implement 2FA for high-risk transactions, Train ML model',
            'Integrate real-time scoring API, Deploy alert system',
            'Review and optimize, Train final model, Launch full system'
        ],
        'Success Metrics': [
            'Risk scores calculated for all transactions',
            '50% reduction in fraud attempts',
            '95% detection accuracy achieved',
            '40% reduction in manual review time'
        ]
    })
    
    st.dataframe(roadmap, use_container_width=True, hide_index=True)
    
    st.markdown("---")
    st.markdown("### 📊 Expected Impact")
    
    impact_col1, impact_col2, impact_col3, impact_col4 = st.columns(4)
    
    with impact_col1:
        st.metric("Fraud Reduction", "50%", delta="Target")
    with impact_col2:
        st.metric("Manual Review Reduction", "40%", delta="Target")
    with impact_col3:
        st.metric("Detection Accuracy", "95%", delta="Target")
    with impact_col4:
        st.metric("False Positive Reduction", "30%", delta="Target")
    
    st.markdown("---")
    st.markdown("### 📝 Summary of Findings")
    
    findings = [
        f"• Overall fraud rate: {df_clean['Fraudulent'].mean() * 100:.2f}%",
        f"• Online Purchase and Bank Transfer show highest fraud rates",
        f"• Unknown/Other devices have {df_clean[df_clean['Device_Used'].isin(['Unknown', 'Other'])]['Fraudulent'].mean() * 100:.2f}% fraud rate",
        f"• Accounts with previous fraud history are {df_clean[df_clean['Previous_Fraudulent_Transactions'] > 0]['Fraudulent'].mean() * 100:.2f}% more likely to commit fraud",
        f"• Peak fraud hours: {df_clean.groupby('Time_of_Transaction')['Fraudulent'].mean().idxmax()}:00",
        f"• Top 5 high-risk locations account for {df_clean.groupby('Location')['Fraudulent'].mean().nlargest(5).mean() * 100:.2f}% fraud rate"
    ]
    
    for finding in findings:
        st.warning(finding)

# Footer
st.markdown("---")
st.markdown(
    "<p style='text-align: center; color: gray;'>Fraud Detection Dashboard | Built with Streamlit</p>",
    unsafe_allow_html=True
)