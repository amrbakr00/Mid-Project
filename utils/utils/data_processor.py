"""
Data processing utilities for fraud detection
"""

import pandas as pd
import numpy as np

def load_data(filepath):
    """Load the fraud detection dataset"""
    return pd.read_csv(filepath)

def clean_data(df):
    """Clean the dataset"""
    df_clean = df.copy()
    
    # Handle missing values
    categorical_cols = ['Transaction_Type', 'Device_Used', 'Location', 'Payment_Method']
    for col in categorical_cols:
        mode_value = df_clean[col].mode()[0] if not df_clean[col].mode().empty else 'Unknown'
        df_clean[col] = df_clean[col].fillna(mode_value)
    
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
    
    # Handle outliers
    Q1 = df_clean['Transaction_Amount'].quantile(0.25)
    Q3 = df_clean['Transaction_Amount'].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    df_clean['Transaction_Amount'] = df_clean['Transaction_Amount'].clip(lower=lower_bound, upper=upper_bound)
    
    return df_clean

def calculate_risk_score(row, df_clean):
    """Calculate risk score for a transaction"""
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

def add_risk_scores(df_clean):
    """Add risk scores to the dataset"""
    df_clean['Risk_Score'] = df_clean.apply(lambda row: calculate_risk_score(row, df_clean), axis=1)
    return df_clean