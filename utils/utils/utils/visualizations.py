"""
Visualization utilities for fraud detection
"""

import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd

def create_fraud_rate_chart(df, category, title):
    """Create fraud rate bar chart"""
    fraud_rate = df.groupby(category)['Fraudulent'].mean() * 100
    fraud_rate = fraud_rate.sort_values(ascending=False)
    
    fig = px.bar(
        x=fraud_rate.values, y=fraud_rate.index,
        orientation='h',
        title=title,
        color=fraud_rate.values,
        color_continuous_scale='RdYlGn_r',
        labels={'x': 'Fraud Rate (%)', 'y': category}
    )
    return fig

def create_risk_distribution_chart(df):
    """Create risk score distribution chart"""
    risk_counts = df['Risk_Score'].value_counts().sort_index()
    
    fig = px.bar(
        x=risk_counts.index, y=risk_counts.values,
        title='Risk Score Distribution',
        labels={'x': 'Risk Score', 'y': 'Number of Transactions'},
        color=risk_counts.values,
        color_continuous_scale='RdYlGn_r'
    )
    return fig

def create_hourly_analysis_chart(df):
    """Create hourly fraud and volume analysis"""
    hourly_fraud = df.groupby('Time_of_Transaction')['Fraudulent'].mean() * 100
    hourly_volume = df.groupby('Time_of_Transaction').size()
    
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    
    fig.add_trace(
        go.Scatter(x=hourly_fraud.index, y=hourly_fraud.values, name="Fraud Rate (%)", line=dict(color='red')),
        secondary_y=False
    )
    
    fig.add_trace(
        go.Scatter(x=hourly_volume.index, y=hourly_volume.values, name="Transaction Volume", line=dict(color='blue')),
        secondary_y=True
    )
    
    fig.update_layout(title='Hourly Fraud Rate and Transaction Volume', xaxis_title='Hour of Day')
    fig.update_yaxes(title_text="Fraud Rate (%)", secondary_y=False)
    fig.update_yaxes(title_text="Transaction Volume", secondary_y=True)
    
    return fig

def create_correlation_heatmap(df, numeric_cols):
    """Create correlation heatmap"""
    corr_matrix = df[numeric_cols].corr()
    
    fig = px.imshow(
        corr_matrix,
        text_auto=True,
        aspect="auto",
        color_continuous_scale='RdBu',
        title='Feature Correlation Matrix'
    )
    return fig

def create_location_risk_chart(df):
    """Create location risk analysis chart"""
    location_risk = df.groupby('Location').agg({
        'Fraudulent': ['mean', 'count']
    }).round(4)
    location_risk.columns = ['Fraud_Rate', 'Transaction_Count']
    location_risk['Fraud_Rate_Pct'] = location_risk['Fraud_Rate'] * 100
    location_risk = location_risk.sort_values('Fraud_Rate', ascending=False).head(20)
    
    fig = px.scatter(
        location_risk,
        x='Transaction_Count',
        y='Fraud_Rate_Pct',
        size='Transaction_Count',
        text=location_risk.index,
        title='Location Risk Matrix',
        labels={'Transaction_Count': 'Number of Transactions', 'Fraud_Rate_Pct': 'Fraud Rate (%)'}
    )
    fig.update_traces(textposition='top center')
    return fig