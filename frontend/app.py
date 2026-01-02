import streamlit as st
import pandas as pd
import requests
import seaborn as sns
import matplotlib.pyplot as plt

# Configuration
API_URL = "http://backend:8000"  # Docker service name

st.set_page_config(page_title="Forecaster", layout="wide")
st.title("📈 Time Series Forecasting App")

# Sidebar
st.sidebar.header("Controls")
if st.sidebar.button("Check API Health"):
    try:
        res = requests.get(f"{API_URL}/health")
        if res.status_code == 200:
            st.sidebar.success("API is Online")
        else:
            st.sidebar.error("API Error")
    except:
        st.sidebar.error("API Unreachable")

# 1. Data Generation Section
st.header("1. Data Generation")
if st.button("Generate Synthetic Data"):
    with st.spinner("Generating..."):
        response = requests.get(f"{API_URL}/generate-data")
        if response.status_code == 200:
            data = response.json()
            df = pd.DataFrame(data)
            df['ds'] = pd.to_datetime(df['ds'])
            
            st.session_state['data'] = df
            st.success("Data Generated!")
        else:
            st.error("Failed to generate data.")

if 'data' in st.session_state:
    st.dataframe(st.session_state['data'].head())
    
    # Visualization using Seaborn as requested
    fig, ax = plt.subplots(figsize=(10, 5))
    sns.lineplot(data=st.session_state['data'], x='ds', y='y', ax=ax)
    ax.set_title("Historical Data")
    st.pyplot(fig)

# 2. Training Section
st.header("2. Model Training")
model_choice = st.selectbox("Select Model", ["prophet", "sarimax (wip)"])

if st.button("Train Model"):
    with st.spinner("Training model... this may take a moment"):
        payload = {"model_type": model_choice}
        try:
            res = requests.post(f"{API_URL}/train", json=payload)
            if res.status_code == 200:
                result = res.json()
                st.session_state['run_id'] = result['run_id']
                st.success(f"Training Complete! Run ID: {result['run_id']}")
            else:
                st.error(f"Training failed: {res.text}")
        except Exception as e:
            st.error(f"Connection error: {e}")

# 3. Prediction Section
st.header("3. Forecasting")
days_to_predict = st.slider("Days to forecast", 7, 365, 30)

if st.button("Forecast"):
    if 'run_id' not in st.session_state:
        st.warning("Please train a model first.")
    else:
        payload = {"run_id": st.session_state['run_id'], "days": days_to_predict}
        res = requests.post(f"{API_URL}/predict", json=payload)
        
        if res.status_code == 200:
            forecast_data = res.json()
            f_df = pd.DataFrame(forecast_data)
            f_df['ds'] = pd.to_datetime(f_df['ds'])
            
            st.write("Forecast Results:")
            st.dataframe(f_df.head())
            
            # Plotting Forecast
            fig2, ax2 = plt.subplots(figsize=(10, 5))
            sns.lineplot(data=f_df, x='ds', y='yhat', ax=ax2, color='green', label='Prediction')
            ax2.fill_between(f_df['ds'], f_df['yhat_lower'], f_df['yhat_upper'], color='green', alpha=0.1)
            ax2.set_title("Forecast with Confidence Intervals")
            st.pyplot(fig2)
        else:
            st.error("Prediction failed")