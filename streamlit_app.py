import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Page configuration
st.set_page_config(
    page_title="ML Assignment 2",
    page_icon="📊",
    layout="wide"
)

# Sidebar navigation
st.sidebar.title("Navigation")
page = st.sidebar.radio("Select a page:", ["Home", "Data Analysis", "Model Training", "Predictions"])

if page == "Home":
    st.title("Welcome to ML Assignment 2 🚀")
    st.write("""
    This is your machine learning assignment application built with Streamlit.
    Use the sidebar to navigate between different sections.
    """)
    
    st.info("📌 Features:")
    st.write("- **Data Analysis**: Explore and visualize your dataset")
    st.write("- **Model Training**: Train and evaluate ML models")
    st.write("- **Predictions**: Make predictions on new data")

elif page == "Data Analysis":
    st.header("Data Analysis")
    
    # Upload data
    uploaded_file = st.file_uploader("Upload your CSV file:", type=['csv'])
    
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        st.success("✅ File uploaded successfully!")
        
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Dataset Shape")
            st.write(f"Rows: {df.shape[0]}")
            st.write(f"Columns: {df.shape[1]}")
        
        with col2:
            st.subheader("Data Types")
            st.write(df.dtypes)
        
        st.subheader("First few rows:")
        st.dataframe(df.head(10))
        
        st.subheader("Statistical Summary:")
        st.dataframe(df.describe())
    else:
        st.warning("⚠️ Please upload a CSV file to get started")

elif page == "Model Training":
    st.header("Model Training")
    st.write("Configure and train your ML model here")
    
    col1, col2 = st.columns(2)
    with col1:
        model_type = st.selectbox("Select Model Type:", ["Linear Regression", "Decision Tree", "Random Forest"])
    
    with col2:
        test_size = st.slider("Test Size:", 0.1, 0.5, 0.2)
    
    if st.button("Train Model"):
        st.info("Model training started...")
        # Add your model training code here
        st.success("✅ Model trained successfully!")

elif page == "Predictions":
    st.header("Make Predictions")
    st.write("Use your trained model to make predictions on new data")
    
    # Add prediction interface here
    st.info("Train a model first to make predictions")