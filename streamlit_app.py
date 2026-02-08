import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import os
import warnings
warnings.filterwarnings('ignore')

from model.obesity_dataset_loader import ObesityDatasetLoader
from model.logistic_regression import LogisticRegressionModel
from model.decision_tree import DecisionTreeModel
from model.knn import KNNModel
from model.naive_bayes import NaiveBayesModel
from model.random_forest import RandomForestModel
from model.xgboost_model import XGBoostModel

# Page configuration
st.set_page_config(
    page_title="ML Assignment 2",
    page_icon="📊",
    layout="wide"
)

# Sidebar navigation
st.sidebar.title("Navigation")
page = st.sidebar.radio("Select a page:", ["Home", "Data Analysis", "Model Training", "Model Validation", "Predictions"])


def load_uploaded_df(uploaded_file):
    try:
        return pd.read_csv(uploaded_file)
    except Exception:
        return None

if page == "Home":
    st.title("Welcome to ML Assignment 2 🚀")
    st.write("""
    This is your machine learning assignment application built with Streamlit.
    Use the sidebar to navigate between different sections.
    """)
    
    st.info("📌 Features:")
    st.write("- **Data Analysis**: Explore and visualize your dataset")
    st.write("- **Model Training**: Train and evaluate Logistic Regression with metrics")
    st.write("- **Predictions**: Make predictions on new data using trained model")

elif page == "Data Analysis":
    st.header("📊 Exploratory Data Analysis (EDA)")
    
    uploaded_file = st.file_uploader("Upload your CSV file:", type=['csv'])
    use_default = st.checkbox("Use default Obesity dataset (local path)")
    
    df = None
    if uploaded_file is not None:
        df = load_uploaded_df(uploaded_file)
        if df is None:
            st.error("❌ Couldn't read uploaded file")
    elif use_default:
        loader = ObesityDatasetLoader()
        df = loader.load_data()
    
    if df is not None:
        st.success("✅ Data loaded successfully!")
        
        # Basic Information
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total Rows", df.shape[0])
        with col2:
            st.metric("Total Columns", df.shape[1])
        with col3:
            st.metric("Missing Values", df.isnull().sum().sum())
        
        st.subheader("Dataset Info:")
        col1, col2 = st.columns(2)
        with col1:
            st.write("**Data Types:**")
            st.write(df.dtypes)
        with col2:
            st.write("**First few rows:**")
            st.dataframe(df.head(5))
        
        st.subheader("📈 Statistical Summary:")
        st.dataframe(df.describe())
        
        # Target Variable Distribution
        if "NObeyesdad" in df.columns:
            st.subheader("🎯 Target Variable Distribution (Obesity Classes):")
            target_counts = df["NObeyesdad"].value_counts()
            
            col1, col2 = st.columns(2)
            with col1:
                fig, ax = plt.subplots(figsize=(8, 5))
                target_counts.plot(kind='bar', ax=ax, color='steelblue')
                ax.set_title("Obesity Class Distribution")
                ax.set_xlabel("Obesity Class")
                ax.set_ylabel("Count")
                plt.xticks(rotation=45, ha='right')
                plt.tight_layout()
                st.pyplot(fig)
            
            with col2:
                fig, ax = plt.subplots(figsize=(8, 5))
                target_counts.plot(kind='pie', ax=ax, autopct='%1.1f%%')
                ax.set_title("Obesity Class Proportion")
                ax.set_ylabel("")
                plt.tight_layout()
                st.pyplot(fig)
        
        # Numerical Features
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        if 'NObeyesdad' in numeric_cols:
            numeric_cols.remove('NObeyesdad')
        
        if numeric_cols:
            st.subheader("📊 Numerical Features Distributions:")
            
            # Histogram
            st.write("**Histograms:**")
            fig, axes = plt.subplots(2, 3, figsize=(15, 10))
            axes = axes.flatten()
            
            for idx, col in enumerate(numeric_cols[:6]):
                axes[idx].hist(df[col].dropna(), bins=30, color='skyblue', edgecolor='black')
                axes[idx].set_title(f"Distribution of {col}")
                axes[idx].set_xlabel(col)
                axes[idx].set_ylabel("Frequency")
            
            # Hide extra subplots
            for idx in range(len(numeric_cols[:6]), 6):
                axes[idx].set_visible(False)
            
            plt.tight_layout()
            st.pyplot(fig)
            
            # Box plots
            st.write("**Box Plots (Numerical Features):**")
            fig, axes = plt.subplots(1, 2, figsize=(15, 5))
            
            # All features box plot
            df[numeric_cols].boxplot(ax=axes[0], rot=45)
            axes[0].set_title("Box Plot of All Numerical Features")
            axes[0].set_ylabel("Values")
            
            # Correlation with target if available
            if "NObeyesdad" in df.columns:
                # Create encoded version for correlation
                from sklearn.preprocessing import LabelEncoder
                df_encoded = df.copy()
                le = LabelEncoder()
                df_encoded["NObeyesdad_encoded"] = le.fit_transform(df["NObeyesdad"])
                
                corr = df_encoded[numeric_cols + ["NObeyesdad_encoded"]].corr()["NObeyesdad_encoded"].drop("NObeyesdad_encoded").sort_values()
                corr.plot(kind='barh', ax=axes[1], color=['red' if x < 0 else 'green' for x in corr.values])
                axes[1].set_title("Correlation with Obesity Class")
                axes[1].set_xlabel("Correlation")
            
            plt.tight_layout()
            st.pyplot(fig)
        
        # Categorical Features
        categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
        if 'NObeyesdad' in categorical_cols:
            categorical_cols.remove('NObeyesdad')
        
        if categorical_cols:
            st.subheader("🏷️ Categorical Features:")
            
            fig, axes = plt.subplots(2, 2, figsize=(15, 10))
            axes = axes.flatten()
            
            for idx, col in enumerate(categorical_cols[:4]):
                counts = df[col].value_counts()
                axes[idx].bar(counts.index, counts.values, color='coral', edgecolor='black')
                axes[idx].set_title(f"{col} Distribution")
                axes[idx].set_xlabel(col)
                axes[idx].set_ylabel("Count")
                plt.setp(axes[idx].xaxis.get_majorticklabels(), rotation=45, ha='right')
            
            # Hide extra subplots
            for idx in range(len(categorical_cols[:4]), 4):
                axes[idx].set_visible(False)
            
            plt.tight_layout()
            st.pyplot(fig)
        
        # Correlation Heatmap
        if len(numeric_cols) > 1:
            st.subheader("🔥 Correlation Heatmap (Numerical Features):")
            fig, ax = plt.subplots(figsize=(12, 8))
            corr_matrix = df[numeric_cols].corr()
            sns.heatmap(corr_matrix, annot=True, fmt='.2f', cmap='coolwarm', center=0, ax=ax)
            ax.set_title("Correlation Matrix")
            plt.tight_layout()
            st.pyplot(fig)
        
        # Feature vs Target
        if "NObeyesdad" in df.columns and numeric_cols:
            st.subheader("📍 Key Features vs Obesity Class:")
            
            fig, axes = plt.subplots(1, 2, figsize=(15, 5))
            
            # Weight vs Obesity
            if "Weight" in df.columns:
                df.boxplot(column='Weight', by='NObeyesdad', ax=axes[0])
                axes[0].set_title("Weight vs Obesity Class")
                axes[0].set_xlabel("Obesity Class")
                axes[0].set_ylabel("Weight")
                plt.sca(axes[0])
                plt.xticks(rotation=45, ha='right')
            
            # Height vs Obesity
            if "Height" in df.columns:
                df.boxplot(column='Height', by='NObeyesdad', ax=axes[1])
                axes[1].set_title("Height vs Obesity Class")
                axes[1].set_xlabel("Obesity Class")
                axes[1].set_ylabel("Height")
                plt.sca(axes[1])
                plt.xticks(rotation=45, ha='right')
            
            plt.tight_layout()
            st.pyplot(fig)
        
        # Missing Values Analysis
        if df.isnull().sum().sum() > 0:
            st.subheader("⚠️ Missing Values Analysis:")
            missing = df.isnull().sum()
            missing = missing[missing > 0].sort_values(ascending=False)
            
            fig, ax = plt.subplots(figsize=(10, 5))
            missing.plot(kind='barh', ax=ax, color='red')
            ax.set_title("Missing Values Count")
            ax.set_xlabel("Count")
            plt.tight_layout()
            st.pyplot(fig)
    else:
        st.info("ℹ️ Upload a CSV file or check the box to use default dataset")

elif page == "Model Training":
    st.header("Model Training — Classification Models")
    st.write("Train and evaluate classification models with evaluation metrics: Accuracy, AUC, Precision, Recall, F1, MCC")
    
    # Model Selection
    model_choice = st.radio("Select Model:", ["Logistic Regression", "Decision Tree", "k-Nearest Neighbors", "Naive Bayes", "Random Forest", "XGBoost"], horizontal=True)
    
    uploaded_file = st.file_uploader("Upload training CSV (optional):", type=['csv'], key='train_upload')
    test_size = st.slider("Test Size:", 0.1, 0.5, 0.2)
    
    # Model-specific parameters
    if model_choice == "Logistic Regression":
        max_iter = st.number_input("Max Iterations:", min_value=100, max_value=5000, value=1000, step=100)
        model_params = {"max_iter": int(max_iter)}
    elif model_choice == "Decision Tree":
        col1, col2, col3 = st.columns(3)
        with col1:
            max_depth = st.number_input("Max Depth:", min_value=1, max_value=50, value=15, step=1)
        with col2:
            min_samples_split = st.number_input("Min Samples Split:", min_value=2, max_value=100, value=2, step=1)
        with col3:
            min_samples_leaf = st.number_input("Min Samples Leaf:", min_value=1, max_value=50, value=1, step=1)
        model_params = {
            "max_depth": int(max_depth),
            "min_samples_split": int(min_samples_split),
            "min_samples_leaf": int(min_samples_leaf),
        }
    elif model_choice == "k-Nearest Neighbors":
        col1, col2, col3 = st.columns(3)
        with col1:
            n_neighbors = st.number_input("k (neighbors):", min_value=1, max_value=50, value=5, step=1)
        with col2:
            weights = st.selectbox("Weights:", ["uniform", "distance"])
        with col3:
            metric = st.selectbox("Distance Metric:", ["minkowski", "euclidean", "manhattan"])
        model_params = {
            "n_neighbors": int(n_neighbors),
            "weights": weights,
            "metric": metric,
        }
    else:  # Naive Bayes
        var_smoothing = st.slider("Variance Smoothing:", 1e-10, 1e-8, 1e-9, format="%.0e")
        model_params = {
            "var_smoothing": float(var_smoothing),
        }
    
    if model_choice == "Random Forest":
        col1, col2, col3 = st.columns(3)
        with col1:
            n_estimators = st.number_input("Number of Trees:", min_value=10, max_value=500, value=100, step=10)
        with col2:
            max_depth_rf = st.number_input("Max Depth:", min_value=1, max_value=50, value=20, step=1)
        with col3:
            min_samples_split_rf = st.number_input("Min Samples Split:", min_value=2, max_value=100, value=2, step=1)
        model_params = {
            "n_estimators": int(n_estimators),
            "max_depth": int(max_depth_rf) if max_depth_rf > 0 else None,
            "min_samples_split": int(min_samples_split_rf),
        }
    elif model_choice == "XGBoost":
        col1, col2, col3 = st.columns(3)
        with col1:
            n_estimators_xgb = st.number_input("Number of Boosting Rounds:", min_value=10, max_value=500, value=100, step=10)
        with col2:
            learning_rate_xgb = st.slider("Learning Rate:", 0.01, 0.5, 0.1)
        with col3:
            max_depth_xgb = st.number_input("Max Depth:", min_value=1, max_value=20, value=6, step=1)
        model_params = {
            "n_estimators": int(n_estimators_xgb),
            "learning_rate": float(learning_rate_xgb),
            "max_depth": int(max_depth_xgb),
        }
    
    if st.button("🚀 Train Model"):
        with st.spinner("Loading dataset..."):
            loader = ObesityDatasetLoader()
            if uploaded_file is not None:
                df_train = load_uploaded_df(uploaded_file)
                if df_train is None:
                    st.error("❌ Could not read uploaded training file")
                else:
                    loader.df = df_train
            else:
                df_train = loader.load_data()
                if df_train is None:
                    st.error("❌ No dataset available for training")
        
        with st.spinner("Preprocessing data..."):
            X, y = loader.preprocess_data()
            X_train, X_test, y_train, y_test = loader.train_test_split_data(test_size=test_size)
        
        with st.spinner(f"Training {model_choice}..."):
            if model_choice == "Logistic Regression":
                model = LogisticRegressionModel()
                model.train(X_train, y_train, **model_params)
            elif model_choice == "Decision Tree":
                model = DecisionTreeModel()
                model.train(X_train, y_train, **model_params)
            elif model_choice == "k-Nearest Neighbors":
                model = KNNModel()
                model.train(X_train, y_train, **model_params)
            elif model_choice == "Naive Bayes":
                model = NaiveBayesModel()
                model.train(X_train, y_train, **model_params)
            elif model_choice == "Random Forest":
                model = RandomForestModel()
                model.train(X_train, y_train, **model_params)
            else:  # XGBoost
                model = XGBoostModel()
                model.train(X_train, y_train, **model_params)
            
            metrics = model.evaluate(X_test, y_test)
        
        st.success(f"✅ {model_choice} Training complete!")
        
        st.subheader("📊 Evaluation Metrics:")
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Accuracy", f"{metrics['accuracy']:.4f}")
        with col2:
            if metrics["auc"] is not None:
                st.metric("AUC (macro)", f"{metrics['auc']:.4f}")
            else:
                st.metric("AUC", "N/A")
        with col3:
            st.metric("Precision (macro)", f"{metrics['precision_macro']:.4f}")
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Recall (macro)", f"{metrics['recall_macro']:.4f}")
        with col2:
            st.metric("F1 (macro)", f"{metrics['f1_macro']:.4f}")
        with col3:
            st.metric("MCC", f"{metrics['mcc']:.4f}" if metrics['mcc'] is not None else "N/A")
        
        st.subheader("📋 Classification Report:")
        st.json(metrics["report"])
        
        st.subheader("🔲 Confusion Matrix:")
        st.write(metrics["confusion_matrix"])
        
        # Feature Importance for Decision Tree, Random Forest, and XGBoost
        if model_choice in ["Decision Tree", "Random Forest", "XGBoost"]:
            st.subheader(f"📊 Feature Importance ({model_choice}):")
            feature_importance = model.get_feature_importance()
            feature_cols = loader.get_feature_columns()
            importance_df = pd.DataFrame({
                "Feature": feature_cols,
                "Importance": feature_importance
            }).sort_values("Importance", ascending=False)
            
            fig, ax = plt.subplots(figsize=(10, 6))
            color_map = {'Decision Tree': 'steelblue', 'Random Forest': 'forestgreen', 'XGBoost': 'darkorange'}
            ax.barh(importance_df["Feature"], importance_df["Importance"], color=color_map.get(model_choice, 'steelblue'))
            ax.set_xlabel("Importance")
            ax.set_title(f"{model_choice} Feature Importance")
            plt.tight_layout()
            st.pyplot(fig)
            
            st.dataframe(importance_df)
        
        st.subheader("💾 Save Model:")
        model_names = {
            "Logistic Regression": "logistic_regression_model",
            "Decision Tree": "decision_tree_model",
            "k-Nearest Neighbors": "knn_model",
            "Naive Bayes": "naive_bayes_model",
            "Random Forest": "random_forest_model",
            "XGBoost": "xgboost_model"
        }
        default_path = f"model/{model_names[model_choice]}.joblib"
        save_path = st.text_input("Model save path:", value=default_path)
        if st.button("Save Model"):
            extras = {
                "label_encoders": loader.label_encoders,
                "feature_columns": loader.get_feature_columns(),
            }
            if model_choice in ["Decision Tree", "Random Forest", "XGBoost"]:
                extras["feature_importance"] = model.get_feature_importance()
            
            model.save(save_path, extras=extras)
            st.success(f"✅ Model saved to {save_path}")

elif page == "Model Validation":
    st.header("🧪 Model Validation & Comparison")
    st.write("Test and compare all 6 classification models on the same dataset")
    
    # Load dataset
    loader = ObesityDatasetLoader()
    df = loader.load_data()
    
    if df is None:
        st.error("❌ Could not load dataset")
    else:
        st.success("✅ Dataset loaded")
        
        # Preprocess and split data
        with st.spinner("Preprocessing data..."):
            X, y = loader.preprocess_data()
            X_train, X_test, y_train, y_test = loader.train_test_split_data(test_size=0.2)
        
        st.write(f"Training set: {X_train.shape[0]} samples | Test set: {X_test.shape[0]} samples")
        
        # Initialize models with default parameters
        models_to_test = {
            "Logistic Regression": LogisticRegressionModel(),
            "Decision Tree": DecisionTreeModel(),
            "k-Nearest Neighbors": KNNModel(),
            "Naive Bayes": NaiveBayesModel(),
            "Random Forest": RandomForestModel(),
            "XGBoost": XGBoostModel(),
        }
        
        model_params = {
            "Logistic Regression": {"max_iter": 1000},
            "Decision Tree": {"max_depth": 15, "min_samples_split": 2, "min_samples_leaf": 1},
            "k-Nearest Neighbors": {"n_neighbors": 5, "weights": "uniform", "metric": "minkowski"},
            "Naive Bayes": {"var_smoothing": 1e-9},
            "Random Forest": {"n_estimators": 100, "max_depth": 20, "min_samples_split": 2},
            "XGBoost": {"n_estimators": 100, "learning_rate": 0.1, "max_depth": 6},
        }
        
        if st.button("🚀 Run Validation Tests"):
            import time
            
            results = {}
            training_times = {}
            inference_times = {}
            
            with st.spinner("Training and evaluating all models..."):
                for model_name, model in models_to_test.items():
                    # Training with timing
                    start_train = time.time()
                    model.train(X_train, y_train, **model_params[model_name])
                    train_time = time.time() - start_train
                    training_times[model_name] = train_time
                    
                    # Evaluation with timing
                    start_infer = time.time()
                    metrics = model.evaluate(X_test, y_test)
                    infer_time = time.time() - start_infer
                    inference_times[model_name] = infer_time
                    
                    results[model_name] = metrics
            
            st.success("✅ Validation complete!")
            
            # Create comparison dataframe
            comparison_data = []
            for model_name, metrics in results.items():
                comparison_data.append({
                    "Model": model_name,
                    "Accuracy": f"{metrics['accuracy']:.4f}",
                    "AUC": f"{metrics['auc']:.4f}" if metrics['auc'] is not None else "N/A",
                    "Precision": f"{metrics['precision_macro']:.4f}",
                    "Recall": f"{metrics['recall_macro']:.4f}",
                    "F1-Score": f"{metrics['f1_macro']:.4f}",
                    "MCC": f"{metrics['mcc']:.4f}" if metrics['mcc'] is not None else "N/A",
                    "Train Time (s)": f"{training_times[model_name]:.4f}",
                    "Infer Time (s)": f"{inference_times[model_name]:.4f}",
                })
            
            comparison_df = pd.DataFrame(comparison_data)
            
            st.subheader("📊 Model Comparison Table:")
            st.dataframe(comparison_df, width='stretch')
            
            # Extract numeric values for plotting
            accuracy_scores = {name: float(results[name]['accuracy']) for name in results}
            auc_scores = {name: float(results[name]['auc']) if results[name]['auc'] is not None else 0 for name in results}
            f1_scores = {name: float(results[name]['f1_macro']) for name in results}
            precision_scores = {name: float(results[name]['precision_macro']) for name in results}
            recall_scores = {name: float(results[name]['recall_macro']) for name in results}
            
            # Performance Metrics Visualization
            st.subheader("📈 Model Performance Comparison:")
            
            col1, col2 = st.columns(2)
            
            with col1:
                fig, ax = plt.subplots(figsize=(12, 6))
                x_pos = np.arange(len(accuracy_scores))
                colors = plt.cm.Set3(np.linspace(0, 1, len(accuracy_scores)))
                ax.bar(x_pos, accuracy_scores.values(), color=colors, edgecolor='black', linewidth=1.5)
                ax.set_xlabel("Model", fontsize=12, fontweight='bold')
                ax.set_ylabel("Accuracy", fontsize=12, fontweight='bold')
                ax.set_title("Accuracy Comparison", fontsize=14, fontweight='bold')
                ax.set_xticks(x_pos)
                ax.set_xticklabels(accuracy_scores.keys(), rotation=45, ha='right')
                ax.set_ylim([0, 1])
                for i, v in enumerate(accuracy_scores.values()):
                    ax.text(i, v + 0.02, f'{v:.3f}', ha='center', fontweight='bold')
                plt.tight_layout()
                st.pyplot(fig)
            
            with col2:
                fig, ax = plt.subplots(figsize=(12, 6))
                x_pos = np.arange(len(f1_scores))
                colors = plt.cm.Set2(np.linspace(0, 1, len(f1_scores)))
                ax.bar(x_pos, f1_scores.values(), color=colors, edgecolor='black', linewidth=1.5)
                ax.set_xlabel("Model", fontsize=12, fontweight='bold')
                ax.set_ylabel("F1-Score", fontsize=12, fontweight='bold')
                ax.set_title("F1-Score Comparison", fontsize=14, fontweight='bold')
                ax.set_xticks(x_pos)
                ax.set_xticklabels(f1_scores.keys(), rotation=45, ha='right')
                ax.set_ylim([0, 1])
                for i, v in enumerate(f1_scores.values()):
                    ax.text(i, v + 0.02, f'{v:.3f}', ha='center', fontweight='bold')
                plt.tight_layout()
                st.pyplot(fig)
            
            # Multi-metric comparison line chart
            st.subheader("📊 Multi-Metric Performance Radar:")
            metrics_list = ["Accuracy", "Precision", "Recall", "F1-Score"]
            
            fig, ax = plt.subplots(figsize=(12, 6))
            x_pos = np.arange(len(metrics_list))
            width = 0.13
            colors = plt.cm.Set3(np.linspace(0, 1, len(results)))
            
            for idx, (model_name, metrics) in enumerate(results.items()):
                values = [
                    metrics['accuracy'],
                    metrics['precision_macro'],
                    metrics['recall_macro'],
                    metrics['f1_macro']
                ]
                ax.bar(x_pos + idx * width, values, width, label=model_name, color=colors[idx], edgecolor='black')
            
            ax.set_xlabel("Metric", fontsize=12, fontweight='bold')
            ax.set_ylabel("Score", fontsize=12, fontweight='bold')
            ax.set_title("Multi-Metric Comparison Across All Models", fontsize=14, fontweight='bold')
            ax.set_xticks(x_pos + width * 2.5)
            ax.set_xticklabels(metrics_list)
            ax.set_ylim([0, 1])
            ax.legend(loc='lower right', fontsize=9)
            ax.grid(axis='y', alpha=0.3)
            plt.tight_layout()
            st.pyplot(fig)
            
            # Training & Inference Time Comparison
            st.subheader("⏱️ Execution Time Comparison:")
            
            col1, col2 = st.columns(2)
            
            with col1:
                fig, ax = plt.subplots(figsize=(10, 6))
                models = list(training_times.keys())
                times = list(training_times.values())
                colors = plt.cm.Pastel1(np.linspace(0, 1, len(models)))
                ax.barh(models, times, color=colors, edgecolor='black', linewidth=1.5)
                ax.set_xlabel("Time (seconds)", fontsize=11, fontweight='bold')
                ax.set_title("Training Time per Model", fontsize=12, fontweight='bold')
                for i, v in enumerate(times):
                    ax.text(v + 0.001, i, f'{v:.4f}s', va='center', fontweight='bold')
                plt.tight_layout()
                st.pyplot(fig)
            
            with col2:
                fig, ax = plt.subplots(figsize=(10, 6))
                models = list(inference_times.keys())
                times = list(inference_times.values())
                colors = plt.cm.Pastel2(np.linspace(0, 1, len(models)))
                ax.barh(models, times, color=colors, edgecolor='black', linewidth=1.5)
                ax.set_xlabel("Time (seconds)", fontsize=11, fontweight='bold')
                ax.set_title("Inference Time per Model", fontsize=12, fontweight='bold')
                for i, v in enumerate(times):
                    ax.text(v + 0.0001, i, f'{v:.4f}s', va='center', fontweight='bold')
                plt.tight_layout()
                st.pyplot(fig)
            
            # Performance Analysis & Insights
            st.subheader("💡 Model Analysis & Insights:")
            
            col1, col2, col3 = st.columns(3)
            
            best_accuracy_model = max(accuracy_scores, key=accuracy_scores.get)
            best_f1_model = max(f1_scores, key=f1_scores.get)
            fastest_train_model = min(training_times, key=training_times.get)
            
            with col1:
                st.info(f"🥇 **Best Accuracy:** {best_accuracy_model}\n\n{accuracy_scores[best_accuracy_model]:.4f}")
            
            with col2:
                st.info(f"📊 **Best F1-Score:** {best_f1_model}\n\n{f1_scores[best_f1_model]:.4f}")
            
            with col3:
                st.info(f"⚡ **Fastest Training:** {fastest_train_model}\n\n{training_times[fastest_train_model]:.4f}s")
            
            # Detailed Analysis
            st.subheader("📋 Detailed Model Analysis:")
            
            analysis_text = f"""
            **Key Findings:**
            
            1. **Best Overall Performance:** {best_accuracy_model} achieved the highest accuracy of {accuracy_scores[best_accuracy_model]:.4f}
            
            2. **Best Balanced Model (F1-Score):** {best_f1_model} with F1-score of {f1_scores[best_f1_model]:.4f}
                - This metric is useful when you need balance between precision and recall
            
            3. **Fastest Training:** {fastest_train_model} trained in just {training_times[fastest_train_model]:.4f} seconds
                - Consider this for real-time applications
            
            4. **Model Categories:**
               - **Linear Models:** Logistic Regression (simple, interpretable)
               - **Tree-based:** Decision Tree (interpretable, prone to overfitting)
               - **Ensemble Methods:** Random Forest, XGBoost (usually best performance)
               - **Distance-based:** k-NN (memory-intensive but effective)
               - **Probabilistic:** Naive Bayes (fast, works well with limited data)
            
            5. **Recommendations:**
               - For **production use:** Consider {best_accuracy_model} for accuracy or {best_f1_model} for balanced performance
               - For **fast inference:** Use {fastest_train_model}
               - For **interpretability:** Use Decision Tree or Logistic Regression
               - For **best generalization:** Ensemble methods (RF, XGBoost) typically perform better
            
            6. **Metric Definitions:**
               - **Accuracy:** Overall correctness of predictions
               - **Precision:** When predicting positive, how often correct (important: avoid false positives)
               - **Recall:** How many actual positives are found (important: avoid false negatives)
               - **F1-Score:** Harmonic mean of precision and recall (balanced metric)
               - **AUC:** Area under ROC curve (probability model quality across thresholds)
               - **MCC:** Matthews Correlation Coefficient (robust, works well with imbalanced data)
            """
            
            st.markdown(analysis_text)
            
            # Store results in session for later use
            st.session_state["validation_results"] = results
            st.session_state["comparison_df"] = comparison_df

elif page == "Predictions":
    st.header("Make Predictions")
    st.write("Load a trained model and upload a CSV file to get predictions")
    
    col1, col2 = st.columns(2)
    with col1:
        model_file = st.text_input("Model file path:", value="model/logistic_regression_model.joblib")
    with col2:
        if st.button("🔄 Load Model"):
            if not os.path.exists(model_file):
                st.error(f"❌ Model file not found: {model_file}")
            else:
                try:
                    payload = joblib.load(model_file)
                    st.session_state["model_payload"] = payload
                    st.success("✅ Model loaded into session")
                except Exception as e:
                    st.error(f"❌ Error loading model: {str(e)}")
    
    if "model_payload" not in st.session_state:
        st.info("ℹ️ Load a trained model first using the button above")
    else:
        st.success("✅ Model ready for predictions")
        pred_upload = st.file_uploader("Upload CSV for predictions:", type=['csv'], key='pred_upload')
        
        if pred_upload is not None:
            payload = st.session_state["model_payload"]
            df_pred = load_uploaded_df(pred_upload)
            
            if df_pred is None:
                st.error("❌ Could not read prediction file")
            else:
                feature_cols = payload.get("feature_columns")
                label_encoders = payload.get("label_encoders", {})
                model = payload.get("model")
                
                missing = [c for c in feature_cols if c not in df_pred.columns]
                if missing:
                    st.error(f"❌ Missing columns in uploaded data: {missing}")
                else:
                    Xp = df_pred[feature_cols].copy()
                    
                    # Apply encoders where available
                    for col, le in label_encoders.items():
                        if col == "NObeyesdad":
                            continue
                        if col in Xp.columns:
                            try:
                                Xp[col] = le.transform(Xp[col])
                            except Exception:
                                st.warning(f"⚠️ Could not encode column {col}; leaving as-is")
                    
                    preds = model.predict(Xp)
                    
                    # Decode target labels if encoder available
                    le_target = label_encoders.get("NObeyesdad")
                    if le_target is not None:
                        preds_decoded = le_target.inverse_transform(preds)
                    else:
                        preds_decoded = preds
                    
                    df_out = df_pred.copy()
                    df_out["prediction"] = preds_decoded
                    
                    st.subheader("🎯 Predictions:")
                    st.dataframe(df_out)
                    
                    # Offer download
                    csv = df_out.to_csv(index=False)
                    st.download_button(
                        label="📥 Download predictions as CSV",
                        data=csv,
                        file_name="predictions.csv",
                        mime="text/csv",
                    )