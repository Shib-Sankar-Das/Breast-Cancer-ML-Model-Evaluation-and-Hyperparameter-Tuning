import streamlit as st
import pandas as pd
import numpy as np
import cloudpickle
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import os
import warnings
from pathlib import Path

warnings.filterwarnings('ignore')

# Fix for PyArrow compatibility issues
def fix_dataframe_types(df):
    """Fix dataframe types for PyArrow compatibility"""
    df = df.copy()
    for col in df.columns:
        if df[col].dtype == 'object':
            # Try to convert object columns to string
            df[col] = df[col].astype(str)
        elif str(df[col].dtype).startswith('float'):
            # Ensure float columns are standard float64
            df[col] = df[col].astype('float64')
        elif str(df[col].dtype).startswith('int'):
            # Ensure int columns are standard int64
            df[col] = df[col].astype('int64')
    return df

# Page configuration
st.set_page_config(
    page_title="Breast Cancer ML Models",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        text-align: center;
        margin: 0.5rem;
    }
    .stButton > button {
        width: 100%;
        background-color: #1f77b4;
        color: white;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_data
def load_data():
    """Load the breast cancer dataset"""
    try:
        data = pd.read_csv('Data/breast-cancer.csv')
        return data
    except FileNotFoundError:
        st.error("Dataset not found! Please ensure 'Data/breast-cancer.csv' exists.")
        return None

@st.cache_data
def load_model(model_path):
    """Load a trained model using cloudpickle"""
    try:
        with open(model_path, 'rb') as f:
            model = cloudpickle.load(f)
        return model
    except FileNotFoundError:
        st.error(f"Model file not found: {model_path}")
        return None
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        return None

@st.cache_data
def load_preprocessing_objects():
    """Load preprocessing objects"""
    try:
        return load_model('models/preprocessing_objects.pkl')
    except:
        return None

@st.cache_data
def load_results_summary():
    """Load results summary"""
    try:
        return load_model('models/results_summary.pkl')
    except:
        return None

def main():
    st.markdown('<h1 class="main-header">🔬 Breast Cancer ML Model Dashboard</h1>', unsafe_allow_html=True)
    
    # Sidebar navigation
    st.sidebar.title("Navigation")
    page = st.sidebar.selectbox(
        "Choose a page:",
        ["Home", "Data Exploration", "Model Comparison", "Model Prediction", "About"]
    )
    
    if page == "Home":
        show_home_page()
    elif page == "Data Exploration":
        show_data_exploration()
    elif page == "Model Comparison":
        show_model_comparison()
    elif page == "Model Prediction":
        show_model_prediction()
    elif page == "About":
        show_about_page()

def show_home_page():
    st.header("Welcome to the Breast Cancer ML Dashboard")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        <div class="metric-card">
            <h3>🎯 Algorithms Implemented</h3>
            <p>13 different ML algorithms including clustering, ensemble methods, and regularization techniques</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="metric-card">
            <h3>📊 Features</h3>
            <p>Comprehensive EDA, hyperparameter tuning, and performance evaluation</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
        <div class="metric-card">
            <h3>🚀 Optimized</h3>
            <p>Efficient implementation with low CPU and RAM usage</p>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Load and display dataset overview
    data = load_data()
    if data is not None:
        st.subheader("📈 Dataset Overview")
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Total Samples", len(data))
        with col2:
            st.metric("Features", len(data.columns) - 2)
        with col3:
            malignant_count = (data['diagnosis'] == 'M').sum()
            st.metric("Malignant Cases", malignant_count)
        with col4:
            benign_count = (data['diagnosis'] == 'B').sum()
            st.metric("Benign Cases", benign_count)
        
        # Quick visualization
        fig = px.pie(
            values=[malignant_count, benign_count],
            names=['Malignant', 'Benign'],
            title="Distribution of Diagnosis",
            color_discrete_sequence=['#ff6b6b', '#4ecdc4']
        )
        st.plotly_chart(fig, use_container_width=True)

def show_data_exploration():
    st.header("📊 Data Exploration")
    
    data = load_data()
    if data is None:
        return
    
    # Data overview
    st.subheader("Dataset Overview")
    
    col1, col2 = st.columns(2)
    with col1:
        st.write("**First 5 rows:**")
        display_df = fix_dataframe_types(data.head())
        st.dataframe(display_df)
    
    with col2:
        st.write("**Dataset Info:**")
        st.write(f"Shape: {data.shape}")
        st.write(f"Missing values: {data.isnull().sum().sum()}")
        st.write("**Data types:**")
        dtype_counts = data.dtypes.value_counts()
        st.write(dtype_counts)
    
    # Statistical summary
    st.subheader("Statistical Summary")
    numerical_cols = data.select_dtypes(include=[np.number]).columns.tolist()
    if 'id' in numerical_cols:
        numerical_cols.remove('id')
    
    summary_df = fix_dataframe_types(data[numerical_cols].describe())
    st.dataframe(summary_df)
    
    # Feature distributions
    st.subheader("Feature Distributions")
    
    # Select features for visualization
    feature_options = [col for col in data.columns if col not in ['id', 'diagnosis']]
    selected_features = st.multiselect(
        "Select features to visualize:",
        feature_options,
        default=feature_options[:4]
    )
    
    if selected_features:
        n_cols = min(2, len(selected_features))
        n_rows = (len(selected_features) + n_cols - 1) // n_cols
        
        fig = make_subplots(
            rows=n_rows, cols=n_cols,
            subplot_titles=selected_features
        )
        
        for i, feature in enumerate(selected_features):
            row = i // n_cols + 1
            col = i % n_cols + 1
            
            # Histogram by diagnosis
            for diagnosis in ['M', 'B']:
                fig.add_trace(
                    go.Histogram(
                        x=data[data['diagnosis'] == diagnosis][feature],
                        name=f'{diagnosis} - {feature}',
                        opacity=0.7
                    ),
                    row=row, col=col
                )
        
        fig.update_layout(height=300*n_rows, showlegend=True)
        st.plotly_chart(fig, use_container_width=True)
    
    # Correlation analysis
    st.subheader("Correlation Analysis")
    
    # Select subset of features for correlation
    mean_features = [col for col in data.columns if 'mean' in col.lower()][:8]
    if mean_features:
        corr_matrix = data[mean_features].corr()
        
        fig = px.imshow(
            corr_matrix,
            title="Correlation Matrix (Mean Features)",
            color_continuous_scale="RdBu",
            aspect="auto"
        )
        st.plotly_chart(fig, use_container_width=True)

def show_model_comparison():
    st.header("🏆 Model Comparison")
    
    # Load results
    results = load_results_summary()
    if results is None:
        st.error("Results summary not found. Please run the Jupyter notebook first.")
        return
    
    # Classification results
    st.subheader("Classification Models Performance")
    
    classification_df = pd.DataFrame(results['classification_results'])
    
    # Performance metrics table
    display_df = fix_dataframe_types(
        classification_df[['Model', 'Test_Accuracy', 'Precision', 'Recall', 'F1_Score', 'AUC_Score']].round(4)
    )
    st.dataframe(display_df)
    
    # Performance visualization
    col1, col2 = st.columns(2)
    
    with col1:
        fig = px.bar(
            classification_df.sort_values('F1_Score', ascending=True),
            x='F1_Score',
            y='Model',
            title="F1 Score Comparison",
            orientation='h'
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        fig = px.bar(
            classification_df.sort_values('Test_Accuracy', ascending=True),
            x='Test_Accuracy',
            y='Model',
            title="Test Accuracy Comparison",
            orientation='h'
        )
        st.plotly_chart(fig, use_container_width=True)
    
    # Training time comparison
    st.subheader("Training Time Analysis")
    fig = px.scatter(
        classification_df,
        x='Training_Time',
        y='F1_Score',
        size='Test_Accuracy',
        hover_name='Model',
        title="Model Performance vs Training Time",
        labels={'Training_Time': 'Training Time (seconds)', 'F1_Score': 'F1 Score'}
    )
    st.plotly_chart(fig, use_container_width=True)
    
    # Best model highlight
    best_model = results['best_model']
    st.subheader("🥇 Best Performing Model")
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Model", best_model['Model'])
    with col2:
        st.metric("F1 Score", f"{best_model['F1_Score']:.4f}")
    with col3:
        st.metric("Accuracy", f"{best_model['Test_Accuracy']:.4f}")
    with col4:
        st.metric("AUC Score", f"{best_model['AUC_Score']:.4f}")
    
    # Clustering results
    st.subheader("Clustering Models Performance")
    
    clustering_df = pd.DataFrame(results['clustering_results'])
    display_clustering_df = fix_dataframe_types(clustering_df.round(4))
    st.dataframe(display_clustering_df)
    
    # Clustering metrics visualization
    clustering_melted = clustering_df.melt(
        id_vars=['Algorithm'],
        value_vars=['ARI', 'Silhouette', 'AMI'],
        var_name='Metric',
        value_name='Score'
    )
    
    fig = px.bar(
        clustering_melted,
        x='Algorithm',
        y='Score',
        color='Metric',
        title="Clustering Metrics Comparison",
        barmode='group'
    )
    st.plotly_chart(fig, use_container_width=True)

def show_model_prediction():
    st.header("🔮 Model Prediction")
    
    # Load preprocessing objects
    preprocessing = load_preprocessing_objects()
    if preprocessing is None:
        st.error("Preprocessing objects not found. Please run the Jupyter notebook first.")
        return
    
    # Model selection
    model_files = {
        'Random Forest': 'models/random_forest_model.pkl',
        'Gradient Boosting': 'models/gradient_boosting_model.pkl',
        'Stacking Ensemble': 'models/stacking_model.pkl',
        'L1 Regularization': 'models/lasso_model.pkl',
        'L2 Regularization': 'models/ridge_model.pkl',
        'ElasticNet': 'models/elasticnet_model.pkl'
    }
    
    selected_model_name = st.selectbox("Select a model for prediction:", list(model_files.keys()))
    
    # Load selected model
    model = load_model(model_files[selected_model_name])
    if model is None:
        st.error(f"Could not load {selected_model_name} model.")
        return
    
    st.subheader(f"Making predictions with {selected_model_name}")
    
    # Input method selection
    input_method = st.radio("Choose input method:", ["Manual Input", "Sample Data"])
    
    if input_method == "Manual Input":
        # Create input fields for features
        st.write("Enter feature values (default values are provided for testing):")
        
        # Default values based on typical breast cancer dataset values
        default_values = {
            'radius_mean': 14.127292,
            'texture_mean': 19.289649,
            'perimeter_mean': 91.969033,
            'area_mean': 654.889104,
            'smoothness_mean': 0.096360,
            'compactness_mean': 0.104341,
            'concavity_mean': 0.088799,
            'concave points_mean': 0.048919,
            'symmetry_mean': 0.181162,
            'fractal_dimension_mean': 0.062798,
            'radius_se': 0.405172,
            'texture_se': 1.216853,
            'perimeter_se': 2.866059,
            'area_se': 40.337079,
            'smoothness_se': 0.007041,
            'compactness_se': 0.025478,
            'concavity_se': 0.031894,
            'concave points_se': 0.011796,
            'symmetry_se': 0.020542,
            'fractal_dimension_se': 0.003795,
            'radius_worst': 16.269190,
            'texture_worst': 25.677223,
            'perimeter_worst': 107.261213,
            'area_worst': 880.583128,
            'smoothness_worst': 0.132369,
            'compactness_worst': 0.254265,
            'concavity_worst': 0.272188,
            'concave points_worst': 0.114606,
            'symmetry_worst': 0.290076,
            'fractal_dimension_worst': 0.083946
        }
        
        feature_names = preprocessing['feature_names']
        input_data = {}
        
        # Add preset examples
        st.write("**Quick Examples:**")
        col_ex1, col_ex2, col_ex3 = st.columns(3)
        
        with col_ex1:
            if st.button("Load Benign Example"):
                st.session_state.preset = "benign"
        with col_ex2:
            if st.button("Load Malignant Example"):
                st.session_state.preset = "malignant"
        with col_ex3:
            if st.button("Load Average Values"):
                st.session_state.preset = "average"
        
        # Set preset values based on button clicks
        if 'preset' in st.session_state:
            if st.session_state.preset == "benign":
                # Benign example values (typically lower)
                preset_values = {
                    'radius_mean': 12.0, 'texture_mean': 15.0, 'perimeter_mean': 75.0, 'area_mean': 450.0,
                    'smoothness_mean': 0.08, 'compactness_mean': 0.06, 'concavity_mean': 0.03, 'concave points_mean': 0.02,
                    'symmetry_mean': 0.16, 'fractal_dimension_mean': 0.055, 'radius_se': 0.3, 'texture_se': 1.0,
                    'perimeter_se': 2.0, 'area_se': 25.0, 'smoothness_se': 0.005, 'compactness_se': 0.015,
                    'concavity_se': 0.015, 'concave points_se': 0.008, 'symmetry_se': 0.015, 'fractal_dimension_se': 0.003,
                    'radius_worst': 13.5, 'texture_worst': 20.0, 'perimeter_worst': 85.0, 'area_worst': 550.0,
                    'smoothness_worst': 0.11, 'compactness_worst': 0.15, 'concavity_worst': 0.12, 'concave points_worst': 0.06,
                    'symmetry_worst': 0.25, 'fractal_dimension_worst': 0.07
                }
            elif st.session_state.preset == "malignant":
                # Malignant example values (typically higher)
                preset_values = {
                    'radius_mean': 18.0, 'texture_mean': 25.0, 'perimeter_mean': 120.0, 'area_mean': 1000.0,
                    'smoothness_mean': 0.12, 'compactness_mean': 0.18, 'concavity_mean': 0.20, 'concave points_mean': 0.12,
                    'symmetry_mean': 0.22, 'fractal_dimension_mean': 0.075, 'radius_se': 0.6, 'texture_se': 1.8,
                    'perimeter_se': 4.5, 'area_se': 70.0, 'smoothness_se': 0.009, 'compactness_se': 0.045,
                    'concavity_se': 0.055, 'concave points_se': 0.020, 'symmetry_se': 0.028, 'fractal_dimension_se': 0.005,
                    'radius_worst': 22.0, 'texture_worst': 35.0, 'perimeter_worst': 145.0, 'area_worst': 1500.0,
                    'smoothness_worst': 0.16, 'compactness_worst': 0.45, 'concavity_worst': 0.50, 'concave points_worst': 0.20,
                    'symmetry_worst': 0.35, 'fractal_dimension_worst': 0.12
                }
            else:  # average
                preset_values = default_values
        else:
            preset_values = default_values
        
        # Create columns for input fields
        n_cols = 3
        cols = st.columns(n_cols)
        
        for i, feature in enumerate(feature_names):
            col_idx = i % n_cols
            with cols[col_idx]:
                default_val = preset_values.get(feature, default_values.get(feature, 0.0))
                input_data[feature] = st.number_input(
                    f"{feature}",
                    value=default_val,
                    format="%.6f",
                    key=f"input_{feature}",
                    help=f"Default: {default_val:.6f}"
                )
        
        if st.button("Make Prediction", type="primary"):
            # Prepare input data
            input_df = pd.DataFrame([input_data])
            input_scaled = preprocessing['scaler'].transform(input_df)
            
            # Show input summary
            st.write("**Input Summary:**")
            summary_cols = st.columns(4)
            with summary_cols[0]:
                st.metric("Radius Mean", f"{input_data['radius_mean']:.2f}")
            with summary_cols[1]:
                st.metric("Texture Mean", f"{input_data['texture_mean']:.2f}")
            with summary_cols[2]:
                st.metric("Area Mean", f"{input_data['area_mean']:.0f}")
            with summary_cols[3]:
                st.metric("Perimeter Mean", f"{input_data['perimeter_mean']:.2f}")
            
            # Make prediction
            try:
                prediction = model.predict(input_scaled)[0]
                prediction_proba = model.predict_proba(input_scaled)[0]
                
                # Display results
                diagnosis = preprocessing['target_names'][prediction]
                confidence = max(prediction_proba) * 100
                
                st.markdown("---")
                st.subheader("🔮 Prediction Results")
                
                col1, col2 = st.columns(2)
                with col1:
                    # Color code the diagnosis
                    if diagnosis == 'M':
                        st.error(f"**Prediction: {diagnosis} (Malignant)**")
                    else:
                        st.success(f"**Prediction: {diagnosis} (Benign)**")
                with col2:
                    st.metric("Confidence", f"{confidence:.2f}%")
                
                # Probability visualization
                fig = go.Figure(data=[
                    go.Bar(
                        x=['Benign (B)', 'Malignant (M)'],
                        y=prediction_proba,
                        marker_color=['green' if i == prediction else 'lightcoral' for i in range(len(prediction_proba))],
                        text=[f'{prob:.3f}' for prob in prediction_proba],
                        textposition='auto'
                    )
                ])
                fig.update_layout(
                    title="Prediction Probabilities",
                    xaxis_title="Diagnosis",
                    yaxis_title="Probability",
                    showlegend=False
                )
                st.plotly_chart(fig, use_container_width=True)
                
            except Exception as e:
                st.error(f"Error making prediction: {str(e)}")
    
    else:  # Sample Data
        # Load sample data
        data = load_data()
        if data is not None:
            st.write("Select a sample from the dataset:")
            
            # Add filtering options
            col1, col2 = st.columns(2)
            with col1:
                diagnosis_filter = st.selectbox(
                    "Filter by diagnosis:", 
                    ["All", "Benign (B)", "Malignant (M)"]
                )
            
            # Filter data based on selection
            if diagnosis_filter == "Benign (B)":
                filtered_data = data[data['diagnosis'] == 'B']
            elif diagnosis_filter == "Malignant (M)":
                filtered_data = data[data['diagnosis'] == 'M']
            else:
                filtered_data = data
            
            with col2:
                max_samples = min(50, len(filtered_data))
                sample_idx = st.selectbox("Sample index:", range(max_samples))
            
            if len(filtered_data) > 0:
                sample_data = filtered_data.iloc[sample_idx]
                
                # Display sample info
                col1, col2 = st.columns(2)
                with col1:
                    st.write(f"**Actual Diagnosis:** {sample_data['diagnosis']}")
                    st.write(f"**Sample ID:** {sample_data['id']}")
                    
                    # Show some key features
                    st.write("**Key Features:**")
                    st.write(f"• Radius Mean: {sample_data['radius_mean']:.2f}")
                    st.write(f"• Texture Mean: {sample_data['texture_mean']:.2f}")
                    st.write(f"• Area Mean: {sample_data['area_mean']:.0f}")
                
                # Prepare features
                features = sample_data.drop(['id', 'diagnosis'])
                input_scaled = preprocessing['scaler'].transform([features])
                
                if st.button("Predict Sample", type="primary"):
                    try:
                        prediction = model.predict(input_scaled)[0]
                        prediction_proba = model.predict_proba(input_scaled)[0]
                        
                        diagnosis = preprocessing['target_names'][prediction]
                        confidence = max(prediction_proba) * 100
                        
                        # Results
                        with col2:
                            st.write(f"**Predicted Diagnosis:** {diagnosis}")
                            st.write(f"**Confidence:** {confidence:.2f}%")
                            
                            if diagnosis == sample_data['diagnosis']:
                                st.success("✅ Correct Prediction!")
                            else:
                                st.error("❌ Incorrect Prediction")
                        
                        # Probability chart
                        fig = go.Figure(data=[
                            go.Bar(
                                x=['Benign (B)', 'Malignant (M)'],
                                y=prediction_proba,
                                marker_color=['green' if i == prediction else 'lightcoral' for i in range(len(prediction_proba))],
                                text=[f'{prob:.3f}' for prob in prediction_proba],
                                textposition='auto'
                            )
                        ])
                        fig.update_layout(
                            title="Prediction Probabilities",
                            xaxis_title="Diagnosis", 
                            yaxis_title="Probability",
                            showlegend=False
                        )
                        st.plotly_chart(fig, use_container_width=True)
                        
                    except Exception as e:
                        st.error(f"Error making prediction: {str(e)}")
            else:
                st.warning("No samples available for the selected filter.")

def show_about_page():
    st.header("ℹ️ About This Project")
    
    st.markdown("""
    ## 🔬 Breast Cancer ML Model Dashboard
    
    This comprehensive machine learning project implements and evaluates multiple algorithms for breast cancer diagnosis prediction.
    
    ### 🎯 Algorithms Implemented:
    
    #### Clustering Algorithms:
    - **K-Means Clustering**: Partitioning method for grouping similar data points
    - **Hierarchical Clustering**: Tree-based clustering with agglomerative approach
    - **DBSCAN**: Density-based clustering for identifying clusters of varying shapes
    - **Gaussian Mixture Models**: Probabilistic model for clustering with overlapping clusters
    
    #### Classification Algorithms:
    - **Random Forest**: Ensemble method combining multiple decision trees
    - **Gradient Boosting**: Sequential ensemble method for improved accuracy
    - **Bagging**: Bootstrap aggregating for variance reduction
    - **AdaBoost**: Adaptive boosting focusing on misclassified samples
    - **Stacking**: Meta-ensemble combining multiple diverse base learners
    
    #### Regularization Techniques:
    - **L1 Regularization (Lasso)**: Feature selection through sparsity
    - **L2 Regularization (Ridge)**: Coefficient shrinkage for better generalization
    - **ElasticNet**: Combination of L1 and L2 regularization
    
    ### 🚀 Key Features:
    - **Comprehensive EDA**: Statistical analysis and interactive visualizations
    - **Hyperparameter Tuning**: GridSearchCV and RandomizedSearchCV optimization
    - **Performance Evaluation**: Multiple metrics including accuracy, precision, recall, F1-score, and AUC
    - **Resource Optimization**: Efficient implementation with memory monitoring
    - **Model Persistence**: CloudPickle serialization for model deployment
    - **Interactive Dashboard**: Real-time predictions and model comparison
    
    ### 📊 Evaluation Metrics:
    - **Accuracy**: Overall correctness of predictions
    - **Precision**: True positive rate (TP / (TP + FP))
    - **Recall**: Sensitivity or true positive rate (TP / (TP + FN))
    - **F1-Score**: Harmonic mean of precision and recall
    - **AUC-ROC**: Area under the receiver operating characteristic curve
    
    ### 🛠️ Technology Stack:
    - **Python**: Core programming language
    - **Scikit-learn**: Machine learning algorithms and tools
    - **Pandas & NumPy**: Data manipulation and numerical computing
    - **Matplotlib & Seaborn**: Statistical visualization
    - **Plotly**: Interactive visualizations
    - **Streamlit**: Web application framework
    - **CloudPickle**: Model serialization
    - **Jupyter Notebook**: Development and analysis environment
    
    ### 📈 Dataset:
    The Breast Cancer Wisconsin (Diagnostic) dataset contains features computed from digitized images of fine needle aspirate (FNA) of breast masses. It includes:
    - **569 samples** with **30 numerical features**
    - **Binary classification**: Malignant (M) vs Benign (B)
    - Features describe characteristics of cell nuclei present in the images
    
    ### 🎓 Use Cases:
    - Medical diagnosis assistance
    - Machine learning education and research
    - Algorithm performance comparison
    - Feature importance analysis
    - Model deployment demonstration
    
    ---
    
    **Created for comprehensive machine learning model evaluation and deployment demonstration.**
    """)

if __name__ == "__main__":
    main()
