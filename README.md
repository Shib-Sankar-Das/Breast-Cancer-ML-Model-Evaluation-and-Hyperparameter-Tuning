# Breast Cancer ML Model Evaluation and Hyperparameter Tuning

This project implements a comprehensive machine learning pipeline for breast cancer diagnosis prediction, featuring multiple algorithms, hyperparameter tuning, and an interactive Streamlit dashboard.

## 🎯 Project Overview

This project implements 13 different machine learning algorithms with comprehensive evaluation and comparison:

### Clustering Algorithms
- K-Means Clustering
- Hierarchical Clustering (Agglomerative)
- DBSCAN
- Gaussian Mixture Models

### Classification Algorithms
- Random Forest (Ensemble)
- Gradient Boosting (Ensemble)
- Bagging (Ensemble)
- AdaBoost (Boosting)
- Stacking (Meta-ensemble)
- L1 Regularization (Lasso)
- L2 Regularization (Ridge)
- ElasticNet Regularization

## ✅ Current Status

**🎉 PROJECT FULLY OPERATIONAL!**

All compatibility issues have been resolved and the project is now running successfully:
- ✅ All packages installed and compatible with Python 3.13
- ✅ Streamlit app running on http://localhost:8503
- ✅ PyArrow compatibility fixes working properly
- ✅ Enhanced dashboard with default values and preset examples

**Ready to use:**
1. **Streamlit Dashboard**: Currently running at http://localhost:8503
2. **Model Training**: Run `model_evaluation_and_tuning.ipynb` to train all 13 ML models
3. **Interactive Predictions**: Use the dashboard with preset examples or custom values

## 🚀 Features

- **Comprehensive EDA**: Statistical analysis and interactive visualizations
- **Hyperparameter Tuning**: GridSearchCV and RandomizedSearchCV optimization
- **Performance Evaluation**: Multiple metrics (accuracy, precision, recall, F1-score, AUC)
- **Resource Optimization**: Efficient implementation with low CPU and RAM usage
- **Model Persistence**: CloudPickle serialization for deployment
- **Interactive Dashboard**: Real-time predictions and model comparison
- **Memory Monitoring**: Track resource usage during training

## 📁 Project Structure

```
├── Data/
│   └── breast-cancer.csv              # Dataset
├── models/                            # Generated model files (after running notebook)
│   ├── kmeans_model.pkl
│   ├── random_forest_model.pkl
│   ├── gradient_boosting_model.pkl
│   ├── stacking_model.pkl
│   ├── preprocessing_objects.pkl
│   └── results_summary.pkl
├── model_evaluation_and_tuning.ipynb  # Main Jupyter notebook
├── streamlit_app.py                   # Streamlit web application  
├── generate_summary.py                # Performance analysis tool
├── requirements.txt                   # Python dependencies
└── README.md                          # This file
```

## 🛠️ Installation

1. **Clone or download this project to your local machine**

2. **Install all dependencies:**
   ```bash
   pip install -r requirements.txt
   ```
   
   **Note**: The Streamlit app includes automatic PyArrow compatibility fixes.

3. **Verify the dataset:**
   - Ensure `Data/breast-cancer.csv` exists in the project directory
   - The dataset should contain 570 rows with breast cancer diagnostic data

## 📊 Usage

### Step 1: Run the Jupyter Notebook

1. **Start Jupyter Notebook:**
   ```bash
   jupyter notebook
   ```

2. **Open and run `model_evaluation_and_tuning.ipynb`:**
   - Execute all cells sequentially
   - This will train all models and save them to the `models/` directory
   - The notebook includes comprehensive EDA, model training, and evaluation
   - **Note**: This may take 10-15 minutes depending on your system

### Step 2: Launch the Streamlit Dashboard

1. **Start the Streamlit application:**
   ```bash
   streamlit run streamlit_app.py
   ```

2. **Access the dashboard:**
   - Open your browser to `http://localhost:8501`
   - Navigate through different pages using the sidebar

### Optional: Generate Performance Summary

After training models, you can generate a detailed performance analysis:

```bash
python generate_summary.py
```

This creates:
- `model_performance_report.txt` - Detailed metrics comparison
- `model_performance_comparison.png` - Performance visualizations

## 🖥️ Dashboard Pages

### 1. Home Page
- Project overview and dataset statistics
- Quick visualization of diagnosis distribution

### 2. Data Exploration
- Dataset overview and statistical summary
- Interactive feature distribution plots
- Correlation analysis with heatmaps

### 3. Model Comparison
- Performance metrics comparison table
- Interactive visualizations of model performance
- Training time vs accuracy analysis
- Best model identification

### 4. Model Prediction
- **Manual Input**: Enter feature values manually for prediction
- **Sample Data**: Select samples from the dataset for testing
- Real-time predictions with confidence scores
- Probability visualization

### 5. About
- Comprehensive project documentation
- Algorithm descriptions and technical details

## 📈 Model Evaluation Metrics

The project evaluates models using multiple metrics:

- **Accuracy**: Overall correctness of predictions
- **Precision**: True positive rate (TP / (TP + FP))
- **Recall**: Sensitivity (TP / (TP + FN))
- **F1-Score**: Harmonic mean of precision and recall
- **AUC-ROC**: Area under the receiver operating characteristic curve
- **Training Time**: Computational efficiency
- **Memory Usage**: Resource consumption monitoring

For clustering algorithms:
- **Adjusted Rand Index (ARI)**: Similarity to ground truth
- **Silhouette Score**: Cluster quality measure
- **Adjusted Mutual Information (AMI)**: Information-theoretic measure

## 🎛️ Hyperparameter Tuning

The project implements both GridSearchCV and RandomizedSearchCV for optimization:

### GridSearchCV
- Used for regularization methods (L1, L2, ElasticNet)
- Exhaustive search over parameter grid
- Ensures optimal parameter selection

### RandomizedSearchCV
- Used for ensemble methods (Random Forest, Gradient Boosting, etc.)
- Efficient sampling of parameter space
- Balances performance and computational cost

## 🧠 Machine Learning Pipeline

1. **Data Loading and Preprocessing**
   - Load breast cancer dataset
   - Handle missing values
   - Feature scaling with StandardScaler
   - Label encoding for target variable

2. **Exploratory Data Analysis**
   - Statistical summaries
   - Feature distributions by diagnosis
   - Correlation analysis
   - Feature importance analysis

3. **Model Training and Evaluation**
   - Train 13 different algorithms
   - Hyperparameter tuning for each model
   - Cross-validation for robust evaluation
   - Performance metric calculation

4. **Model Comparison and Selection**
   - Compare all models using multiple metrics
   - Identify best-performing model
   - Analyze trade-offs (accuracy vs speed)

5. **Model Persistence**
   - Save all trained models using CloudPickle
   - Store preprocessing objects
   - Save evaluation results

## 🔧 Performance Optimization

The project is optimized for efficiency:

- **Memory Monitoring**: Track RAM usage during training
- **Parallel Processing**: Use `n_jobs=-1` for multi-core utilization
- **Efficient Algorithms**: Optimized parameter ranges for faster tuning
- **Resource-Aware Implementation**: Monitor and minimize resource consumption

## 📋 Requirements

### Software Requirements
- Python 3.8 or higher
- Jupyter Notebook
- Web browser (for Streamlit dashboard)

### Hardware Recommendations
- **RAM**: Minimum 4GB, recommended 8GB+
- **CPU**: Multi-core processor recommended
- **Storage**: ~500MB for models and data

## 🚨 Troubleshooting

### Common Issues

1. **"Dataset not found" error:**
   - Ensure `Data/breast-cancer.csv` exists
   - Check file path and permissions

2. **"Model file not found" error:**
   - Run the Jupyter notebook first to generate models
   - Ensure the `models/` directory contains .pkl files

3. **Memory errors during training:**
   - Reduce parameter grid sizes in hyperparameter tuning
   - Close other applications to free RAM

4. **Slow model training:**
   - Reduce `n_iter` in RandomizedSearchCV
   - Use smaller parameter grids
   - Ensure `n_jobs=-1` is set for parallel processing

5. **Streamlit compatibility issues:**
   - The app includes automatic PyArrow compatibility fixes
   - If issues persist, restart the Streamlit app
   - Clear Streamlit cache: Delete `.streamlit` folder in your home directory

### Performance Tips

- **For faster training**: Reduce cross-validation folds (cv=3 instead of cv=5)
- **For lower memory usage**: Train models individually rather than all at once  
- **For quicker results**: Use RandomizedSearchCV with fewer iterations

## 🚀 Streamlit Cloud Deployment

### Deployment Issues Fixed

The initial deployment failures were due to:
1. **PyArrow build failure** - Missing cmake dependency
2. **Pandas compilation errors** - Python 3.13 compatibility issues

### Solution Applied

**Updated requirements.txt** with compatible versions:
```
streamlit==1.29.0
pandas==2.1.4
numpy==1.24.4
scikit-learn==1.3.2
plotly==5.17.0
cloudpickle==3.0.0
joblib==1.3.2
```

### Alternative Deployment Options

If you still encounter issues, try these alternatives:

1. **Minimal Requirements** (use `requirements_minimal.txt`):
```
streamlit
pandas
numpy
scikit-learn
plotly
cloudpickle
```

2. **Backup Requirements** (use `requirements_backup.txt`):
```
streamlit>=1.28.0
pandas>=2.0.0
numpy>=1.21.0
scikit-learn>=1.2.0
plotly>=5.0.0
cloudpickle>=2.0.0
setuptools>=60.0.0
wheel>=0.37.0
```

### Deployment Steps

1. **Push to GitHub** with the updated `requirements.txt`
2. **Connect to Streamlit Cloud** at https://share.streamlit.io/
3. **Select your repository** and branch
4. **Set main file** as `streamlit_app.py`
5. **Deploy** - should work without compilation errors

### Local Testing

Before deploying, test locally:
```bash
python test_imports.py  # Test all imports
streamlit run streamlit_app.py  # Test locally
```

## 🔧 Troubleshooting

### Common Issues and Solutions

1. **Import Errors**: Use the test script to verify all packages work
2. **Model Loading Errors**: Ensure all `.pkl` files are in the `models/` directory
3. **Data Loading Errors**: Check that `Data/breast-cancer.csv` exists
4. **Memory Issues**: Use the minimal requirements for resource-constrained deployments

### Configuration

The project includes a `.streamlit/config.toml` file for optimal deployment:
- Legacy DataFrame serialization for compatibility
- Optimized server settings for cloud deployment
- Custom theme matching the dashboard design

## 📄 License

This project is for educational and research purposes. The breast cancer dataset is publicly available and widely used in machine learning research.

---

**Happy Machine Learning! 🚀**
