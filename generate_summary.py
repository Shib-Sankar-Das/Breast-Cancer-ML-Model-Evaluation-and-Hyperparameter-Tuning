"""
Model Performance Summary Generator
This script generates a comprehensive performance summary after model training
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import cloudpickle
import os
from pathlib import Path
import warnings

warnings.filterwarnings('ignore')

def load_results():
    """Load the results summary"""
    try:
        with open('models/results_summary.pkl', 'rb') as f:
            return cloudpickle.load(f)
    except FileNotFoundError:
        print("❌ Results summary not found. Please run the Jupyter notebook first.")
        return None

def generate_classification_summary(results):
    """Generate classification model summary"""
    classification_df = pd.DataFrame(results['classification_results'])
    
    print("🏆 CLASSIFICATION MODELS PERFORMANCE SUMMARY")
    print("=" * 60)
    
    # Sort by F1 score
    classification_df_sorted = classification_df.sort_values('F1_Score', ascending=False)
    
    print("\n📊 Performance Ranking (by F1 Score):")
    print("-" * 40)
    for i, (_, row) in enumerate(classification_df_sorted.iterrows(), 1):
        print(f"{i}. {row['Model']:<20} F1: {row['F1_Score']:.4f}")
    
    print(f"\n🥇 Best Model: {classification_df_sorted.iloc[0]['Model']}")
    best_model = classification_df_sorted.iloc[0]
    print(f"   • Accuracy: {best_model['Test_Accuracy']:.4f}")
    print(f"   • Precision: {best_model['Precision']:.4f}")
    print(f"   • Recall: {best_model['Recall']:.4f}")
    print(f"   • F1 Score: {best_model['F1_Score']:.4f}")
    print(f"   • AUC Score: {best_model['AUC_Score']:.4f}")
    print(f"   • Training Time: {best_model['Training_Time']:.4f}s")
    
    # Performance categories
    print(f"\n📈 Performance Categories:")
    print("-" * 30)
    
    high_perf = classification_df[classification_df['F1_Score'] >= 0.95]
    medium_perf = classification_df[(classification_df['F1_Score'] >= 0.90) & (classification_df['F1_Score'] < 0.95)]
    low_perf = classification_df[classification_df['F1_Score'] < 0.90]
    
    print(f"High Performance (F1 ≥ 0.95): {len(high_perf)} models")
    for _, row in high_perf.iterrows():
        print(f"   • {row['Model']}")
    
    print(f"Medium Performance (0.90 ≤ F1 < 0.95): {len(medium_perf)} models")
    for _, row in medium_perf.iterrows():
        print(f"   • {row['Model']}")
    
    print(f"Lower Performance (F1 < 0.90): {len(low_perf)} models")
    for _, row in low_perf.iterrows():
        print(f"   • {row['Model']}")
    
    # Speed analysis
    print(f"\n⚡ Training Speed Analysis:")
    print("-" * 30)
    fastest = classification_df.loc[classification_df['Training_Time'].idxmin()]
    slowest = classification_df.loc[classification_df['Training_Time'].idxmax()]
    
    print(f"Fastest: {fastest['Model']} ({fastest['Training_Time']:.4f}s)")
    print(f"Slowest: {slowest['Model']} ({slowest['Training_Time']:.4f}s)")
    
    # Efficiency score (F1 / Training Time)
    classification_df['Efficiency'] = classification_df['F1_Score'] / classification_df['Training_Time']
    most_efficient = classification_df.loc[classification_df['Efficiency'].idxmax()]
    print(f"Most Efficient: {most_efficient['Model']} (Efficiency: {most_efficient['Efficiency']:.2f})")
    
    return classification_df_sorted

def generate_clustering_summary(results):
    """Generate clustering model summary"""
    clustering_df = pd.DataFrame(results['clustering_results'])
    
    print(f"\n\n🎯 CLUSTERING MODELS PERFORMANCE SUMMARY")
    print("=" * 60)
    
    # Sort by ARI score
    clustering_df_sorted = clustering_df.sort_values('ARI', ascending=False)
    
    print("\n📊 Performance Ranking (by ARI Score):")
    print("-" * 40)
    for i, (_, row) in enumerate(clustering_df_sorted.iterrows(), 1):
        print(f"{i}. {row['Algorithm']:<20} ARI: {row['ARI']:.4f}")
    
    print(f"\n🥇 Best Clustering Algorithm: {clustering_df_sorted.iloc[0]['Algorithm']}")
    best_clustering = clustering_df_sorted.iloc[0]
    print(f"   • ARI Score: {best_clustering['ARI']:.4f}")
    print(f"   • Silhouette Score: {best_clustering['Silhouette']:.4f}")
    print(f"   • AMI Score: {best_clustering['AMI']:.4f}")
    print(f"   • Training Time: {best_clustering['Time']:.4f}s")
    
    return clustering_df_sorted

def generate_algorithm_comparison():
    """Generate algorithm type comparison"""
    print(f"\n\n🔬 ALGORITHM TYPE COMPARISON")
    print("=" * 60)
    
    results = load_results()
    if not results:
        return
    
    classification_df = pd.DataFrame(results['classification_results'])
    
    # Group by algorithm type
    ensemble_methods = ['Random Forest', 'Gradient Boosting', 'Bagging', 'AdaBoost', 'Stacking']
    regularization_methods = ['L1 Regularization', 'L2 Regularization', 'ElasticNet']
    
    ensemble_perf = classification_df[classification_df['Model'].isin(ensemble_methods)]
    regularization_perf = classification_df[classification_df['Model'].isin(regularization_methods)]
    
    print(f"\n🎲 Ensemble Methods (Average Performance):")
    print(f"   • Average F1 Score: {ensemble_perf['F1_Score'].mean():.4f}")
    print(f"   • Average Accuracy: {ensemble_perf['Test_Accuracy'].mean():.4f}")
    print(f"   • Average Training Time: {ensemble_perf['Training_Time'].mean():.4f}s")
    
    print(f"\n📏 Regularization Methods (Average Performance):")
    print(f"   • Average F1 Score: {regularization_perf['F1_Score'].mean():.4f}")
    print(f"   • Average Accuracy: {regularization_perf['Test_Accuracy'].mean():.4f}")
    print(f"   • Average Training Time: {regularization_perf['Training_Time'].mean():.4f}s")
    
    # Best from each category
    best_ensemble = ensemble_perf.loc[ensemble_perf['F1_Score'].idxmax()]
    best_regularization = regularization_perf.loc[regularization_perf['F1_Score'].idxmax()]
    
    print(f"\n🏆 Best from Each Category:")
    print(f"   • Best Ensemble: {best_ensemble['Model']} (F1: {best_ensemble['F1_Score']:.4f})")
    print(f"   • Best Regularization: {best_regularization['Model']} (F1: {best_regularization['F1_Score']:.4f})")

def generate_recommendations():
    """Generate model selection recommendations"""
    print(f"\n\n💡 MODEL SELECTION RECOMMENDATIONS")
    print("=" * 60)
    
    results = load_results()
    if not results:
        return
    
    classification_df = pd.DataFrame(results['classification_results'])
    
    print(f"\n🎯 Use Case Recommendations:")
    print("-" * 30)
    
    # Best overall
    best_overall = classification_df.loc[classification_df['F1_Score'].idxmax()]
    print(f"✨ Best Overall Performance: {best_overall['Model']}")
    print(f"   Use when: Accuracy is the top priority")
    
    # Fastest
    fastest = classification_df.loc[classification_df['Training_Time'].idxmin()]
    print(f"\n⚡ Fastest Training: {fastest['Model']}")
    print(f"   Use when: Quick deployment is needed")
    
    # Most balanced
    classification_df['Balance_Score'] = (
        classification_df['F1_Score'] * 0.6 + 
        (1 / classification_df['Training_Time']) * 0.4
    )
    most_balanced = classification_df.loc[classification_df['Balance_Score'].idxmax()]
    print(f"\n⚖️ Best Balance (Performance/Speed): {most_balanced['Model']}")
    print(f"   Use when: Need good performance with reasonable training time")
    
    # High precision
    high_precision = classification_df.loc[classification_df['Precision'].idxmax()]
    print(f"\n🎯 Highest Precision: {high_precision['Model']}")
    print(f"   Use when: False positives must be minimized")
    
    # High recall
    high_recall = classification_df.loc[classification_df['Recall'].idxmax()]
    print(f"\n🔍 Highest Recall: {high_recall['Model']}")
    print(f"   Use when: False negatives must be minimized (medical diagnosis)")

def save_summary_report(classification_df, clustering_df):
    """Save a comprehensive summary report"""
    report_file = "model_performance_report.txt"
    
    with open(report_file, 'w') as f:
        f.write("BREAST CANCER ML MODEL PERFORMANCE REPORT\n")
        f.write("=" * 50 + "\n\n")
        
        f.write("CLASSIFICATION MODELS\n")
        f.write("-" * 20 + "\n")
        f.write(classification_df.to_string(index=False))
        
        f.write("\n\nCLUSTERING MODELS\n")
        f.write("-" * 20 + "\n")
        f.write(clustering_df.to_string(index=False))
        
        f.write(f"\n\nBEST CLASSIFICATION MODEL\n")
        f.write("-" * 25 + "\n")
        best_model = classification_df.iloc[0]
        for col in best_model.index:
            f.write(f"{col}: {best_model[col]}\n")
    
    print(f"\n📄 Detailed report saved to: {report_file}")

def create_visualization():
    """Create performance visualization"""
    results = load_results()
    if not results:
        return
    
    classification_df = pd.DataFrame(results['classification_results'])
    
    # Create performance comparison plot
    plt.figure(figsize=(15, 10))
    
    # Subplot 1: F1 Score comparison
    plt.subplot(2, 2, 1)
    sorted_df = classification_df.sort_values('F1_Score')
    plt.barh(range(len(sorted_df)), sorted_df['F1_Score'])
    plt.yticks(range(len(sorted_df)), sorted_df['Model'].tolist())
    plt.xlabel('F1 Score')
    plt.title('F1 Score Comparison')
    plt.grid(axis='x', alpha=0.3)
    
    # Subplot 2: Training time comparison
    plt.subplot(2, 2, 2)
    sorted_time = classification_df.sort_values('Training_Time')
    plt.barh(range(len(sorted_time)), sorted_time['Training_Time'])
    plt.yticks(range(len(sorted_time)), sorted_time['Model'].tolist())
    plt.xlabel('Training Time (seconds)')
    plt.title('Training Time Comparison')
    plt.grid(axis='x', alpha=0.3)
    
    # Subplot 3: Performance vs Speed scatter
    plt.subplot(2, 2, 3)
    plt.scatter(classification_df['Training_Time'], classification_df['F1_Score'])
    for i, model in enumerate(classification_df['Model']):
        plt.annotate(model, (classification_df['Training_Time'].iloc[i], classification_df['F1_Score'].iloc[i]),
                    fontsize=8, rotation=45)
    plt.xlabel('Training Time (seconds)')
    plt.ylabel('F1 Score')
    plt.title('Performance vs Training Speed')
    plt.grid(alpha=0.3)
    
    # Subplot 4: Metrics comparison for top 5 models
    plt.subplot(2, 2, 4)
    top_5 = classification_df.nlargest(5, 'F1_Score')
    metrics = ['Test_Accuracy', 'Precision', 'Recall', 'F1_Score']
    x = np.arange(len(top_5))
    width = 0.2
    
    for i, metric in enumerate(metrics):
        plt.bar(x + i*width, top_5[metric], width, label=metric.replace('_', ' '))
    
    plt.xlabel('Models')
    plt.ylabel('Score')
    plt.title('Top 5 Models - All Metrics')
    plt.xticks(x + width*1.5, top_5['Model'].tolist(), rotation=45)
    plt.legend()
    plt.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('model_performance_comparison.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print(f"\n📊 Performance visualization saved to: model_performance_comparison.png")

def main():
    """Main function to generate comprehensive summary"""
    print("🔬 Breast Cancer ML Project - Performance Summary Generator")
    print("=" * 70)
    
    # Load results
    results = load_results()
    if not results:
        return
    
    # Generate summaries
    classification_df = generate_classification_summary(results)
    clustering_df = generate_clustering_summary(results)
    
    # Generate comparisons and recommendations
    generate_algorithm_comparison()
    generate_recommendations()
    
    # Save report
    save_summary_report(classification_df, clustering_df)
    
    # Create visualization
    create_visualization()
    
    print(f"\n🎉 Summary generation completed!")
    print(f"Check the generated files:")
    print(f"   • model_performance_report.txt")
    print(f"   • model_performance_comparison.png")

if __name__ == "__main__":
    main()
