import numpy as np
import pandas as pd
from sklearn.metrics import (
    confusion_matrix, classification_report, accuracy_score,
    precision_recall_fscore_support, roc_auc_score, roc_curve
)
from sklearn.preprocessing import label_binarize
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import os
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')
from tqdm import tqdm

class ModelEvaluator:
    """
    Class untuk evaluasi model performance
    """
    
    def __init__(self, class_names):
        """
        Initialize evaluator
        
        Args:
            class_names (list): List of class names
        """
        self.class_names = class_names
        self.num_classes = len(class_names)
        self.results = {}
    
    def predict_model(self, model_trainer, X_test):
        """
        Get predictions dari trained model
        
        Args:
            model_trainer: Trained model trainer object
            X_test (array): Test data
            
        Returns:
            tuple: (predictions, probabilities)
        """
        # Reshape input data sesuai model type
        X_test_reshaped = model_trainer.reshape_input_data(X_test)
        
        # Get predictions
        predictions = model_trainer.model.model.predict(X_test_reshaped)
        
        # Get class predictions
        y_pred = np.argmax(predictions, axis=1)
        y_prob = predictions
        
        return y_pred, y_prob
    
    def compute_metrics(self, y_true, y_pred, y_prob=None):
        """
        Compute comprehensive evaluation metrics
        
        Args:
            y_true (array): True labels
            y_pred (array): Predicted labels
            y_prob (array): Prediction probabilities
            
        Returns:
            dict: Dictionary of metrics
        """
        metrics = {}
        
        # Basic metrics
        metrics['accuracy'] = accuracy_score(y_true, y_pred)
        
        # Per-class metrics
        precision, recall, f1, support = precision_recall_fscore_support(
            y_true, y_pred, average=None, zero_division=0
        )
        
        metrics['precision_per_class'] = precision
        metrics['recall_per_class'] = recall
        metrics['f1_per_class'] = f1
        metrics['support_per_class'] = support
        
        # Average metrics
        metrics['precision_macro'] = np.mean(precision)
        metrics['recall_macro'] = np.mean(recall)
        metrics['f1_macro'] = np.mean(f1)
        
        # Weighted averages
        metrics['precision_weighted'] = np.average(precision, weights=support)
        metrics['recall_weighted'] = np.average(recall, weights=support)
        metrics['f1_weighted'] = np.average(f1, weights=support)
        
        # Confusion matrix
        metrics['confusion_matrix'] = confusion_matrix(y_true, y_pred)
        
        # ROC AUC (untuk multi-class)
        if y_prob is not None and self.num_classes > 2:
            try:
                # Binarize labels untuk multi-class ROC
                y_true_bin = label_binarize(y_true, classes=range(self.num_classes))
                metrics['roc_auc_ovr'] = roc_auc_score(y_true_bin, y_prob, 
                                                      multi_class='ovr', average='macro')
                metrics['roc_auc_ovo'] = roc_auc_score(y_true_bin, y_prob,
                                                      multi_class='ovo', average='macro')
            except:
                metrics['roc_auc_ovr'] = None
                metrics['roc_auc_ovo'] = None
        elif y_prob is not None and self.num_classes == 2:
            try:
                metrics['roc_auc'] = roc_auc_score(y_true, y_prob[:, 1])
            except:
                metrics['roc_auc'] = None
        
        return metrics
    
    def evaluate_model(self, model_trainer, X_test, y_test, model_name):
        """
        Evaluate single model
        
        Args:
            model_trainer: Trained model trainer
            X_test (array): Test data
            y_test (array): Test labels
            model_name (str): Name of the model
            
        Returns:
            dict: Evaluation results
        """
        print(f"\nEvaluating {model_name}...")
        
        # Get predictions
        y_pred, y_prob = self.predict_model(model_trainer, X_test)
        
        # Compute metrics
        metrics = self.compute_metrics(y_test, y_pred, y_prob)
        
        # Store results
        results = {
            'model_name': model_name,
            'y_true': y_test,
            'y_pred': y_pred,
            'y_prob': y_prob,
            'metrics': metrics
        }
        
        self.results[model_name] = results
        
        # Print summary
        print(f"   Accuracy: {metrics['accuracy']:.4f}")
        print(f"   F1-Score (macro): {metrics['f1_macro']:.4f}")
        print(f"   F1-Score (weighted): {metrics['f1_weighted']:.4f}")
        
        return results
    
    def evaluate_multiple_models(self, trained_models, split_data):
        """
        Evaluate multiple models dengan progress bar
        """
        X_test = split_data['X_test']
        y_test = split_data['y_test']
        
        print("\n" + "="*80)
        print("MODEL EVALUATION")
        print("="*80)
        
        all_results = {}
        
        model_list = [(k, v) for k, v in trained_models.items() if 'error' not in v]
        total_models = len(model_list)
        
        for idx, (model_name, model_info) in enumerate(tqdm(
            model_list,
            desc="Evaluating Models",
            unit="model",
            ncols=80
        )):
            print(f"\n[{idx+1}/{total_models}] Evaluating: {model_name}")
            
            try:
                results = self.evaluate_model(
                    model_info['trainer'], X_test, y_test, model_name
                )
                all_results[model_name] = results
                
            except Exception as e:
                print(f"Error evaluating {model_name}: {str(e)}")
                all_results[model_name] = {'error': str(e)}
        
        print(f"\n{'='*80}")
        print(f"Evaluation progress: 100.0%")
        print(f"{'='*80}")
        
        return all_results

    
    def plot_confusion_matrix(self, model_name, save_path=None, figsize=(8, 6)):
        """
        Plot confusion matrix untuk specific model
        
        Args:
            model_name (str): Name of model to plot
            save_path (str): Path to save plot
            figsize (tuple): Figure size
        """
        if model_name not in self.results:
            print(f"No results found for model: {model_name}")
            return
        
        cm = self.results[model_name]['metrics']['confusion_matrix']
        
        plt.figure(figsize=figsize)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                   xticklabels=self.class_names,
                   yticklabels=self.class_names)
        
        plt.title(f'Confusion Matrix - {model_name}')
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Confusion matrix saved to: {save_path}")
        
        plt.show()
    
    def plot_all_confusion_matrices(self, save_dir=None, figsize=(15, 10)):
        """
        Plot confusion matrices untuk all evaluated models
        
        Args:
            save_dir (str): Directory to save plots
            figsize (tuple): Figure size
        """
        num_models = len(self.results)
        if num_models == 0:
            print("No models to plot")
            return
        
        # Calculate subplot layout
        cols = min(3, num_models)
        rows = (num_models + cols - 1) // cols
        
        fig, axes = plt.subplots(rows, cols, figsize=figsize)
        if num_models == 1:
            axes = [axes]
        else:
            axes = axes.flatten() if rows > 1 else axes
        
        for idx, (model_name, results) in enumerate(self.results.items()):
            if 'error' in results:
                continue
                
            cm = results['metrics']['confusion_matrix']
            
            ax = axes[idx] if num_models > 1 else axes[0]
            
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                       xticklabels=self.class_names,
                       yticklabels=self.class_names,
                       ax=ax)
            
            ax.set_title(f'{model_name}\nAcc: {results["metrics"]["accuracy"]:.3f}')
            ax.set_xlabel('Predicted')
            ax.set_ylabel('Actual')
        
        # Hide empty subplots
        for idx in range(num_models, len(axes)):
            axes[idx].set_visible(False)
        
        plt.tight_layout()
        
        if save_dir:
            os.makedirs(save_dir, exist_ok=True)
            save_path = os.path.join(save_dir, 'all_confusion_matrices.png')
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"All confusion matrices saved to: {save_path}")
        
        plt.show()
    
    def create_comparison_table(self):
        """
        Create comparison table untuk all models
        
        Returns:
            DataFrame: Comparison table
        """
        if not self.results:
            print("No results to compare")
            return None
        
        comparison_data = []
        
        for model_name, results in self.results.items():
            if 'error' in results:
                continue
                
            metrics = results['metrics']
            
            row = {
                'Model': model_name,
                'Accuracy': metrics['accuracy'],
                'Precision (Macro)': metrics['precision_macro'],
                'Recall (Macro)': metrics['recall_macro'],
                'F1-Score (Macro)': metrics['f1_macro'],
                'Precision (Weighted)': metrics['precision_weighted'],
                'Recall (Weighted)': metrics['recall_weighted'],
                'F1-Score (Weighted)': metrics['f1_weighted']
            }
            
            # Add ROC AUC if available
            if 'roc_auc_ovr' in metrics and metrics['roc_auc_ovr'] is not None:
                row['ROC AUC (OvR)'] = metrics['roc_auc_ovr']
            elif 'roc_auc' in metrics and metrics['roc_auc'] is not None:
                row['ROC AUC'] = metrics['roc_auc']
            
            comparison_data.append(row)
        
        df = pd.DataFrame(comparison_data)
        
        # Sort by F1-Score (Weighted) descending
        if 'F1-Score (Weighted)' in df.columns:
            df = df.sort_values('F1-Score (Weighted)', ascending=False)
        
        return df
    
    def print_detailed_report(self, model_name):
        """
        Print detailed classification report untuk specific model
        
        Args:
            model_name (str): Name of model
        """
        if model_name not in self.results:
            print(f"No results found for model: {model_name}")
            return
        
        results = self.results[model_name]
        if 'error' in results:
            print(f"Model {model_name} has error: {results['error']}")
            return
        
        y_true = results['y_true']
        y_pred = results['y_pred']
        metrics = results['metrics']
        
        print(f"\n{'='*60}")
        print(f"DETAILED REPORT - {model_name}")
        print(f"{'='*60}")
        
        # Overall metrics
        print(f"Overall Accuracy: {metrics['accuracy']:.4f}")
        print(f"Macro Average F1-Score: {metrics['f1_macro']:.4f}")
        print(f"Weighted Average F1-Score: {metrics['f1_weighted']:.4f}")
        
        if 'roc_auc_ovr' in metrics and metrics['roc_auc_ovr']:
            print(f"ROC AUC (One-vs-Rest): {metrics['roc_auc_ovr']:.4f}")
        
        # Per-class metrics
        print(f"\nPer-Class Metrics:")
        print(f"{'Class':<15} {'Precision':<10} {'Recall':<10} {'F1-Score':<10} {'Support':<10}")
        print(f"{'-'*65}")
        
        for i, class_name in enumerate(self.class_names):
            precision = metrics['precision_per_class'][i]
            recall = metrics['recall_per_class'][i]
            f1 = metrics['f1_per_class'][i]
            support = metrics['support_per_class'][i]
            
            print(f"{class_name:<15} {precision:<10.4f} {recall:<10.4f} {f1:<10.4f} {support:<10}")
        
        # Sklearn classification report
        print(f"\nSklearn Classification Report:")
        report = classification_report(y_true, y_pred, target_names=self.class_names)
        print(report)
    
    def save_results(self, save_path):
        """
        Save evaluation results
        
        Args:
            save_path (str): Path to save results
        """
        # Create summary results (without large arrays)
        summary_results = {}
        
        for model_name, results in self.results.items():
            if 'error' in results:
                summary_results[model_name] = {'error': results['error']}
                continue
            
            summary_results[model_name] = {
                'model_name': results['model_name'],
                'metrics': results['metrics']
            }
        
        # Save results
        with open(save_path, 'wb') as f:
            pickle.dump({
                'class_names': self.class_names,
                'summary_results': summary_results,
                'full_results': self.results
            }, f)
        
        print(f"Evaluation results saved to: {save_path}")

def evaluate_from_saved_models(models_dir, split_data_path, results_dir='evaluation_results'):
    """
    Evaluate models dari saved model files
    
    Args:
        models_dir (str): Directory containing saved models
        split_data_path (str): Path to split data file
        results_dir (str): Directory to save results
        
    Returns:
        tuple: (evaluator, results)
    """
    # Load split data
    with open(split_data_path, 'rb') as f:
        split_data = pickle.load(f)
    
    class_names = split_data['class_names']
    
    # Initialize evaluator
    evaluator = ModelEvaluator(class_names)
    
    # Create results directory
    os.makedirs(results_dir, exist_ok=True)
    
    # Find saved models
    model_files = [f for f in os.listdir(models_dir) if f.endswith('.h5')]
    
    print(f"Found {len(model_files)} saved models")
    
    # Load and evaluate each model
    results = {}
    
    for model_file in model_files:
        model_name = model_file.replace('.h5', '')
        model_path = os.path.join(models_dir, model_file)
        metadata_path = os.path.join(models_dir, f"{model_name}_metadata.pkl")
        
        print(f"\nLoading and evaluating: {model_name}")
        
        try:
            # Load metadata
            if os.path.exists(metadata_path):
                with open(metadata_path, 'rb') as f:
                    metadata = pickle.load(f)
                
                # Create trainer dengan metadata
                from train import ModelTrainer
                trainer = ModelTrainer(
                    model_type=metadata['model_type'],
                    architecture=metadata['architecture'],
                    **metadata.get('model_params', {})
                )
                
                # Load model
                trainer.model.model = tf.keras.models.load_model(model_path)
                
                # Evaluate
                result = evaluator.evaluate_model(
                    trainer, split_data['X_test'], split_data['y_test'], model_name
                )
                
                results[model_name] = {'trainer': trainer, **result}
                
            else:
                print(f"   Warning: No metadata found for {model_name}")
                
        except Exception as e:
            print(f"   Error loading {model_name}: {str(e)}")
            results[model_name] = {'error': str(e)}
    
    # Generate comparison table
    comparison_df = evaluator.create_comparison_table()
    if comparison_df is not None:
        comparison_path = os.path.join(results_dir, 'model_comparison.csv')
        comparison_df.to_csv(comparison_path, index=False)
        print(f"\nModel comparison saved to: {comparison_path}")
        print("\nModel Comparison:")
        print(comparison_df.to_string(index=False))
    
    # Plot confusion matrices
    evaluator.plot_all_confusion_matrices(save_dir=results_dir)
    
    # Save detailed results
    results_path = os.path.join(results_dir, 'evaluation_results.pkl')
    evaluator.save_results(results_path)
    
    return evaluator, results

if __name__ == "__main__":
    print("Testing Evaluation Module...")
    
    np.random.seed(42)
    n_samples = 200
    n_classes = 4
    class_names = ['RSM', 'LSM', 'RMM', 'LMM']
    
    y_true = np.random.randint(0, n_classes, n_samples)
    y_pred = np.random.randint(0, n_classes, n_samples)
    y_prob = np.random.rand(n_samples, n_classes)
    y_prob = y_prob / y_prob.sum(axis=1, keepdims=True)  # Normalize to probabilities
    
    print(f"Generated dummy data: {n_samples} samples, {n_classes} classes")
    
    print("\n1. Testing ModelEvaluator:")
    evaluator = ModelEvaluator(class_names)
    
    metrics = evaluator.compute_metrics(y_true, y_pred, y_prob)
    print(f"   Accuracy: {metrics['accuracy']:.4f}")
    print(f"   F1-Score (macro): {metrics['f1_macro']:.4f}")
    print(f"   Confusion Matrix shape: {metrics['confusion_matrix'].shape}")
    print("    Metrics computation successful!")
    
    dummy_results = {
        'model_name': 'test_model',
        'y_true': y_true,
        'y_pred': y_pred,
        'y_prob': y_prob,
        'metrics': metrics
    }
    
    evaluator.results['test_model'] = dummy_results
    
    # Test comparison table
    print("\n2. Testing Comparison Table:")
    comparison_df = evaluator.create_comparison_table()
    if comparison_df is not None:
        print("Comparison table columns:", list(comparison_df.columns))
        print("Comparison table created successfully!")
    
    # Test detailed report
    print("\n3.Testing Detailed Report:")
    try:
        evaluator.print_detailed_report('test_model')
        print("Detailed report generated successfully!")
    except Exception as e:
        print(f"Error in detailed report: {str(e)}")
    
    print("\n All evaluation module tests completed!")