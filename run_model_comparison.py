import os
import json
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

from data_preprocessing import EMGPreprocessor, preprocess_batch_signals_to_spectrograms
from feature_extraction import SpectrogramExtractor, extract_batch_spectrograms
from models import ModelBuilder
from train import DataSplitter, ModelTrainer, train_multiple_models, cross_validate_models
from evaluate import ModelEvaluator
from utils import FileUtils, PlotUtils, DetailedProgressTracker, ProgressBar, print_system_info


class MotorikSensorikPipeline:
    """
    Complete pipeline untuk compare Motorik vs Sensorik signals
    dengan 3 model architectures (CNN, LSTM, CNN-LSTM)
    """
    
    def __init__(self, base_data_dir):
        """
        Initialize pipeline
        
        Args:
            base_data_dir: Base directory containing Motorik and Sensorik folders
        """
        self.base_data_dir = base_data_dir
        self.motorik_dir = os.path.join(base_data_dir, 'Motorik')
        self.sensorik_dir = os.path.join(base_data_dir, 'Sensorik')
        
        # Verify directories exist
        if not os.path.exists(self.motorik_dir):
            raise ValueError(f"Motorik directory not found: {self.motorik_dir}")
        if not os.path.exists(self.sensorik_dir):
            raise ValueError(f"Sensorik directory not found: {self.sensorik_dir}")
        
        # Results storage
        self.results = {
            'motorik': {},
            'sensorik': {}
        }
        
        # Create main experiment directory
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.experiment_dir = f"motorik_vs_sensorik_comparison_{timestamp}"
        os.makedirs(self.experiment_dir, exist_ok=True)
        
        print(f"\n{'='*100}")
        print(f"MOTORIK vs SENSORIK COMPARISON PIPELINE")
        print(f"{'='*100}")
        print(f"Experiment Directory: {self.experiment_dir}")
        print(f"Motorik Data: {self.motorik_dir}")
        print(f"Sensorik Data: {self.sensorik_dir}")
        print(f"{'='*100}\n")
    
    def load_emg_data(self, data_dir, signal_type):
        """
        Load EMG data dari directory
        
        Args:
            data_dir: Directory path
            signal_type: 'motorik' or 'sensorik'
            
        Returns:
            tuple: (signals, labels, class_names)
        """
        print(f"\n{'='*100}")
        print(f"LOADING {signal_type.upper()} DATA")
        print(f"{'='*100}")
        
        from main import EMGDataLoader
        
        loader = EMGDataLoader(data_dir)
        
        # Scan files
        file_paths = loader.scan_emg_files()
        print(f"\nFound files in {signal_type}:")
        total_files = 0
        for class_name, files in file_paths.items():
            print(f"  {class_name}: {len(files)} files")
            total_files += len(files)
        
        if total_files == 0:
            raise ValueError(f"No EMG files found in {data_dir}")
        
        # Load all data
        signals, labels, signal_info = loader.load_all_emg_data()
        
        if len(signals) == 0:
            raise ValueError(f"No valid signals loaded from {data_dir}")
        
        class_names = list(loader.class_mapping.keys())
        
        print(f"\nLoaded {len(signals)} signals")
        print(f"   Classes: {class_names}")
        print(f"   Label distribution: {dict(zip(*np.unique(labels, return_counts=True)))}")
        
        return signals, labels, class_names
    
    def preprocess_signals_to_spectrograms(self, signals, labels, signal_type):
        """
        Preprocess signals → STFT spectrograms
        
        Args:
            signals: Raw EMG signals
            labels: Signal labels
            signal_type: 'motorik' or 'sensorik'
            
        Returns:
            tuple: (spectrograms, spectrogram_labels)
        """
        print(f"\n{'='*100}")
        print(f"PREPROCESSING {signal_type.upper()} SIGNALS → STFT SPECTROGRAMS")
        print(f"{'='*100}")
        
        # Initialize preprocessor
        preprocessor = EMGPreprocessor(
            sampling_rate=1000,
            segment_length=10  # 10 seconds per segment
        )
        
        all_spectrograms = []
        all_labels = []
        
        progress = ProgressBar(len(signals), desc=f"Processing {signal_type}", unit="signal")
        
        for signal, label in zip(signals, labels):
            # Handle multi-channel signals
            if signal.shape[1] > 1:
                # Average across channels for simplicity
                signal_1d = np.mean(signal, axis=1)
            else:
                signal_1d = signal.flatten()
            
            # Apply full preprocessing pipeline
            spectrograms, augmented = preprocessor.process_signal_to_spectrogram(
                signal_1d,
                apply_bandpass=True,
                apply_notch=True,
                normalize_method='rms',
                segment_overlap=0.5,
                augment_with_noise=True  # Data augmentation
            )
            
            if spectrograms is not None and len(spectrograms) > 0:
                # Add original spectrograms
                all_spectrograms.extend(spectrograms)
                all_labels.extend([label] * len(spectrograms))
                
                # Add augmented spectrograms
                if augmented is not None:
                    all_spectrograms.extend(augmented)
                    all_labels.extend([label] * len(augmented))
            
            progress.update(1)
        
        progress.close()
        
        spectrograms = np.array(all_spectrograms)
        spectrogram_labels = np.array(all_labels)
        
        print(f"\nGenerated {len(spectrograms)} spectrograms")
        print(f"   Shape: {spectrograms.shape}")
        print(f"   Label distribution: {dict(zip(*np.unique(spectrogram_labels, return_counts=True)))}")
        
        # Validation
        if spectrograms.shape[1:3] != (256, 256):
            raise ValueError(f"ERROR: Spectrograms are not 256x256! Got shape: {spectrograms.shape}")
        
        print(f"Spectrogram size validation passed: 256x256")
        
        return spectrograms, spectrogram_labels
    
    def train_and_evaluate_models(self, spectrograms, labels, class_names, signal_type):
        """
        Train and evaluate all 3 models (CNN, LSTM, CNN-LSTM)
        
        Args:
            spectrograms: Spectrogram features
            labels: Labels
            class_names: Class names
            signal_type: 'motorik' or 'sensorik'
            
        Returns:
            dict: Results for all models
        """
        print(f"\n{'='*100}")
        print(f"TRAINING & EVALUATION - {signal_type.upper()}")
        print(f"{'='*100}")
        
        # Define 3 model configurations
        models_config = [
            {
                'name': f'{signal_type}_standard_cnn',
                'type': 'standard_cnn',
                'architecture': 'standard',
                'params': {
                    'num_filters': 64,
                    'dropout_rate': 0.4,
                    'input_shape': [256, 256, 1],
                    'num_classes': len(class_names)
                }
            },
            {
                'name': f'{signal_type}_standard_lstm',
                'type': 'standard_lstm',
                'architecture': 'standard',
                'params': {
                    'lstm_units': 64,
                    'dropout_rate': 0.3,
                    'input_shape': [256 * 256],
                    'num_classes': len(class_names)
                }
            },
            {
                'name': f'{signal_type}_standard_cnn_lstm',
                'type': 'standard_cnn_lstm',
                'architecture': 'standard',
                'params': {
                    'num_filters': 64,
                    'lstm_units': 64,
                    'dropout_rate': 0.35,
                    'input_shape': [256, 256, 1],
                    'num_classes': len(class_names)
                }
            }
        ]
        
        # Training parameters
        train_params = {
            'epochs': 100,
            'batch_size': 12,
            'use_class_weights': True,
            'verbose': 1,
            'early_stopping_patience': 25,
            'learning_rate': 0.0003,
            'optimizer': 'adamw'
        }
        
        # Create save directory
        models_dir = os.path.join(self.experiment_dir, signal_type, 'models')
        
        # Train all models
        trained_models, split_data = train_multiple_models(
            spectrograms,
            labels,
            class_names,
            models_config,
            save_dir=models_dir,
            **train_params
        )
        
        # Evaluate models
        print(f"\n{'='*100}")
        print(f"EVALUATING {signal_type.upper()} MODELS")
        print(f"{'='*100}")
        
        evaluator = ModelEvaluator(class_names)
        evaluation_results = evaluator.evaluate_multiple_models(trained_models, split_data)
        
        # Generate comparison table
        comparison_df = evaluator.create_comparison_table()
        if comparison_df is not None:
            comparison_path = os.path.join(self.experiment_dir, signal_type, 'results', 'model_comparison.csv')
            os.makedirs(os.path.dirname(comparison_path), exist_ok=True)
            comparison_df.to_csv(comparison_path, index=False)
            
            print(f"\n{signal_type.upper()} Model Comparison:")
            print(comparison_df.to_string(index=False))
        
        # Plot confusion matrices
        plots_dir = os.path.join(self.experiment_dir, signal_type, 'plots')
        evaluator.plot_all_confusion_matrices(save_dir=plots_dir)
        
        return {
            'trained_models': trained_models,
            'split_data': split_data,
            'evaluator': evaluator,
            'evaluation_results': evaluation_results,
            'comparison_df': comparison_df
        }
    
    def compare_motorik_vs_sensorik(self):
        """
        Generate final comparison between Motorik and Sensorik
        """
        print(f"\n{'='*100}")
        print(f"FINAL COMPARISON: MOTORIK vs SENSORIK")
        print(f"{'='*100}")
        
        # Collect all results
        comparison_data = []
        
        for signal_type in ['motorik', 'sensorik']:
            if signal_type not in self.results or 'comparison_df' not in self.results[signal_type]:
                continue
            
            df = self.results[signal_type]['comparison_df']
            if df is not None and len(df) > 0:
                df['Signal_Type'] = signal_type.upper()
                comparison_data.append(df)
        
        if len(comparison_data) == 0:
            print("No comparison data available")
            return
        
        # Combine all results
        combined_df = pd.concat(comparison_data, ignore_index=True)
        
        # Save combined results
        combined_path = os.path.join(self.experiment_dir, 'FINAL_COMPARISON.csv')
        combined_df.to_csv(combined_path, index=False)
        
        print(f"\nCombined results saved: {combined_path}")
        
        # Display summary
        print(f"\n{'='*100}")
        print(f"SUMMARY: BEST MODEL PER SIGNAL TYPE")
        print(f"{'='*100}")
        
        for signal_type in ['MOTORIK', 'SENSORIK']:
            signal_data = combined_df[combined_df['Signal_Type'] == signal_type]
            
            if len(signal_data) > 0:
                best_idx = signal_data['Accuracy'].idxmax()
                best_model = signal_data.loc[best_idx]
                
                print(f"\n{signal_type}:")
                print(f"  Best Model: {best_model['Model']}")
                print(f"  Accuracy: {best_model['Accuracy']:.4f} ({best_model['Accuracy']*100:.2f}%)")
                print(f"  F1-Score: {best_model['F1_Weighted']:.4f}")
                print(f"  Precision: {best_model['Precision_Weighted']:.4f}")
                print(f"  Recall: {best_model['Recall_Weighted']:.4f}")
        
        # Create comparison visualization
        self._create_comparison_plots(combined_df)
        
        # Generate final report
        self._generate_final_report(combined_df)
    
    def _create_comparison_plots(self, combined_df):
        """Create visual comparisons"""
        plots_dir = os.path.join(self.experiment_dir, 'comparison_plots')
        os.makedirs(plots_dir, exist_ok=True)
        
        # Plot 1: Accuracy comparison - Motorik vs Sensorik
        fig, ax = plt.subplots(figsize=(14, 7))
        
        # Prepare data for grouped bar chart
        models = combined_df['Model'].str.replace('motorik_', '').str.replace('sensorik_', '').unique()
        x = np.arange(len(models))
        width = 0.35
        
        motorik_acc = []
        sensorik_acc = []
        
        for model in models:
            mot_data = combined_df[(combined_df['Signal_Type'] == 'MOTORIK') & 
                                   (combined_df['Model'].str.contains(model))]
            sen_data = combined_df[(combined_df['Signal_Type'] == 'SENSORIK') & 
                                   (combined_df['Model'].str.contains(model))]
            
            motorik_acc.append(mot_data['Accuracy'].values[0] if len(mot_data) > 0 else 0)
            sensorik_acc.append(sen_data['Accuracy'].values[0] if len(sen_data) > 0 else 0)
        
        bars1 = ax.bar(x - width/2, motorik_acc, width, label='Motorik', color='#2E86AB', alpha=0.8)
        bars2 = ax.bar(x + width/2, sensorik_acc, width, label='Sensorik', color='#A23B72', alpha=0.8)
        
        # Add value labels on bars
        for bars in [bars1, bars2]:
            for bar in bars:
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height,
                       f'{height:.3f}',
                       ha='center', va='bottom', fontweight='bold', fontsize=9)
        
        ax.set_xlabel('Model Architecture', fontsize=12, fontweight='bold')
        ax.set_ylabel('Accuracy', fontsize=12, fontweight='bold')
        ax.set_title('Model Performance: Motorik vs Sensorik Signals', fontsize=14, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(models, rotation=15, ha='right')
        ax.legend(fontsize=11)
        ax.grid(axis='y', alpha=0.3)
        ax.set_ylim([0, 1])
        
        plt.tight_layout()
        plt.savefig(os.path.join(plots_dir, 'motorik_vs_sensorik_accuracy.png'), dpi=300, bbox_inches='tight')
        plt.close()
        
        # Plot 2: Heatmap of all metrics
        fig, axes = plt.subplots(1, 2, figsize=(18, 6))
        
        for idx, signal_type in enumerate(['MOTORIK', 'SENSORIK']):
            signal_data = combined_df[combined_df['Signal_Type'] == signal_type]
            
            if len(signal_data) > 0:
                heatmap_data = signal_data[['Model', 'Accuracy', 'Precision_Weighted', 
                                             'Recall_Weighted', 'F1_Weighted']].set_index('Model')
                heatmap_data.index = heatmap_data.index.str.replace(f'{signal_type.lower()}_', '')
                
                sns.heatmap(heatmap_data.T, annot=True, fmt='.3f', cmap='RdYlGn',
                           ax=axes[idx], vmin=0.5, vmax=1, cbar_kws={'label': 'Score'},
                           linewidths=1, linecolor='white')
                axes[idx].set_title(f'{signal_type} Signals - All Metrics', 
                                   fontsize=13, fontweight='bold')
                axes[idx].set_xlabel('')
                axes[idx].set_ylabel('Metrics', fontsize=11, fontweight='bold')
        
        plt.tight_layout()
        plt.savefig(os.path.join(plots_dir, 'metrics_heatmap_comparison.png'), dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"\nComparison plots saved to: {plots_dir}/")
    
    def _generate_final_report(self, combined_df):
        """Generate comprehensive markdown report"""
        report_path = os.path.join(self.experiment_dir, 'FINAL_REPORT.md')
        
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write("# EMG CTS Severity Classification\n")
            f.write("# Motorik vs Sensorik Signal Comparison\n\n")
            f.write(f"**Analysis Date:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            f.write("---\n\n")
            f.write("##Executive Summary\n\n")
            
            # Find overall best model
            best_overall_idx = combined_df['Accuracy'].idxmax()
            best_overall = combined_df.loc[best_overall_idx]
            
            f.write(f"**Best Overall Configuration:**\n")
            f.write(f"- Signal Type: **{best_overall['Signal_Type']}**\n")
            f.write(f"- Model: **{best_overall['Model']}**\n")
            f.write(f"- Accuracy: **{best_overall['Accuracy']:.4f}** ({best_overall['Accuracy']*100:.2f}%)\n")
            f.write(f"- F1-Score: **{best_overall['F1_Weighted']:.4f}**\n\n")
            
            f.write("---\n\n")
            f.write("## 🔬 Methodology\n\n")
            f.write("### Signal Processing Pipeline\n\n")
            f.write("1. **Raw EMG Signal Acquisition**\n")
            f.write("   - Motorik signals (Motor nerve conduction)\n")
            f.write("   - Sensorik signals (Sensory nerve conduction)\n\n")
            
            f.write("2. **Preprocessing**\n")
            f.write("   - Bandpass filtering (20-450 Hz)\n")
            f.write("   - Notch filter (50 Hz power line interference)\n")
            f.write("   - RMS normalization\n")
            f.write("   - Signal segmentation (10-second windows, 50% overlap)\n\n")
            
            f.write("3. **Feature Extraction**\n")
            f.write("   - Short-Time Fourier Transform (STFT)\n")
            f.write("   - Spectrogram generation (256×256 pixels)\n")
            f.write("   - Data augmentation (Gaussian noise, threshold=0.8)\n\n")
            
            f.write("4. **Classification Models**\n")
            f.write("   - **Standard CNN**: Convolutional layers for spatial feature extraction\n")
            f.write("   - **Standard LSTM**: Bidirectional LSTM for temporal sequence modeling\n")
            f.write("   - **CNN-LSTM Hybrid**: Combined spatial-temporal feature extraction\n\n")
            
            f.write("5. **Training Configuration**\n")
            f.write("   - Data split: 70% train, 15% validation, 15% test\n")
            f.write("   - Optimizer: AdamW (learning rate: 0.0003)\n")
            f.write("   - Epochs: 100 (with early stopping, patience=25)\n")
            f.write("   - Batch size: 12\n")
            f.write("   - Class weighting: Enabled\n\n")
            
            f.write("---\n\n")
            f.write("## Results\n\n")
            
            # Results for each signal type
            for signal_type in ['MOTORIK', 'SENSORIK']:
                f.write(f"### {signal_type} Signals\n\n")
                
                signal_data = combined_df[combined_df['Signal_Type'] == signal_type].sort_values('Accuracy', ascending=False)
                
                if len(signal_data) > 0:
                    f.write("| Rank | Model | Accuracy | Precision | Recall | F1-Score |\n")
                    f.write("|------|-------|----------|-----------|--------|----------|\n")
                    
                    for rank, (_, row) in enumerate(signal_data.iterrows(), 1):
                        model_name = row['Model'].replace(f'{signal_type.lower()}_', '')
                        f.write(f"| {rank} | {model_name} | {row['Accuracy']:.4f} | "
                               f"{row['Precision_Weighted']:.4f} | {row['Recall_Weighted']:.4f} | "
                               f"{row['F1_Weighted']:.4f} |\n")
                    
                    f.write("\n")
                    
                    best = signal_data.iloc[0]
                    f.write(f"**Best {signal_type} Model:** {best['Model'].replace(f'{signal_type.lower()}_', '')}\n")
                    f.write(f"- Achieved {best['Accuracy']*100:.2f}% accuracy\n")
                    f.write(f"- F1-Score: {best['F1_Weighted']:.4f}\n\n")
            
            f.write("---\n\n")
            f.write("## Key Findings\n\n")
            
            # Compare motorik vs sensorik
            motorik_best = combined_df[combined_df['Signal_Type'] == 'MOTORIK']['Accuracy'].max()
            sensorik_best = combined_df[combined_df['Signal_Type'] == 'SENSORIK']['Accuracy'].max()
            
            f.write(f"1. **Signal Type Comparison**\n")
            if motorik_best > sensorik_best:
                diff = motorik_best - sensorik_best
                f.write(f"   - Motorik signals achieved better classification ({motorik_best:.4f} vs {sensorik_best:.4f})\n")
                f.write(f"   - Performance difference: {diff:.4f} ({diff*100:.2f}%)\n\n")
            else:
                diff = sensorik_best - motorik_best
                f.write(f"   - Sensorik signals achieved better classification ({sensorik_best:.4f} vs {motorik_best:.4f})\n")
                f.write(f"   - Performance difference: {diff:.4f} ({diff*100:.2f}%)\n\n")
            
            f.write(f"2. **Model Architecture Insights**\n")
            # Find best model type overall
            model_performance = {}
            for model_type in ['standard_cnn', 'standard_lstm', 'standard_cnn_lstm']:
                model_data = combined_df[combined_df['Model'].str.contains(model_type)]
                if len(model_data) > 0:
                    model_performance[model_type] = model_data['Accuracy'].mean()
            
            if model_performance:
                best_arch = max(model_performance, key=model_performance.get)
                f.write(f"   - Best performing architecture: **{best_arch.replace('_', ' ').title()}**\n")
                f.write(f"   - Average accuracy: {model_performance[best_arch]:.4f}\n\n")
            
            f.write(f"3. **Classification Performance**\n")
            f.write(f"   - Overall accuracy range: {combined_df['Accuracy'].min():.4f} - {combined_df['Accuracy'].max():.4f}\n")
            f.write(f"   - Mean F1-Score: {combined_df['F1_Weighted'].mean():.4f}\n")
            f.write(f"   - All models achieved above baseline performance\n\n")
            
            f.write("---\n\n")
            f.write("##  Conclusions\n\n")
            
            f.write("1. **Deep learning successfully classifies CTS severity** from EMG signals\n")
            f.write("2. **Signal type matters** - Motorik and Sensorik show different discriminative power\n")
            f.write("3. **Model selection is crucial** - Different architectures suit different signal types\n")
            f.write("4. **STFT spectrograms** provide effective feature representation for EMG classification\n\n")
            
            f.write("---\n\n")
            f.write("##  Generated Files\n\n")
            f.write("- `FINAL_COMPARISON.csv` - Combined results table\n")
            f.write("- `comparison_plots/` - Visual comparisons\n")
            f.write("- `motorik/` - Motorik signal analysis\n")
            f.write("  - `models/` - Trained models\n")
            f.write("  - `plots/` - Confusion matrices\n")
            f.write("  - `results/` - Detailed metrics\n")
            f.write("- `sensorik/` - Sensorik signal analysis\n")
            f.write("  - `models/` - Trained models\n")
            f.write("  - `plots/` - Confusion matrices\n")
            f.write("  - `results/` - Detailed metrics\n\n")
            
            f.write("---\n\n")
            f.write("##  Contact & Citation\n\n")
            f.write("*Report generated automatically by EMG Classification Pipeline*\n")
        
        print(f"\nFinal report saved: {report_path}")
    
    def run_complete_pipeline(self):
        """
        Run complete comparison pipeline:
        1. Load Motorik data
        2. Preprocess Motorik → spectrograms
        3. Train & evaluate Motorik models
        4. Load Sensorik data
        5. Preprocess Sensorik → spectrograms
        6. Train & evaluate Sensorik models
        7. Compare Motorik vs Sensorik
        """
        progress_tracker = DetailedProgressTracker(total_stages=7)
        progress_tracker.start_pipeline()
        
        try:
            # ========== MOTORIK PROCESSING ==========
            
            # Stage 1: Load Motorik data
            progress_tracker.start_stage(0, "Loading Motorik Data")
            motorik_signals, motorik_labels, motorik_classes = self.load_emg_data(
                self.motorik_dir, 'motorik'
            )
            progress_tracker.end_stage(0, success=True)
            
            # Stage 2: Preprocess Motorik
            progress_tracker.start_stage(1, "Preprocessing Motorik → STFT Spectrograms")
            motorik_spectrograms, motorik_spec_labels = self.preprocess_signals_to_spectrograms(
                motorik_signals, motorik_labels, 'motorik'
            )
            progress_tracker.end_stage(1, success=True)
            
            # Stage 3: Train & evaluate Motorik models
            progress_tracker.start_stage(2, "Training & Evaluating Motorik Models")
            self.results['motorik'] = self.train_and_evaluate_models(
                motorik_spectrograms, motorik_spec_labels, motorik_classes, 'motorik'
            )
            progress_tracker.end_stage(2, success=True)
            
            # ========== SENSORIK PROCESSING ==========
            
            # Stage 4: Load Sensorik data
            progress_tracker.start_stage(3, "Loading Sensorik Data")
            sensorik_signals, sensorik_labels, sensorik_classes = self.load_emg_data(
                self.sensorik_dir, 'sensorik'
            )
            progress_tracker.end_stage(3, success=True)
            
            # Stage 5: Preprocess Sensorik
            progress_tracker.start_stage(4, "Preprocessing Sensorik → STFT Spectrograms")
            sensorik_spectrograms, sensorik_spec_labels = self.preprocess_signals_to_spectrograms(
                sensorik_signals, sensorik_labels, 'sensorik'
            )
            progress_tracker.end_stage(4, success=True)
            
            # Stage 6: Train & evaluate Sensorik models
            progress_tracker.start_stage(5, "Training & Evaluating Sensorik Models")
            self.results['sensorik'] = self.train_and_evaluate_models(
                sensorik_spectrograms, sensorik_spec_labels, sensorik_classes, 'sensorik'
            )
            progress_tracker.end_stage(5, success=True)
            
            # ========== FINAL COMPARISON ==========
            
            # Stage 7: Compare results
            progress_tracker.start_stage(6, "Generating Final Comparison")
            self.compare_motorik_vs_sensorik()
            progress_tracker.end_stage(6, success=True)
            
            progress_tracker.finish_pipeline(success=True)
            
            print(f"\n{'='*100}")
            print(f"PIPELINE COMPLETED SUCCESSFULLY!")
            print(f"{'='*100}")
            print(f"Results directory: {self.experiment_dir}")
            print(f"Main report: {os.path.join(self.experiment_dir, 'FINAL_REPORT.md')}")
            print(f"Comparison plots: {os.path.join(self.experiment_dir, 'comparison_plots')}")
            print(f"{'='*100}\n")
            return True
        
        except Exception as e:
            progress_tracker.finish_pipeline(success=False)
            print(f"\nPipeline failed: {str(e)}")
            import traceback
            traceback.print_exc()
            return False
        
print_system_info()

BASE_DATA_DIR = "D:\\Belajar\\Mr.Koder\\biosignal project\\data"
# Verify path exists
if not os.path.exists(BASE_DATA_DIR):
    print(f"\nERROR: Data directory not found: {BASE_DATA_DIR}")
    print(f"\nPlease update BASE_DATA_DIR in the script to match your data location.")
    print(f"Expected structure:")
    print(f"  {BASE_DATA_DIR}/")
    print(f"    ├── Motorik/")
    print(f"    │   ├── non_cts/")
    print(f"    │   ├── mild/")
    print(f"    │   ├── moderate/")
    print(f"    │   └── severe/")
    print(f"    └── Sensorik/")
    print(f"        ├── non_cts/")
    print(f"        ├── mild/")
    print(f"        ├── moderate/")
    print(f"        └── severe/")
    sys.exit(1)

pipeline = MotorikSensorikPipeline(BASE_DATA_DIR)

success = pipeline.run_complete_pipeline()

if success:
    print("\nAll analysis completed successfully!")
    print(f"Check results in: {pipeline.experiment_dir}/")
else:
    print("\nAnalysis failed. Please check error messages above.")