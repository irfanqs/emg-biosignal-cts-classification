import os
import numpy as np
import pandas as pd
from datetime import datetime
import json
import warnings
warnings.filterwarnings('ignore')

from data_preprocessing import EMGPreprocessor, preprocess_batch_signals_to_spectrograms
from feature_extraction import SpectrogramExtractor, extract_batch_spectrograms
from channel_selection_enhanced import perform_complete_channel_analysis
from models import ModelBuilder
from train import DataSplitter, ModelTrainer, train_multiple_models, cross_validate_models
from evaluate import ModelEvaluator
from utils import FileUtils, PlotUtils, LogUtils, print_system_info, DetailedProgressTracker


class EMGDataLoader:
    """Load EMG data dari struktur folder"""
    
    def __init__(self, data_directory):
        self.data_directory = data_directory
        self.class_mapping = {
            'non_cts': 0,
            'mild': 1, 
            'moderate': 2,
            'severe': 3
        }
    
    def scan_emg_files(self):
        """Scan semua EMG files"""
        file_paths = {}
        
        for class_name in self.class_mapping.keys():
            class_dir = os.path.join(self.data_directory, class_name)
            if os.path.exists(class_dir):
                files = [f for f in os.listdir(class_dir) 
                        if f.endswith(('.txt', '.csv'))]
                file_paths[class_name] = [os.path.join(class_dir, f) for f in files]
            else:
                file_paths[class_name] = []
        
        return file_paths
    
    def load_single_file(self, filepath):
        """Load single EMG file"""
        try:
            data = np.loadtxt(filepath)
            if len(data.shape) == 1:
                data = data.reshape(-1, 1)
            return data
        except Exception as e:
            print(f"Error loading {filepath}: {e}")
            return None
    
    def load_all_emg_data(self, max_files_per_class=None):
        """Load semua EMG data"""
        file_paths = self.scan_emg_files()
        
        all_signals = []
        all_labels = []
        signal_info = []
        
        for class_name, files in file_paths.items():
            if max_files_per_class:
                files = files[:max_files_per_class]
            
            class_label = self.class_mapping[class_name]
            
            for filepath in files:
                signal = self.load_single_file(filepath)
                if signal is not None:
                    if signal.shape[1] > 1:
                        all_signals.append(signal)
                    else:
                        all_signals.append(signal)
                    
                    all_labels.append(class_label)
                    signal_info.append({
                        'filepath': filepath,
                        'class': class_name,
                        'label': class_label,
                        'shape': signal.shape
                    })
        
        return all_signals, np.array(all_labels), signal_info


class EMGPipeline:
    """Pipeline FIXED - CNN vs LSTM vs CNN-LSTM"""
    
    def __init__(self, config):
        self.config = config
        self.logger = None
        self.experiment_dir = None
        
        self.raw_signals = None
        self.labels = None
        self.class_names = None
        
        self.channel_analysis_results = None
        self.selected_channels = None
        
        self.spectrograms = None
        self.spectrogram_labels = None
        
        self.cv_results = None
        self.cv_validator = None
        self.models = None
        self.split_data = None
        self.evaluator = None
    
    def setup_experiment(self):
        """Setup experiment directory"""
        exp_name = self.config.get('experiment_name', 
                                   f"emg_cnn_lstm_comparison_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
        self.experiment_dir = FileUtils.create_experiment_folder(
            self.config.get('experiments_dir', 'experiments'), exp_name
        )
        
        log_file = os.path.join(self.experiment_dir, 'logs', 'experiment.log')
        self.logger = LogUtils.setup_logging(log_file, self.config.get('log_level', 'INFO'))
        
        LogUtils.log_experiment_info(self.logger, self.config)
        print_system_info()
        
        config_path = os.path.join(self.experiment_dir, 'config.json')
        FileUtils.save_json(self.config, config_path)
        
        return self.experiment_dir
    
    def load_data(self):
        """Load EMG data"""
        print("\n" + "="*80)
        print("STEP 1: LOADING EMG DATA")
        print("="*80)
        
        data_loader = EMGDataLoader(self.config['data_directory'])
        file_paths = data_loader.scan_emg_files()
        
        print("\nFound files:")
        for class_name, files in file_paths.items():
            print(f"  {class_name}: {len(files)} files")
        
        if sum(len(files) for files in file_paths.values()) == 0:
            raise ValueError("No EMG files found!")
        
        self.raw_signals, self.labels, signal_info = data_loader.load_all_emg_data(
            max_files_per_class=self.config.get('max_files_per_class', None)
        )
        
        if len(self.raw_signals) == 0:
            raise ValueError("No valid EMG signals loaded!")
        
        self.class_names = list(data_loader.class_mapping.keys())
        
        print(f"\nLoaded {len(self.raw_signals)} signals from {len(self.class_names)} classes")
        print(f"  Classes: {self.class_names}")
        
        if self.raw_signals[0].shape[1] > 1:
            print(f"  Detected multi-channel data: {self.raw_signals[0].shape[1]} channels per signal")
        
        return self.raw_signals, self.labels, self.class_names
    
    def perform_channel_selection(self):
        """Statistical channel selection (jika data multi-channel)"""
        print("\n" + "="*80)
        print("STEP 2: CHANNEL ANALYSIS & SELECTION")
        print("="*80)
        
        if self.raw_signals[0].shape[1] <= 1:
            print("\nSingle channel data detected - skipping channel selection")
            self.selected_channels = [0]
            return [0]
        
        print(f"\nAnalyzing {self.raw_signals[0].shape[1]} channels...")
        
        analysis_dir = os.path.join(self.experiment_dir, 'channel_analysis')
        
        self.channel_analysis_results = perform_complete_channel_analysis(
            self.raw_signals,
            self.labels,
            self.class_names,
            sampling_rate=self.config.get('sampling_rate', 1000),
            top_channels=min(10, self.raw_signals[0].shape[1]),
            save_dir=analysis_dir
        )
        
        self.selected_channels = self.channel_analysis_results['selected_channels']
        
        print(f"\nSelected channels: {self.selected_channels}")
        
        return self.selected_channels
    
    def extract_features(self):
        """Extract spectrogram features"""
        print("\n" + "="*80)
        print("STEP 3: FEATURE EXTRACTION (STFT Spectrograms)")
        print("="*80)
        
        preprocessor = EMGPreprocessor(
            sampling_rate=self.config.get('sampling_rate', 1000),
            segment_length=self.config.get('segment_length', 10)
        )
        
        preprocessing_params = self.config.get('preprocessing', {})
        
        print("\nProcessing signals to spectrograms...")
        
        all_spectrograms = []
        all_labels = []
        
        from utils import ProgressBar
        progress = ProgressBar(len(self.raw_signals), desc="Processing", unit="signal")
        
        for signal_idx, (signal, label) in enumerate(zip(self.raw_signals, self.labels)):
            if signal.shape[1] > 1 and self.selected_channels:
                signal_selected = signal[:, self.selected_channels]
                signal_1d = np.mean(signal_selected, axis=1)
            else:
                signal_1d = signal.flatten()
            
            spectrograms, augmented = preprocessor.process_signal_to_spectrogram(
                signal_1d,
                **preprocessing_params
            )
            
            if spectrograms is not None:
                all_spectrograms.extend(spectrograms)
                all_labels.extend([label] * len(spectrograms))
                
                if augmented is not None and preprocessing_params.get('augment_data', True):
                    all_spectrograms.extend(augmented)
                    all_labels.extend([label] * len(augmented))
            
            progress.update(1)
        
        progress.close()
        
        self.spectrograms = np.array(all_spectrograms)
        self.spectrogram_labels = np.array(all_labels)
        
        print(f"\n Generated {len(self.spectrograms)} spectrograms")
        print(f"  Spectrogram shape: {self.spectrograms.shape}")
        
        return self.spectrograms, self.spectrogram_labels
    
    def run_cross_validation(self):
        """5-Fold cross validation"""
        cv_config = self.config.get('cross_validation', {})
        
        if not cv_config.get('enabled', False):
            print("\nCross validation disabled")
            return None
        
        print("\n" + "="*80)
        print("STEP 4: 5-FOLD CROSS VALIDATION")
        print("="*80)
        
        models_config = self.config.get('models', [])
        cv_epochs = cv_config.get('cv_epochs', 50)
        cv_batch_size = cv_config.get('cv_batch_size', 12)
        cv_verbose = cv_config.get('verbose', 0)
        
        cv_results_dir = os.path.join(self.experiment_dir, 'cv_results')
        
        print(f"\nModels to evaluate:")
        for model_cfg in models_config:
            print(f"  - {model_cfg['name']} ({model_cfg['type']})")
        
        self.cv_results, self.cv_validator = cross_validate_models(
            self.spectrograms,
            self.spectrogram_labels,
            self.class_names,
            models_config,
            cv_epochs=cv_epochs,
            batch_size=cv_batch_size,
            verbose=cv_verbose,
            save_dir=cv_results_dir
        )
        
        return self.cv_results
    
    def train_models(self):
        """Train 3 model architectures"""
        print("\n" + "="*80)
        print("STEP 5: TRAINING MODELS")
        print("="*80)
        
        models_config = self.config.get('models', [])
        train_params = self.config.get('training', {})
        
        models_dir = os.path.join(self.experiment_dir, 'models')
        self.models, self.split_data = train_multiple_models(
            self.spectrograms,
            self.spectrogram_labels,
            self.class_names,
            models_config,
            save_dir=models_dir,
            **train_params
        )
        
        return self.models, self.split_data
    
    def evaluate_models(self):
        """Evaluate trained models"""
        print("\n" + "="*80)
        print("STEP 6: MODEL EVALUATION")
        print("="*80)
        
        self.evaluator = ModelEvaluator(self.class_names)
        
        results = self.evaluator.evaluate_multiple_models(self.models, self.split_data)
        
        comparison_df = self.evaluator.create_comparison_table()
        if comparison_df is not None:
            comparison_path = os.path.join(self.experiment_dir, 'results', 'model_comparison.csv')
            comparison_df.to_csv(comparison_path, index=False)
            
            print("\n" + "="*80)
            print("MODEL COMPARISON RESULTS")
            print("="*80)
            print(comparison_df.to_string(index=False))
            print("="*80)
        
        self.evaluator.plot_all_confusion_matrices(
            save_dir=os.path.join(self.experiment_dir, 'plots')
        )
        
        return results
    
    def generate_final_report(self):
        """Generate comprehensive report"""
        report_path = os.path.join(self.experiment_dir, 'FINAL_REPORT.md')
        
        with open(report_path, 'w') as f:
            f.write("# EMG Classification - Final Report\n\n")
            f.write(f"**Date:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            f.write("## 1. Experiment Overview\n\n")
            f.write("**Objective:** Compare CNN vs LSTM vs CNN-LSTM for EMG-based CTS severity classification\n\n")
            
            f.write("## 2. Dataset Information\n\n")
            f.write(f"- Total signals: {len(self.raw_signals)}\n")
            f.write(f"- Classes: {self.class_names}\n")
            f.write(f"- Class distribution:\n")
            for class_name, label in zip(self.class_names, range(len(self.class_names))):
                count = np.sum(self.labels == label)
                f.write(f"  - {class_name}: {count} samples\n")
            f.write(f"\n- Total spectrograms: {len(self.spectrograms)}\n")
            f.write(f"- Spectrogram shape: {self.spectrograms.shape}\n\n")
            
            if self.selected_channels:
                f.write("## 3. Channel Selection Results\n\n")
                f.write(f"Selected channels: {self.selected_channels}\n\n")
            
            f.write("## 4. Model Architectures\n\n")
            f.write("| Model | Type | Description |\n")
            f.write("|-------|------|-------------|\n")
            f.write("| Standard CNN | CNN | Convolutional layers for spatial feature extraction |\n")
            f.write("| Standard LSTM | LSTM | Bidirectional LSTM for temporal sequence modeling |\n")
            f.write("| CNN-LSTM | Hybrid | CNN feature extraction + LSTM temporal modeling |\n\n")
            
            f.write("## 5. Performance Results\n\n")
            
            if self.evaluator and self.evaluator.results:
                f.write("### Test Set Performance\n\n")
                f.write("| Rank | Model | Accuracy | Precision | Recall | F1-Score |\n")
                f.write("|------|-------|----------|-----------|--------|----------|\n")
                
                results_list = []
                for model_name, result in self.evaluator.results.items():
                    if 'metrics' in result:
                        metrics = result['metrics']
                        results_list.append({
                            'name': model_name,
                            'accuracy': metrics['accuracy'],
                            'precision': metrics['precision_weighted'],
                            'recall': metrics['recall_weighted'],
                            'f1': metrics['f1_weighted']
                        })
                
                results_list.sort(key=lambda x: x['accuracy'], reverse=True)
                
                for rank, res in enumerate(results_list, 1):
                    f.write(f"| {rank} | {res['name']} | {res['accuracy']:.4f} | "
                           f"{res['precision']:.4f} | {res['recall']:.4f} | {res['f1']:.4f} |\n")
                
                if results_list:
                    best = results_list[0]
                    f.write(f"\n**Best Model:** {best['name']}\n")
                    f.write(f"- Test Accuracy: {best['accuracy']:.4f} ({best['accuracy']*100:.2f}%)\n")
                    f.write(f"- F1-Score: {best['f1']:.4f}\n\n")
            
            if self.cv_validator and hasattr(self.cv_validator, 'all_cv_results'):
                f.write("### Cross-Validation Results (5-Fold)\n\n")
                f.write("| Model | Mean Accuracy | Std Dev |\n")
                f.write("|-------|---------------|----------|\n")
                
                cv_results_list = []
                for model_name, results in self.cv_validator.all_cv_results.items():
                    if 'error' not in results:
                        cv_results_list.append({
                            'name': model_name,
                            'mean': results['mean_accuracy'],
                            'std': results['std_accuracy']
                        })
                
                cv_results_list.sort(key=lambda x: x['mean'], reverse=True)
                
                for res in cv_results_list:
                    f.write(f"| {res['name']} | {res['mean']:.4f} | {res['std']:.4f} |\n")
            
            f.write("\n## 6. Conclusion\n\n")
            f.write("Based on the experimental results:\n\n")
            
            if self.evaluator and self.evaluator.results and results_list:
                best_model = results_list[0]
                f.write(f"- **Best performing architecture:** {best_model['name']}\n")
                f.write(f"- **Achieved accuracy:** {best_model['accuracy']*100:.2f}%\n")
                f.write(f"- The comparison demonstrates the effectiveness of deep learning approaches for EMG-based CTS severity classification.\n\n")
            
            f.write("## 7. Files Generated\n\n")
            f.write("- Configuration: `config.json`\n")
            f.write("- Channel analysis: `channel_analysis/`\n")
            f.write("- Cross-validation: `cv_results/`\n")
            f.write("- Trained models: `models/`\n")
            f.write("- Evaluation results: `results/`\n")
            f.write("- Visualizations: `plots/`\n")
            f.write("- Logs: `logs/`\n")
        
        print(f"\n Final report saved: {report_path}")
    
    def run_complete_pipeline(self):
        """Run complete pipeline"""
        progress = DetailedProgressTracker(total_stages=6)
        progress.start_pipeline()
        
        try:
            self.setup_experiment()
            
            # Stage 1: Load data
            progress.start_stage(0, "Loading EMG Data")
            self.load_data()
            progress.end_stage(0, success=True)
            
            # Stage 2: Channel selection (if multi-channel)
            progress.start_stage(1, "Channel Selection")
            self.perform_channel_selection()
            progress.end_stage(1, success=True)
            
            # Stage 3: Feature extraction
            progress.start_stage(2, "Feature Extraction")
            self.extract_features()
            progress.end_stage(2, success=True)
            
            # Stage 4: Cross validation
            if self.config.get('cross_validation', {}).get('enabled', False):
                progress.start_stage(3, "Cross Validation")
                self.run_cross_validation()
                progress.end_stage(3, success=True)
            else:
                progress.current_stage = 3
            
            # Stage 5: Training
            progress.start_stage(4, "Training Models")
            self.train_models()
            progress.end_stage(4, success=True)
            
            # Stage 6: Evaluation
            progress.start_stage(5, "Evaluation")
            self.evaluate_models()
            progress.end_stage(5, success=True)
            
            # Generate report
            self.generate_final_report()
            
            progress.finish_pipeline(success=True)
            
            print(f"\n{'='*80}")
            print(f"PIPELINE COMPLETED SUCCESSFULLY!")
            print(f"Results: {self.experiment_dir}")
            print(f"{'='*80}\n")
            
            return self.evaluator.results
        
        except Exception as e:
            progress.finish_pipeline(success=False)
            print(f"\n✗ Pipeline failed: {str(e)}")
            import traceback
            traceback.print_exc()
            raise


def create_default_config():
    """Create configuration untuk CNN vs LSTM vs CNN-LSTM"""
    config = {
        'experiment_name': f'emg_cnn_lstm_comparison_{datetime.now().strftime("%Y%m%d_%H%M%S")}',
        
        'data_directory': 'D:\\Belajar\\Mr.Koder\\biosignal project\\data',
        
        'experiments_dir': 'experiments',
        'log_level': 'INFO',
        
        'sampling_rate': 1000,
        'segment_length': 10,
        'max_files_per_class': None,
        
        'preprocessing': {
            'apply_bandpass': True,
            'apply_notch': True,
            'normalize_method': 'rms',
            'segment_overlap': 0.5,
            'augment_data': True,
            'window_size': 512,
            'n_frames': 20
        },
        
        'feature_extraction': {
            'representation': 'power_spectral_density',
            'target_shape': [256, 256],
            'normalize': True,
            'normalize_method': 'minmax',
            'nperseg': 512,
            'noverlap': 256
        },
        
        'models': [
            {
                'name': 'standard_cnn',
                'type': 'standard_cnn',
                'architecture': 'standard',
                'params': {
                    'num_filters': 64,
                    'dropout_rate': 0.4,
                    'input_shape': [256, 256, 1],
                    'num_classes': 4
                }
            },
            {
                'name': 'standard_lstm',
                'type': 'standard_lstm',
                'architecture': 'standard',
                'params': {
                    'lstm_units': 64,
                    'dropout_rate': 0.3,
                    'input_shape': [256 * 256],
                    'num_classes': 4
                }
            },
            {
                'name': 'standard_cnn_lstm',
                'type': 'standard_cnn_lstm',
                'architecture': 'standard',
                'params': {
                    'num_filters': 64,
                    'lstm_units': 64,
                    'dropout_rate': 0.35,
                    'input_shape': [256, 256, 1],
                    'num_classes': 4
                }
            }
        ],
        
        'training': {
            'epochs': 100,
            'batch_size': 12,
            'use_class_weights': True,
            'verbose': 1,
            'early_stopping_patience': 25,
            'learning_rate': 0.0003,
            'optimizer': 'adamw'
        },
        
        'cross_validation': {
            'enabled': True,
            'n_splits': 5,
            'cv_epochs': 50,
            'cv_batch_size': 12,
            'random_state': 42,
            'stratify': True,
            'verbose': 1
        }
    }
    
    return config


def main():
    """Main function"""
    import argparse
    
    parser = argparse.ArgumentParser(description='EMG CNN-LSTM Comparison Pipeline')
    parser.add_argument('--config', type=str, default='config_cnn_lstm.json',
                       help='Configuration file')
    parser.add_argument('--data-dir', type=str, help='Data directory')
    parser.add_argument('--create-config', action='store_true',
                       help='Create default config')
    
    args = parser.parse_args()
    
    if args.create_config:
        config = create_default_config()
        FileUtils.save_json(config, 'config_cnn_lstm.json')
        print(" Configuration created: config_cnn_lstm.json")
        print("Please update data_directory and run again!")
        return
    
    # Load config
    if os.path.exists(args.config):
        config = FileUtils.load_json(args.config)
    else:
        print(f"Config not found: {args.config}")
        print("Creating default config...")
        config = create_default_config()
        FileUtils.save_json(config, args.config)
        print(f" Config saved: {args.config}")
        print("Please update and run again!")
        return
    
    if args.data_dir:
        config['data_directory'] = args.data_dir
    
    if not os.path.exists(config['data_directory']):
        print(f"✗ Data directory not found: {config['data_directory']}")
        return
    
    print("="*80)
    print("EMG CLASSIFICATION: CNN vs LSTM vs CNN-LSTM")
    print("="*80)
    
    pipeline = EMGPipeline(config)
    
    try:
        results = pipeline.run_complete_pipeline()
        print("\n Pipeline completed successfully!")
    except Exception as e:
        print(f"\n✗ Pipeline failed: {str(e)}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    import matplotlib
    matplotlib.use('Agg')
    
    main()