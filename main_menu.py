import os
import numpy as np
import pandas as pd
import sys
import argparse
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

from emg_data_loader import EMGDataLoader

from data_preprocessing import EMGPreprocessor, preprocess_batch_signals_to_spectrograms
from models import ModelBuilder
from train import DataSplitter, ModelTrainer, train_multiple_models
from evaluate import ModelEvaluator
from utils import FileUtils, PlotUtils, DetailedProgressTracker, ProgressBar, print_system_info

try:
    from config_loader import ConfigLoader
    CONFIG_AVAILABLE = True
except ImportError:
    CONFIG_AVAILABLE = False
    print("Warning: config_loader not found. Using default settings.")

DEFAULT_CONFIG = "config.json"


def print_banner():
    """Print welcome banner"""
    print("\n" + "="*90)
    print("║" + " "*88 + "║")
    print("║" + "EMG BIOSIGNAL CLASSIFICATION SYSTEM".center(88) + "║")
    print("║" + "Carpal Tunnel Syndrome (CTS) Severity Detection".center(88) + "║")
    print("║" + " "*88 + "║")
    print("="*90)
    print("║ 4-Class Classification: non-cts, mild, moderate, severe".ljust(89) + "║")
    print("║ Models: CNN, LSTM, CNN-LSTM".ljust(89) + "║")
    print("║ STFT Spectrogram Feature Extraction".ljust(89) + "║")
    print("║ 70-15-15 Data Split".ljust(89) + "║")
    print("="*90 + "\n")


def print_main_menu():
    """Print main menu"""
    print("\n" + "="*70)
    print("MAIN MENU")
    print("="*70)
    print("1.Process FULL_DATA (All data combined)")
    print("2.Process MOTORIK Signals")
    print("3.Process SENSORIK Signals")
    print("4.Process MOTORIK vs SENSORIK Comparison")
    print("5.Configuration Settings")
    print("6.System Information")
    print("0. Exit")
    print("="*70)


def print_model_selection_menu():
    """Print model selection menu"""
    print("\n" + "="*70)
    print("MODEL SELECTION")
    print("="*70)
    print("1.CNN (Convolutional Neural Network)")
    print("2.LSTM (Long Short-Term Memory)")
    print("3.CNN-LSTM (Hybrid Architecture)")
    print("4.ALL MODELS (Comparison)")
    print("0.Back to Main Menu")
    print("="*70)


def get_signal_type_choice():
    """Get user choice for signal type"""
    while True:
        print_main_menu()
        choice = input("\nEnter your choice (0-6): ").strip()
        
        if choice == "0":
            print("\n Thank you for using EMG Classification System!")
            sys.exit(0)
        elif choice == "1":
            return "full_data"
        elif choice == "2":
            return "motorik"
        elif choice == "3":
            return "sensorik"
        elif choice == "4":
            return "comparison"
        elif choice == "5":
            configure_settings()
            continue
        elif choice == "6":
            print_system_info()
            input("\nPress Enter to continue...")
            continue
        else:
            print("\nInvalid choice! Please enter 0-6.")


def get_model_selection():
    """Get user choice for model"""
    while True:
        print_model_selection_menu()
        choice = input("\nEnter your choice (0-4): ").strip()
        
        if choice == "0":
            return None
        elif choice == "1":
            return ["standard_cnn"]
        elif choice == "2":
            return ["standard_lstm"]
        elif choice == "3":
            return ["standard_cnn_lstm"]
        elif choice == "4":
            return ["standard_cnn", "standard_lstm", "standard_cnn_lstm"]
        else:
            print("\nInvalid choice! Please enter 0-4.")


def configure_settings():
    """Configure pipeline settings"""
    print("\n" + "="*70)
    print("CONFIGURATION SETTINGS")
    print("="*70)
    print("\nAdvanced settings - Use default values if unsure")
    print("\nCurrent settings:")
    print("  • Sampling Rate: 1000 Hz (auto-detected from header)")
    print("  • Segment Length: 10 seconds")
    print("  • Overlap: 50%")
    print("  • Augmentation: Enabled (Gaussian Noise)")
    print("  • Epochs: 100")
    print("  • Batch Size: 12")
    print("  • Learning Rate: 0.0003")
    print("\nTo modify, edit 'config_pipeline.json' manually.")
    input("\nPress Enter to continue...")


def validate_data_directory(base_dir, signal_type):
    """Validate that data directory exists"""
    
    if signal_type == "comparison":
        # Check both Motorik and Sensorik
        motorik_dir = os.path.join(base_dir, 'Motorik')
        sensorik_dir = os.path.join(base_dir, 'Sensorik')
        
        if not os.path.exists(motorik_dir):
            print(f"\nERROR: Motorik directory not found: {motorik_dir}")
            return False
        if not os.path.exists(sensorik_dir):
            print(f"\nERROR: Sensorik directory not found: {sensorik_dir}")
            return False
        
        # Check subfolders
        for signal_dir in [motorik_dir, sensorik_dir]:
            for class_folder in ['non_cts', 'mild', 'moderate', 'severe']:
                class_path = os.path.join(signal_dir, class_folder)
                if not os.path.exists(class_path):
                    print(f"\nERROR: Missing folder: {class_path}")
                    return False
        
        return True
    
    else:
        # Map signal_type to folder name
        folder_mapping = {
            'full_data': 'Full_Data',
            'motorik': 'Motorik',
            'sensorik': 'Sensorik'
        }
        
        folder_name = folder_mapping.get(signal_type)
        signal_dir = os.path.join(base_dir, folder_name)
        
        if not os.path.exists(signal_dir):
            print(f"\nERROR: Directory not found: {signal_dir}")
            print(f"\nExpected structure:")
            print(f"  {base_dir}/")
            print(f"    ├── {folder_name}/")
            print(f"    │   ├── non_cts/")
            print(f"    │   ├── mild/")
            print(f"    │   ├── moderate/")
            print(f"    │   └── severe/")
            return False
        
        # Check subfolders
        for class_folder in ['non_cts', 'mild', 'moderate', 'severe']:
            class_path = os.path.join(signal_dir, class_folder)
            if not os.path.exists(class_path):
                print(f"\nERROR: Missing folder: {class_path}")
                return False
        
        return True

def estimate_memory_requirements(n_signals, signal_length, segment_length, 
                                sampling_rate=1000, overlap=0.5, augment=True):
    """
    Estimate memory requirements before processing
    
    Returns:
        dict with memory estimates
    """
    segment_samples = int(segment_length * sampling_rate)
    step = int(segment_samples * (1 - overlap))
    
    segments_per_signal = max(1, (signal_length - segment_samples) // step + 1)
    
    if augment:
        segments_per_signal *= 2  # Augmentation doubles data
    
    total_segments = n_signals * segments_per_signal
    
    # Memory calculation (float32)
    memory_per_spec = 256 * 256 * 4  # bytes
    total_memory_bytes = total_segments * memory_per_spec
    total_memory_gb = total_memory_bytes / (1024**3)
    
    return {
        'segments_per_signal': segments_per_signal,
        'total_segments': total_segments,
        'memory_gb': total_memory_gb,
        'memory_mb': total_memory_gb * 1024,
        'safe': total_memory_gb < 4.0  # Safe if < 4GB
    }


def run_single_signal_pipeline(base_dir, signal_type, selected_models, config=None):
    """Run pipeline untuk single signal type with proper config loading"""
    
    # Map signal_type ke folder name
    folder_mapping = {
        'full_data': 'Full_Data',
        'motorik': 'Motorik',
        'sensorik': 'Sensorik'
    }
    
    folder_name = folder_mapping.get(signal_type)
    data_dir = os.path.join(base_dir, folder_name)
    
    print(f"\n{'='*90}")
    print(f"PROCESSING {folder_name.upper()} SIGNALS")
    print(f"{'='*90}")
    
    if config is None:
        if CONFIG_AVAILABLE and os.path.exists(DEFAULT_CONFIG):
            config_loader = ConfigLoader(DEFAULT_CONFIG)
            config = config_loader.load()
        else:
            raise ValueError("Config file is REQUIRED! Create config.json first.")
    preprocess_config = config.get('preprocessing', {})
    signal_config = config.get('signal_processing', {})
    feature_config = config.get('feature_extraction', {})
    sampling_rate = signal_config.get('sampling_rate', 12804)  
    segment_length = preprocess_config.get('segment_length_seconds', 0.04)

    print(f"\n{'='*70}")
    print(f"CONFIGURATION VERIFICATION")
    print(f"{'='*70}")
    print(f"  Sampling Rate: {sampling_rate} Hz")
    print(f"  Segment Length: {segment_length} seconds ({segment_length*1000:.1f} ms)")
    print(f"  Segment Samples: {int(segment_length * sampling_rate)}")
    print(f"  Batch Size: {config.get('training', {}).get('hyperparameters', {}).get('batch_size', 16)}")
    print(f"  Learning Rate: {config.get('training', {}).get('hyperparameters', {}).get('learning_rate', 0.001)}")
    print(f"{'='*70}\n")

    # Initialize progress tracker
    progress = DetailedProgressTracker(total_stages=5)
    progress.start_pipeline()
    
    try:
        # Stage 1: Load Data
        progress.start_stage(0, f"Loading {folder_name} Data")
        
        loader = EMGDataLoader(data_dir)
        signals, labels, info_list = loader.load_all_emg_data()
        
        if len(signals) == 0:
            print(f"\nNo data loaded from {data_dir}")
            return False, None
        
        loader.print_dataset_info(signals, labels, info_list)
        class_names = list(loader.class_mapping.keys())
        
        progress.end_stage(0, success=True)
        
        # Stage 2: Preprocessing & Feature Extraction
        progress.start_stage(1, "Preprocessing → STFT Spectrograms")
        
        sampling_rate = config.get('signal_processing', {}).get('sampling_rate', 1000)
        segment_length = config.get('preprocessing', {}).get('segment_length_seconds', 1.0)
        segment_overlap = config.get('preprocessing', {}).get('segment_overlap', 0.5)
        augment_enabled = config.get('preprocessing', {}).get('augmentation', {}).get('enabled', True)

        print(f"\nUsing OPTIMIZED parameters:")
        print(f"  Sampling rate: {sampling_rate} Hz")
        print(f"  Segment length: {segment_length} seconds ({segment_length*1000:.0f} ms)")
        print(f"  Segment overlap: {segment_overlap*100:.0f}%")
        print(f"  Samples per segment: {int(segment_length * sampling_rate)}")
        avg_signal_length = int(np.mean([len(s) if len(s.shape) == 1 else len(s) 
                                         for s in signals]))
        
        mem_estimate = estimate_memory_requirements(
            n_signals=len(signals),
            signal_length=avg_signal_length,
            segment_length=segment_length,
            sampling_rate=sampling_rate,
            overlap=segment_overlap,
            augment=augment_enabled
        )
        print(f"\nMemory Estimate:")
        print(f"   Segments per signal: ~{mem_estimate['segments_per_signal']}")
        print(f"   Total spectrograms: ~{mem_estimate['total_segments']:,}")
        print(f"   Memory required: ~{mem_estimate['memory_mb']:.0f} MB ({mem_estimate['memory_gb']:.2f} GB)")

        if not mem_estimate['safe']:
            print(f"\n WARNING: Memory usage may exceed available RAM!")
            print(f"   Recommended actions:")
            print(f"     1. Increase segment_length_seconds to 1.0-2.0")
            print(f"     2. Reduce segment_overlap to 0.3")
            print(f"     3. Disable augmentation temporarily")
            
            proceed = input(f"\n   Continue anyway? (y/n): ").strip().lower()
            if proceed != 'y':
                print(f"\nProcessing cancelled by user")
                return False, None

        print(f"Using config values:")
        print(f"  - Sampling rate: {sampling_rate} Hz")
        print(f"  - Segment length: {segment_length} seconds ({segment_length*1000:.1f} ms)")
        print(f"  - Segment overlap: {segment_overlap*100:.0f}%")
        
        preprocessor = EMGPreprocessor(
            sampling_rate=sampling_rate,
            segment_length=segment_length, 
            window_size=feature_config.get('stft', {}).get('window_size', 256),
            n_frames=feature_config.get('stft', {}).get('n_frames', 20)
        )
        
        all_spectrograms = []
        all_labels = []
        
        prog_bar = ProgressBar(len(signals), desc="Processing", unit="signal")
        
        successful_signals = 0
        failed_signals = 0
        
        for signal, label in zip(signals, labels):
            try:
                # Handle multi-channel: average across channels
                if len(signal.shape) > 1 and signal.shape[1] > 1:
                    # Pilih channel dengan variance tertinggi
                    variances = np.var(signal, axis=0)
                    best_channel = np.argmax(variances)
                    signal_1d = signal[:, best_channel]
                else:
                    signal_1d = signal.flatten()
                
                # Validasi signal
                if len(signal_1d) < 256:  # Minimum samples
                    print(f"  Signal too short: {len(signal_1d)} samples")
                    failed_signals += 1
                    continue
                
                # Preprocess to spectrograms dengan parameter optimal
                spectrograms, augmented = preprocessor.process_signal_to_spectrogram(
                    signal_1d,
                    apply_bandpass=config.get('preprocessing', {}).get('apply_bandpass_filter', True),
                    apply_notch=config.get('preprocessing', {}).get('apply_notch_filter', True),
                    normalize_method=config.get('preprocessing', {}).get('normalization_method', 'zscore'),
                    segment_overlap=segment_overlap,
                    augment_with_noise=augment_enabled,
                    augment_time_shift=True,
                    augment_amplitude=True
                )
                
                if spectrograms is not None and len(spectrograms) > 0:
                    # Validasi spectrograms
                    valid_spectrograms = []
                    for spec in spectrograms:
                        if spec.shape == (64, 64) and not np.all(spec == 0):
                            valid_spectrograms.append(spec)
                    
                    if valid_spectrograms:
                        all_spectrograms.extend(valid_spectrograms)
                        all_labels.extend([label] * len(valid_spectrograms))
                        successful_signals += 1
                    
                    # Augmented data
                    if augment_enabled and augmented is not None:
                        valid_augmented = []
                        for spec in augmented:
                            if spec.shape == (64, 64) and not np.all(spec == 0):
                                valid_augmented.append(spec)
                        
                        if valid_augmented:
                            all_spectrograms.extend(valid_augmented)
                            all_labels.extend([label] * len(valid_augmented))
                
            except Exception as e:
                print(f"  Error processing signal: {str(e)}")
                failed_signals += 1
            
            prog_bar.update(1)
        
        prog_bar.close()
        print(f"\nSignal processing summary:")
        print(f"  Successful: {successful_signals}/{len(signals)}")
        print(f"  Failed: {failed_signals}/{len(signals)}")
    
        if len(all_spectrograms) == 0:
            print("ERROR: No valid spectrograms generated!")
            return False, None
        
        spectrograms = np.array(all_spectrograms)
        spectrogram_labels = np.array(all_labels)
        
        print(f"\nGenerated {len(spectrograms)} spectrograms")
        print(f"  Shape: {spectrograms.shape}")
        print(f"  Labels distribution: {np.bincount(spectrogram_labels)}")
        
        
        # Verify shape
        if spectrograms.shape[1:3] != (64, 64):
            print(f"ERROR: Spectrogram shape mismatch: {spectrograms.shape[1:3]}")
            print(f"  Attempting to resize...")
        from scipy.ndimage import zoom
        resized_spectrograms = []
        for spec in spectrograms:
            zoom_factors = (256.0 / spec.shape[0], 256.0 / spec.shape[1])
            resized = zoom(spec, zoom_factors, order=1)
            resized_spectrograms.append(resized)
        spectrograms = np.array(resized_spectrograms)
        print(f"  Resized to: {spectrograms.shape}")
        progress.end_stage(1, success=True)
        
        # Stage 3: Model Training
        progress.start_stage(2, "Training Models")
        
        # Create experiment directory
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        experiment_dir = f"experiments/{folder_name.lower()}_classification_{timestamp}"
        os.makedirs(experiment_dir, exist_ok=True)
        
        models_config = []
        # Get target size from config
        target_h, target_w = config.get('feature_extraction', {}).get('spectrogram', {}).get('target_size', [64, 64])
        
        for model_type in selected_models:
            if model_type == 'standard_cnn':
                input_shape = [target_h, target_w, 1]
            elif model_type == 'standard_lstm':
                input_shape = [target_h * target_w]  # Flattened untuk LSTM
            elif model_type == 'standard_cnn_lstm':
                input_shape = [target_h, target_w, 1]
            else:
                input_shape = [target_h, target_w, 1]
            config_entry = {
                'name': f'{folder_name.lower()}_{model_type}',
                'type': model_type,
                'architecture': 'standard',
                'params': {
                    'input_shape': input_shape,
                    'num_classes': len(class_names)
                }
            }
            
            if model_type == 'standard_cnn':
                config_entry['params']['num_filters'] = 32
                config_entry['params']['dropout_rate'] = 0.5
            elif model_type == 'standard_lstm':
                config_entry['params']['lstm_units'] = 128
                config_entry['params']['dropout_rate'] = 0.4
            elif model_type == 'standard_cnn_lstm':
                config_entry['params']['num_filters'] = 32
                config_entry['params']['lstm_units'] = 64
                config_entry['params']['dropout_rate'] = 0.5
            
            models_config.append(config_entry)
        
        train_config = config.get('training', {})
        hyperparams = train_config.get('hyperparameters', {})
        
        train_params = {
            'epochs': hyperparams.get('epochs', 150),
            'batch_size': hyperparams.get('batch_size', 16),
            'learning_rate': hyperparams.get('learning_rate', 0.0001),
            'optimizer': hyperparams.get('optimizer', 'adamw'),
            'use_class_weights': train_config.get('class_weights', {}).get('enabled', True),
            'verbose': 1,
            'early_stopping_patience': train_config.get('callbacks', {}).get('early_stopping', {}).get('patience', 30)
        }
        
        print(f"\nTraining configuration:")
        print(f"  Models to train: {len(models_config)}")
        print(f"  Epochs: {train_params['epochs']}")
        print(f"  Batch size: {train_params['batch_size']}")
        print(f"  Learning rate: {train_params['learning_rate']}")
        
        # Train models
    except Exception as e:
        print(f"\nERROR in training: {str(e)}")
    try:
        models_dir = os.path.join(experiment_dir, 'models')
        trained_models, split_data = train_multiple_models(
            spectrograms,
            spectrogram_labels,
            class_names,
            models_config,
            save_dir=models_dir,
            **train_params
        )
        print(f"\nTraining completed. Models trained: {len(trained_models)}")
        successful_models = [name for name, result in trained_models.items() if 'error' not in result]
        if len(successful_models) == 0:
            print("WARNING: No models trained successfully!")
            progress.end_stage(2, success=False)
            progress.finish_pipeline(success=False)
            return False, experiment_dir    
        progress.end_stage(2, success=True)
    except Exception as e:
            print(f"\nERROR in training: {str(e)}")
            progress.end_stage(2, success=False)
            progress.finish_pipeline(success=False)
            return False, experiment_dir
        # Stage 4: Evaluation
    progress.start_stage(3, "Model Evaluation")

    progress.start_stage(4, "Generating Report")
    try:
        generate_final_report(experiment_dir, folder_name, comparison_df, class_names)
        progress.end_stage(4, success=True)
    except Exception as e:
        print(f"\nERROR generating report: {str(e)}")
        try:
            report_path = os.path.join(experiment_dir, 'FINAL_REPORT.md')
            with open(report_path, 'w', encoding='utf-8') as f:
                f.write(f"# EMG Classification Report - {folder_name}\n\n")
                f.write(f"**Date:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
                f.write(f"## Error Summary\n\n")
                f.write(f"Pipeline completed with errors.\n")
                f.write(f"Error: {str(e)}\n")
            print(f"Error report saved: {report_path}")
        except:
            pass
        progress.end_stage(4, success=False)

    progress.finish_pipeline(success=True if len(successful_models) > 0 else False)


      
    try:
        evaluator = ModelEvaluator(class_names)
        
        # Filter hanya models yang berhasil di-train
        valid_models = {}
        for model_name, model_info in trained_models.items():
            if 'error' not in model_info and 'trainer' in model_info:
                valid_models[model_name] = model_info
        
        if len(valid_models) > 0:
            evaluation_results = evaluator.evaluate_multiple_models(valid_models, split_data)
            
            # Generate comparison table
            comparison_df = evaluator.create_comparison_table()
            if comparison_df is not None and len(comparison_df) > 0:
                results_dir = os.path.join(experiment_dir, 'results')
                os.makedirs(results_dir, exist_ok=True)
                
                comparison_path = os.path.join(results_dir, 'model_comparison.csv')
                comparison_df.to_csv(comparison_path, index=False)
                
                print(f"\n{folder_name.upper()} Model Comparison:")
                print(comparison_df.to_string(index=False))
            else:
                print(f"\nNo comparison table generated")
                comparison_df = pd.DataFrame()  # Empty dataframe untuk avoid None
        else:
            print(f"\nNo valid models to evaluate")
            comparison_df = pd.DataFrame()
        
        # Plot confusion matrices jika ada results
        if hasattr(evaluator, 'results') and evaluator.results:
            plots_dir = os.path.join(experiment_dir, 'plots')
            evaluator.plot_all_confusion_matrices(save_dir=plots_dir)
        
        progress.end_stage(3, success=True)
    
    except Exception as e:
        print(f"\nERROR in evaluation: {str(e)}")
        comparison_df = pd.DataFrame()  # Empty dataframe
        progress.end_stage(3, success=False)

def generate_final_report(experiment_dir, dataset_name, comparison_df, class_names):
    """Generate final report markdown dengan handle untuk empty comparison_df"""
    
    report_path = os.path.join(experiment_dir, 'FINAL_REPORT.md')
    
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write(f"# EMG Classification Report - {dataset_name}\n\n")
        f.write(f"**Date:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        f.write("## Dataset Information\n\n")
        f.write(f"- **Dataset**: {dataset_name}\n")
        f.write(f"- **Classes**: {', '.join(class_names)}\n")
        
        if comparison_df is not None and hasattr(comparison_df, '__len__'):
            f.write(f"- **Models Trained**: {len(comparison_df)} models\n\n")
        else:
            f.write(f"- **Models Trained**: 0 models (training failed)\n\n")
        
        f.write("## Performance Results\n\n")
        
        if comparison_df is not None and hasattr(comparison_df, '__len__') and len(comparison_df) > 0:
            f.write("### Model Comparison\n\n")
            f.write("| Rank | Model | Accuracy | Precision | Recall | F1-Score |\n")
            f.write("|------|-------|----------|-----------|--------|----------|\n")
            
            for idx, row in comparison_df.iterrows():
                rank = idx + 1
                f.write(f"| {rank} | {row['Model']} | {row['Accuracy']:.4f} | ")
                f.write(f"{row['Precision (Weighted)']:.4f} | {row['Recall (Weighted)']:.4f} | ")
                f.write(f"{row['F1-Score (Weighted)']:.4f} |\n")
            
            # Best model
            best_model = comparison_df.iloc[0]
            f.write(f"\n### Best Model\n\n")
            f.write(f"- **Model**: {best_model['Model']}\n")
            f.write(f"- **Accuracy**: {best_model['Accuracy']:.4f} ({best_model['Accuracy']*100:.2f}%)\n")
            f.write(f"- **F1-Score**: {best_model['F1-Score (Weighted)']:.4f}\n\n")
        else:
            f.write("### No Performance Results Available\n\n")
            f.write("Model training or evaluation failed. Please check the logs for errors.\n\n")
        
        f.write("## Files Generated\n\n")
        f.write("- `models/` - Trained models (.h5)\n")
        f.write("- `plots/` - Confusion matrices & training curves\n")
        f.write("- `results/` - Evaluation metrics (CSV)\n")
        f.write("- `FINAL_REPORT.md` - This report\n")
        f.write("- `experiment.log` - Log file\n\n")
    
    print(f"Report saved: {report_path}")


def main():
    """Main function"""
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='EMG Biosignal Classification')
    parser.add_argument('--config', type=str, default=DEFAULT_CONFIG,
                       help='Path to config file (default: config.json)')
    args = parser.parse_args()
    
    # Load configuration
    if CONFIG_AVAILABLE and os.path.exists(args.config):
        print(f"\nLoading configuration from: {args.config}")
        config_loader = ConfigLoader(args.config)
        config = config_loader.load()
        config_loader.print_summary()
        
        # Get data directory from config
        DEFAULT_DATA_DIR = config.get('data.base_directory', 
                                      "D:\\Belajar\\Mr.Koder\\biosignal project\\data")
    else:
        if CONFIG_AVAILABLE:
            print(f"\nConfig file not found: {args.config}")
        print("Using default settings...")
        config = None
        DEFAULT_DATA_DIR = "D:\\Belajar\\Mr.Koder\\biosignal project\\data"
    
    print_banner()
    print_system_info()
    
    # Check if default exists
    if not os.path.exists(DEFAULT_DATA_DIR):
        print(f"\nDefault data directory not found: {DEFAULT_DATA_DIR}")
        print("\nPlease enter the path to your data directory:")
        print("(The directory should contain 'Full_Data', 'Motorik', and/or 'Sensorik' folders)")
        
        while True:
            user_dir = input("\nData directory path: ").strip().strip('"\'')
            if os.path.exists(user_dir):
                DEFAULT_DATA_DIR = user_dir
                break
            else:
                print(f"Directory not found: {user_dir}")
                retry = input("Try again? (y/n): ").strip().lower()
                if retry != 'y':
                    print("\n Exiting...")
                    return
    
    print(f"\nData directory: {DEFAULT_DATA_DIR}")
    
    # Main loop
    while True:
        # Get signal type choice
        signal_type = get_signal_type_choice()
        
        # Validate data directory
        if not validate_data_directory(DEFAULT_DATA_DIR, signal_type):
            input("\nPress Enter to continue...")
            continue
        
        # Get model selection
        selected_models = get_model_selection()
        if selected_models is None:
            continue
        
        # Confirm before starting
        print(f"\n{'='*70}")
        print("CONFIRMATION")
        print(f"{'='*70}")
        print(f"Signal Type: {signal_type.upper()}")
        print(f"Models: {', '.join(selected_models)}")
        print(f"Data Directory: {DEFAULT_DATA_DIR}")
        print(f"{'='*70}")
        
        confirm = input("\nStart processing? (y/n): ").strip().lower()
        if confirm != 'y':
            print("\nCancelled.")
            continue
        
        # Run pipeline
        if signal_type == "comparison":
            print("\nComparison mode will be implemented separately.")
            print("Please use 'Motorik' or 'Sensorik' individually for now.")
            continue
        else:
            success, result_dir = run_single_signal_pipeline(DEFAULT_DATA_DIR, signal_type, selected_models)
        
        # Ask if user wants to continue
        print(f"\n{'='*70}")
        another = input("\nProcess another dataset? (y/n): ").strip().lower()
        if another != 'y':
            print(f"\n{'='*70}")
            print("Thank you for using EMG Classification System!")
            print(f"{'='*70}\n")
            break




if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n Program interrupted by user. Exiting...")
    except Exception as e:
        print(f"\n\nFatal error: {str(e)}")
        import traceback
        traceback.print_exc()
        input("\nPress Enter to exit...")