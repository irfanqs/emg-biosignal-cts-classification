"""
Utility Functions untuk EMG Biosignal Classification
Plot functions, save/load utilities, dan helper functions
"""
import numpy as np
import matplotlib
matplotlib.use('Agg') 
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import pickle
import json
import os
import time
import sys
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')


plt.style.use('default')
sns.set_palette("husl")

class PlotUtils:
    """
    Utility class untuk plotting dan visualisasi
    """
    
    @staticmethod
    def plot_signal(signal, sampling_rate=1000, title="EMG Signal", 
                   figsize=(12, 4), save_path=None):
        """
        Plot time-domain signal
        
        Args:
            signal (array): Signal data
            sampling_rate (int): Sampling rate
            title (str): Plot title
            figsize (tuple): Figure size
            save_path (str): Path to save plot
        """
        time = np.arange(len(signal)) / sampling_rate
        
        plt.figure(figsize=figsize)
        plt.plot(time, signal, 'b-', linewidth=0.8)
        plt.xlabel('Time (s)')
        plt.ylabel('Amplitude')
        plt.title(title)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Signal plot saved to: {save_path}")
        
        plt.show()
    
    @staticmethod
    def plot_spectrogram(spectrogram, frequencies=None, times=None, 
                        title="Spectrogram", figsize=(10, 6), save_path=None):
        """
        Plot spectrogram
        
        Args:
            spectrogram (array): Spectrogram data
            frequencies (array): Frequency values
            times (array): Time values
            title (str): Plot title
            figsize (tuple): Figure size
            save_path (str): Path to save plot
        """
        plt.figure(figsize=figsize)
        
        if frequencies is not None and times is not None:
            extent = [times[0], times[-1], frequencies[0], frequencies[-1]]
            plt.imshow(spectrogram, aspect='auto', origin='lower', 
                      extent=extent, cmap='viridis')
            plt.xlabel('Time (s)')
            plt.ylabel('Frequency (Hz)')
        else:
            plt.imshow(spectrogram, aspect='auto', origin='lower', cmap='viridis')
            plt.xlabel('Time bins')
            plt.ylabel('Frequency bins')
        
        plt.colorbar(label='Magnitude')
        plt.title(title)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Spectrogram plot saved to: {save_path}")
        
        plt.show()
    
    @staticmethod
    def plot_multiple_signals(signals, labels, sampling_rate=1000, 
                             figsize=(15, 10), save_path=None):
        """
        Plot multiple signals dalam subplots
        
        Args:
            signals (list): List of signals
            labels (list): List of labels
            sampling_rate (int): Sampling rate
            figsize (tuple): Figure size
            save_path (str): Path to save plot
        """
        n_signals = len(signals)
        rows = (n_signals + 2) // 3  
        cols = min(3, n_signals)
        
        fig, axes = plt.subplots(rows, cols, figsize=figsize)
        if n_signals == 1:
            axes = [axes]
        elif rows == 1:
            axes = axes if n_signals > 1 else [axes]
        else:
            axes = axes.flatten()
        
        for i, (signal, label) in enumerate(zip(signals, labels)):
            time = np.arange(len(signal)) / sampling_rate
            
            ax = axes[i] if n_signals > 1 else axes[0]
            ax.plot(time, signal, linewidth=0.8)
            ax.set_title(f'{label}')
            ax.set_xlabel('Time (s)')
            ax.set_ylabel('Amplitude')
            ax.grid(True, alpha=0.3)
        
        for i in range(n_signals, len(axes)):
            axes[i].set_visible(False)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Multiple signals plot saved to: {save_path}")
        
        plt.show()
    
    @staticmethod
    def plot_training_history(history, save_path=None, figsize=(15, 5)):
        """
        Plot training history dengan loss dan accuracy
        
        Args:
            history: Keras training history atau dict dengan 'loss', 'accuracy', dll
            save_path (str): Path untuk save plot
            figsize (tuple): Ukuran figure
        """
        # Handle both Keras history object and dict
        if hasattr(history, 'history'):
            history_dict = history.history
        else:
            history_dict = history
        
        metrics_to_plot = []
        if 'accuracy' in history_dict:
            metrics_to_plot.append(('accuracy', 'Accuracy'))
        if 'loss' in history_dict:
            metrics_to_plot.append(('loss', 'Loss'))
        
        if not metrics_to_plot:
            print("No metrics found in history")
            return
        
        n_metrics = len(metrics_to_plot)
        fig, axes = plt.subplots(1, n_metrics, figsize=figsize)
        
        if n_metrics == 1:
            axes = [axes]
        
        for idx, (metric_key, metric_title) in enumerate(metrics_to_plot):
            ax = axes[idx]
            
            epochs = range(1, len(history_dict[metric_key]) + 1)
            
            # Plot training metric
            ax.plot(epochs, history_dict[metric_key], 'b-o', 
                   label=f'Training {metric_title}', linewidth=2, markersize=4)
            
            # Plot validation metric if exists
            val_metric_key = f'val_{metric_key}'
            if val_metric_key in history_dict:
                ax.plot(epochs, history_dict[val_metric_key], 'r-s',
                       label=f'Validation {metric_title}', linewidth=2, markersize=4)
            
            ax.set_xlabel('Epoch', fontsize=12, fontweight='bold')
            ax.set_ylabel(metric_title, fontsize=12, fontweight='bold')
            ax.set_title(f'Model {metric_title}', fontsize=14, fontweight='bold')
            ax.legend(loc='best', fontsize=10)
            ax.grid(True, alpha=0.3)
            
            # Add best value annotation
            if val_metric_key in history_dict:
                if metric_key == 'loss':
                    best_val = min(history_dict[val_metric_key])
                    best_epoch = history_dict[val_metric_key].index(best_val) + 1
                else:
                    best_val = max(history_dict[val_metric_key])
                    best_epoch = history_dict[val_metric_key].index(best_val) + 1
                
                ax.axvline(x=best_epoch, color='g', linestyle='--', alpha=0.5)
                ax.text(best_epoch, ax.get_ylim()[1] * 0.95,
                       f'Best: {best_val:.4f}\n(Epoch {best_epoch})',
                       ha='center', va='top', fontsize=9,
                       bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.7))
        
        plt.tight_layout()
        
        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Training history plot saved: {save_path}")
        
        plt.close()
    
    @staticmethod
    def plot_class_distribution(labels, class_names, title="Class Distribution", 
                               figsize=(10, 6), save_path=None):
        """
        Plot class distribution
        
        Args:
            labels (array): Label array
            class_names (list): Class names
            title (str): Plot title
            figsize (tuple): Figure size
            save_path (str): Path to save plot
        """
        unique_labels, counts = np.unique(labels, return_counts=True)
        
        plt.figure(figsize=figsize)
        bars = plt.bar([class_names[i] for i in unique_labels], counts, 
                       color=sns.color_palette("husl", len(unique_labels)))
        
        for bar, count in zip(bars, counts):
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                    str(count), ha='center', va='bottom')
        
        plt.xlabel('Classes')
        plt.ylabel('Count')
        plt.title(title)
        plt.xticks(rotation=45)
        plt.grid(axis='y', alpha=0.3)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Class distribution plot saved to: {save_path}")
        
        plt.show()
    
    @staticmethod
    def plot_model_comparison(comparison_df, metric='F1-Score (Weighted)', 
                             figsize=(12, 6), save_path=None):
        """
        Plot model comparison
        
        Args:
            comparison_df (DataFrame): Model comparison data
            metric (str): Metric to plot
            figsize (tuple): Figure size
            save_path (str): Path to save plot
        """
        if metric not in comparison_df.columns:
            print(f"Metric '{metric}' not found in comparison data")
            return
        
        plt.figure(figsize=figsize)
        
        df_sorted = comparison_df.sort_values(metric, ascending=True)
        
        bars = plt.barh(df_sorted['Model'], df_sorted[metric], 
                       color=sns.color_palette("viridis", len(df_sorted)))
        
        for bar, value in zip(bars, df_sorted[metric]):
            plt.text(value + 0.005, bar.get_y() + bar.get_height()/2, 
                    f'{value:.3f}', va='center', ha='left')
        
        plt.xlabel(metric)
        plt.ylabel('Models')
        plt.title(f'Model Comparison - {metric}')
        plt.grid(axis='x', alpha=0.3)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Model comparison plot saved to: {save_path}")
        
        plt.show()

class FileUtils:
    """
    Utility class untuk file operations
    """
    
    @staticmethod
    def save_pickle(data, filepath):
        """
        Save data sebagai pickle file
        
        Args:
            data: Data to save
            filepath (str): Path to save file
        """
        dir_path = os.path.dirname(filepath)
        
        if dir_path:  
            os.makedirs(dir_path, exist_ok=True)
        
        with open(filepath, 'wb') as f:
            pickle.dump(data, f)
        
        print(f"Data saved to: {filepath}")
    
    @staticmethod
    def load_pickle(filepath):
        """
        Load data dari pickle file
        
        Args:
            filepath (str): Path to pickle file
            
        Returns:
            Data from pickle file
        """
        with open(filepath, 'rb') as f:
            data = pickle.load(f)
        
        print(f"Data loaded from: {filepath}")
        return data
    
    @staticmethod
    @staticmethod
    def save_json(data, filepath, indent=2):
        """
        Save data sebagai JSON file
        
        Args:
            data: Data to save (must be JSON serializable)
            filepath (str): Path to save file
            indent (int): JSON indentation
        """
        dir_path = os.path.dirname(filepath)
        
        if dir_path:  
            os.makedirs(dir_path, exist_ok=True)
        
        def convert_numpy(obj):
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, dict):
                return {key: convert_numpy(value) for key, value in obj.items()}
            elif isinstance(obj, list):
                return [convert_numpy(item) for item in obj]
            else:
                return obj
        
        data_converted = convert_numpy(data)
        
        with open(filepath, 'w') as f:
            json.dump(data_converted, f, indent=indent)
        
        print(f"JSON data saved to: {filepath}")
    
    @staticmethod
    def load_json(filepath):
        """
        Load data dari JSON file
        
        Args:
            filepath (str): Path to JSON file
            
        Returns:
            Data from JSON file
        """
        with open(filepath, 'r') as f:
            data = json.load(f)
        
        print(f"JSON data loaded from: {filepath}")
        return data
    
    @staticmethod
    def create_experiment_folder(base_dir='experiments', experiment_name=None):
        """
        Create folder untuk experiment dengan timestamp
        
        Args:
            base_dir (str): Base directory
            experiment_name (str): Experiment name
            
        Returns:
            str: Path to experiment folder
        """
        if experiment_name is None:
            experiment_name = datetime.now().strftime("exp_%Y%m%d_%H%M%S")
        
        exp_folder = os.path.join(base_dir, experiment_name)
        os.makedirs(exp_folder, exist_ok=True)
        
        subfolders = ['models', 'plots', 'results', 'logs']
        for subfolder in subfolders:
            os.makedirs(os.path.join(exp_folder, subfolder), exist_ok=True)
        
        print(f"Experiment folder created: {exp_folder}")
        return exp_folder

class DataUtils:
    """
    Utility class untuk data processing operations
    """
    
    @staticmethod
    def normalize_array(arr, method='minmax'):
        """
        Normalize array menggunakan different methods
        
        Args:
            arr (array): Input array
            method (str): Normalization method
            
        Returns:
            array: Normalized array
        """
        if method == 'minmax':
            min_val, max_val = np.min(arr), np.max(arr)
            if max_val > min_val:
                return (arr - min_val) / (max_val - min_val)
            else:
                return arr
                
        elif method == 'zscore':
            mean_val, std_val = np.mean(arr), np.std(arr)
            if std_val > 0:
                return (arr - mean_val) / std_val
            else:
                return arr
                
        elif method == 'robust':
            median_val = np.median(arr)
            mad = np.median(np.abs(arr - median_val))
            if mad > 0:
                return (arr - median_val) / mad
            else:
                return arr
        
        return arr
    
    @staticmethod
    def pad_sequences(sequences, maxlen=None, padding='post', truncating='post', value=0.0):
        """
        Pad sequences to same length
        
        Args:
            sequences (list): List of sequences
            maxlen (int): Maximum length
            padding (str): Padding direction ('pre' or 'post')
            truncating (str): Truncating direction ('pre' or 'post')
            value (float): Padding value
            
        Returns:
            array: Padded sequences
        """
        if maxlen is None:
            maxlen = max(len(seq) for seq in sequences)
        
        padded = np.full((len(sequences), maxlen), value, dtype=np.float32)
        
        for i, seq in enumerate(sequences):
            seq = np.array(seq, dtype=np.float32)
            
            if len(seq) > maxlen:
                if truncating == 'pre':
                    seq = seq[-maxlen:]
                else:  # 'post'
                    seq = seq[:maxlen]
            
            if len(seq) < maxlen:
                if padding == 'pre':
                    padded[i, -len(seq):] = seq
                else:  # 'post'
                    padded[i, :len(seq)] = seq
            else:
                padded[i] = seq
        
        return padded
    
    @staticmethod
    def balance_dataset(X, y, method='oversample', random_state=42):
        """
        Balance dataset menggunakan sampling techniques
        
        Args:
            X (array): Features
            y (array): Labels
            method (str): Balancing method ('oversample', 'undersample')
            random_state (int): Random state
            
        Returns:
            tuple: (X_balanced, y_balanced)
        """
        from collections import Counter
        
        unique_classes, class_counts = np.unique(y, return_counts=True)
        print(f"Original class distribution: {dict(zip(unique_classes, class_counts))}")
        
        if method == 'oversample':
            target_count = max(class_counts)
            
            X_balanced = []
            y_balanced = []
            
            np.random.seed(random_state)
            
            for class_label in unique_classes:
                class_indices = np.where(y == class_label)[0]
                class_X = X[class_indices]
                class_y = y[class_indices]
                
                if len(class_indices) < target_count:
                    oversample_indices = np.random.choice(
                        len(class_indices), 
                        target_count - len(class_indices), 
                        replace=True
                    )
                    
                    oversample_X = class_X[oversample_indices]
                    oversample_y = class_y[oversample_indices]
                    
                    class_X = np.vstack([class_X, oversample_X])
                    class_y = np.hstack([class_y, oversample_y])
                
                X_balanced.append(class_X)
                y_balanced.append(class_y)
            
            X_balanced = np.vstack(X_balanced)
            y_balanced = np.hstack(y_balanced)
            
        elif method == 'undersample':
            target_count = min(class_counts)
            
            X_balanced = []
            y_balanced = []
            
            np.random.seed(random_state)
            
            for class_label in unique_classes:
                class_indices = np.where(y == class_label)[0]
                
                undersample_indices = np.random.choice(
                    class_indices, target_count, replace=False
                )
                
                X_balanced.append(X[undersample_indices])
                y_balanced.append(y[undersample_indices])
            
            X_balanced = np.vstack(X_balanced)
            y_balanced = np.hstack(y_balanced)
        
        else:
            raise ValueError(f"Unknown balancing method: {method}")
        
        shuffle_indices = np.random.permutation(len(X_balanced))
        X_balanced = X_balanced[shuffle_indices]
        y_balanced = y_balanced[shuffle_indices]
        
        new_unique, new_counts = np.unique(y_balanced, return_counts=True)
        print(f"Balanced class distribution: {dict(zip(new_unique, new_counts))}")
        
        return X_balanced, y_balanced

class LogUtils:
    """
    Utility class untuk logging
    """
    
    @staticmethod
    def setup_logging(log_file=None, level='INFO'):
        """
        Setup logging configuration
        
        Args:
            log_file (str): Log file path
            level (str): Logging level
        """
        import logging
        
        log_format = '%(asctime)s - %(levelname)s - %(message)s'
        
        if log_file:
            os.makedirs(os.path.dirname(log_file), exist_ok=True)
            logging.basicConfig(
                level=getattr(logging, level.upper()),
                format=log_format,
                handlers=[
                    logging.FileHandler(log_file),
                    logging.StreamHandler()
                ]
            )
        else:
            logging.basicConfig(
                level=getattr(logging, level.upper()),
                format=log_format
            )
        
        return logging.getLogger(__name__)
    
    @staticmethod
    def log_experiment_info(logger, config):
        """
        Log experiment information
        
        Args:
            logger: Logger instance
            config (dict): Configuration dictionary
        """
        logger.info("="*60)
        logger.info("EXPERIMENT CONFIGURATION")
        logger.info("="*60)
        
        for key, value in config.items():
            logger.info(f"{key}: {value}")
        
        logger.info("="*60)

def print_system_info():
    """
    Print system information
    """
    import platform
    import tensorflow as tf
    
    print("="*60)
    print("SYSTEM INFORMATION")
    print("="*60)
    print(f"Python version: {platform.python_version()}")
    print(f"Platform: {platform.platform()}")
    print(f"TensorFlow version: {tf.__version__}")
    
    physical_devices = tf.config.list_physical_devices('GPU')
    if physical_devices:
        print(f"GPU devices: {len(physical_devices)}")
        for i, device in enumerate(physical_devices):
            print(f"  GPU {i}: {device}")
    else:
        print("No GPU devices found")
    
    print("="*60)

def create_project_structure(base_dir='emg_classification_project'):
    """
    Create complete project structure
    
    Args:
        base_dir (str): Base project directory
        
    Returns:
        str: Project directory path
    """
    directories = [
        '',  
        'data',
        'data/raw',
        'data/processed',
        'experiments',
        'models',
        'results',
        'plots',
        'logs',
        'config'
    ]
    
    for directory in directories:
        dir_path = os.path.join(base_dir, directory)
        os.makedirs(dir_path, exist_ok=True)
    
    init_files = [
        '__init__.py',
        'config/__init__.py'
    ]
    
    for init_file in init_files:
        init_path = os.path.join(base_dir, init_file)
        if not os.path.exists(init_path):
            with open(init_path, 'w') as f:
                f.write('# EMG Classification Project\n')
    
    print(f"Project structure created at: {base_dir}")
    return base_dir

if __name__ == "__main__":
    print("Testing Utility Functions...")
    
    print("\n1. Testing PlotUtils:")
    
    np.random.seed(42)
    dummy_signal = np.random.randn(1000) + np.sin(np.linspace(0, 10*np.pi, 1000))
    dummy_spectrogram = np.random.rand(64, 128)
    
    try:
        plt.ioff()  
        PlotUtils.plot_signal(dummy_signal, title="Test Signal")
        plt.close('all')
        print("Signal plotting successful!")
    except Exception as e:
        print(f"Signal plotting error: {str(e)}")
    
    print("\n2. Testing FileUtils:")
    
    test_data = {
        'array': np.array([1, 2, 3, 4, 5]),
        'list': [1, 2, 3],
        'dict': {'key': 'value'},
        'number': 42
    }
    
    try:
        test_file = 'test_data.pkl'
        FileUtils.save_pickle(test_data, test_file)
        loaded_data = FileUtils.load_pickle(test_file)
        os.remove(test_file)  
        print("Pickle save/load successful!")
    except Exception as e:
        print(f"Pickle save/load error: {str(e)}")
    
    print("\n3. Testing DataUtils:")
    
    test_array = np.array([1, 5, 10, 15, 20])
    normalized = DataUtils.normalize_array(test_array, method='minmax')
    print(f"   Original: {test_array}")
    print(f"   Normalized: {normalized}")
    print("Array normalization successful!")
    
    sequences = [[1, 2, 3], [4, 5], [6, 7, 8, 9, 10]]
    padded = DataUtils.pad_sequences(sequences, maxlen=5)
    print(f"   Original sequences: {sequences}")
    print(f"   Padded shape: {padded.shape}")
    print("Sequence padding successful!")
    
    print("\n4. Testing Project Structure:")
    try:
        test_project = create_project_structure('test_project')
        import shutil
        if os.path.exists('test_project'):
            shutil.rmtree('test_project')
        print("Project structure creation successful!")
    except Exception as e:
        print(f"Project structure error: {str(e)}")
    
    print("\nAll utility function tests completed!")


class ProgressTracker:
    """
    Class untuk tracking progress di seluruh pipeline
    """
    
    def __init__(self, total_stages=6):
        """
        Initialize progress tracker
        
        Args:
            total_stages: Total tahap dalam pipeline
        """
        self.total_stages = total_stages
        self.current_stage = 0
        self.stage_names = [
            "Loading Data",
            "Preprocessing",
            "Feature Extraction",
            "Cross Validation",
            "Model Training",
            "Evaluation"
        ]
    
    def update_stage(self, stage_num, stage_name):
        """
        Update tahap saat ini
        
        Args:
            stage_num: Nomor tahap (0-based)
            stage_name: Nama tahap
        """
        self.current_stage = stage_num
        stage_progress = ((stage_num) / self.total_stages) * 100
        
        print(f"\n{'='*80}")
        print(f"PROGRESS: Stage {stage_num+1}/{self.total_stages} ({stage_progress:.1f}%)")
        print(f"{'='*80}")
        print(f"Current: {stage_name}")
        print(f"{'='*80}")
    
    def print_overall_progress(self):
        """Print overall pipeline progress"""
        overall_progress = ((self.current_stage + 1) / self.total_stages) * 100
        bar_length = 40
        filled = int((overall_progress / 100) * bar_length)
        bar = '█' * filled + '░' * (bar_length - filled)
        
        print(f"\nOverall Progress: [{bar}] {overall_progress:.1f}%")
class DetailedProgressTracker:
    """
    Enhanced progress tracker with time estimation and detailed logging
    """
    
    def __init__(self, total_stages=6):
        self.total_stages = total_stages
        self.current_stage = 0
        self.stage_start_times = {}
        self.stage_end_times = {}
        self.pipeline_start_time = time.time()
        
        self.stage_names = [
            "Loading EMG Data",
            "Preprocessing Signals", 
            "Extracting Features (STFT)",
            "Cross Validation (5-Fold)",
            "Training Models",
            "Evaluating Models"
        ]
        
        self.estimated_times = {
            0: 2,    # Loading: 2 min
            1: 10,   # Preprocessing: 10 min
            2: 15,   # Feature extraction: 15 min
            3: 120,  # Cross validation: 2 hours
            4: 180,  # Training: 3 hours
            5: 5     # Evaluation: 5 min
        }
    
    def start_pipeline(self):
        """Start pipeline tracking"""
        self.pipeline_start_time = time.time()
        print("\n" + "="*90)
        print("EMG CLASSIFICATION PIPELINE STARTED")
        print("="*90)
        print(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"Total stages: {self.total_stages}")
        
        total_estimated = sum(self.estimated_times.values())
        print(f"Estimated total time: {self._format_time(total_estimated * 60)}")
        print("="*90 + "\n")
    
    def start_stage(self, stage_num, stage_name=None):
        """Start a new stage"""
        self.current_stage = stage_num
        self.stage_start_times[stage_num] = time.time()
        
        if stage_name is None:
            stage_name = self.stage_names[stage_num]
        
        stage_progress = ((stage_num) / self.total_stages) * 100
        
        print(f"\n{'='*90}")
        print(f"STAGE {stage_num + 1}/{self.total_stages}: {stage_name.upper()}")
        print(f"{'='*90}")
        print(f"Stage progress: {stage_progress:.1f}% | Start: {datetime.now().strftime('%H:%M:%S')}")
        
        estimated_mins = self.estimated_times.get(stage_num, 0)
        if estimated_mins > 0:
            print(f"Estimated time for this stage: {self._format_time(estimated_mins * 60)}")
        
        self._print_eta()
        print("="*90 + "\n")
    
    def end_stage(self, stage_num, success=True):
        """End current stage"""
        if stage_num not in self.stage_start_times:
            return
        
        self.stage_end_times[stage_num] = time.time()
        elapsed = self.stage_end_times[stage_num] - self.stage_start_times[stage_num]
        
        status = "COMPLETED" if success else "FAILED"
        stage_name = self.stage_names[stage_num]
        
        print(f"\n{'-'*90}")
        print(f"{status} - Stage {stage_num + 1}: {stage_name}")
        print(f"Time taken: {self._format_time(elapsed)}")
        print(f"{'-'*90}\n")
        
        self._print_overall_progress()
    
    def update_substage(self, substage_name, current, total, extra_info=""):
        """Update progress within a stage"""
        if total > 0:
            percentage = (current / total) * 100
            bar_length = 50
            filled = int((percentage / 100) * bar_length)
            bar = '█' * filled + '░' * (bar_length - filled)
            
            status = f"\r  └─ {substage_name}: [{bar}] {percentage:.1f}% ({current}/{total})"
            if extra_info:
                status += f" | {extra_info}"
            
            print(status, end='', flush=True)
            
            if current == total:
                print()
    
    def _print_overall_progress(self):
        """Print overall pipeline progress"""
        completed_stages = len(self.stage_end_times)
        overall_progress = (completed_stages / self.total_stages) * 100
        
        bar_length = 50
        filled = int((overall_progress / 100) * bar_length)
        bar = '█' * filled + '░' * (bar_length - filled)
        
        elapsed_total = time.time() - self.pipeline_start_time
        
        print(f"\n{'OVERALL PIPELINE PROGRESS':^90}")
        print(f"{'-'*90}")
        print(f"  [{bar}] {overall_progress:.1f}%")
        print(f"  Completed: {completed_stages}/{self.total_stages} stages")
        print(f"  Total elapsed: {self._format_time(elapsed_total)}")
        
        print(f"\n  Stage Breakdown:")
        for i in range(self.total_stages):
            if i in self.stage_end_times:
                stage_time = self.stage_end_times[i] - self.stage_start_times[i]
                print(f"    {i+1}. {self.stage_names[i]:<30} {self._format_time(stage_time)}")
            elif i in self.stage_start_times:
                print(f"    {i+1}. {self.stage_names[i]:<30} In progress...")
            else:
                print(f"    {i+1}. {self.stage_names[i]:<30}  Pending")
        
        print(f"{'-'*90}\n")
    
    def _print_eta(self):
        """Print estimated time to completion"""
        if len(self.stage_end_times) == 0:
            remaining_time = sum(self.estimated_times.values()) * 60
        else:
            completed_stages = len(self.stage_end_times)
            avg_time_per_stage = sum(
                self.stage_end_times[i] - self.stage_start_times[i] 
                for i in self.stage_end_times
            ) / completed_stages
            
            remaining_stages = self.total_stages - completed_stages - 1  # -1 for current
            remaining_time = remaining_stages * avg_time_per_stage
            
            if self.current_stage in self.estimated_times:
                remaining_time += self.estimated_times[self.current_stage] * 60
        
        eta = datetime.now() + timedelta(seconds=remaining_time)
        print(f"Estimated completion: {eta.strftime('%Y-%m-%d %H:%M:%S')} (in {self._format_time(remaining_time)})")
    
    def finish_pipeline(self, success=True):
        """Finish pipeline tracking"""
        total_time = time.time() - self.pipeline_start_time
        
        print("\n" + "="*90)
        if success:
            print("PIPELINE COMPLETED SUCCESSFULLY!")
        else:
            print("PIPELINE COMPLETED WITH ERRORS")
        print("="*90)
        print(f"End time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"Total time: {self._format_time(total_time)}")
        print(f"Stages completed: {len(self.stage_end_times)}/{self.total_stages}")
        
        # Show time per stage
        print("\nTime breakdown:")
        for i in range(self.total_stages):
            if i in self.stage_end_times:
                stage_time = self.stage_end_times[i] - self.stage_start_times[i]
                percentage = (stage_time / total_time) * 100
                print(f"  {i+1}. {self.stage_names[i]:<30} {self._format_time(stage_time):>12} ({percentage:.1f}%)")
        
        print("="*90 + "\n")
    
    def _format_time(self, seconds):
        """Format seconds to human readable time"""
        if seconds < 60:
            return f"{int(seconds)}s"
        elif seconds < 3600:
            mins = int(seconds // 60)
            secs = int(seconds % 60)
            return f"{mins}m {secs}s"
        else:
            hours = int(seconds // 3600)
            mins = int((seconds % 3600) // 60)
            return f"{hours}h {mins}m"


# Progress bar helper for tqdm compatibility
class ProgressBar:
    """Simple progress bar for operations"""
    
    def __init__(self, total, desc="Processing", unit="item"):
        self.total = total
        self.current = 0
        self.desc = desc
        self.unit = unit
        self.start_time = time.time()
    
    def update(self, n=1):
        """Update progress by n items"""
        self.current = min(self.current + n, self.total)
        self._display()
    
    def _display(self):
        """Display progress bar"""
        percentage = (self.current / self.total) * 100 if self.total > 0 else 0
        bar_length = 40
        filled = int((percentage / 100) * bar_length)
        bar = '█' * filled + '░' * (bar_length - filled)
        
        elapsed = time.time() - self.start_time
        rate = self.current / elapsed if elapsed > 0 else 0
        
        status = f"\r{self.desc}: [{bar}] {percentage:.1f}% | {self.current}/{self.total} {self.unit}s | {rate:.1f} {self.unit}s/s"
        print(status, end='', flush=True)
        
        if self.current >= self.total:
            print()  # Newline when complete
    
    def close(self):
        """Close progress bar"""
        if self.current < self.total:
            self.current = self.total
            self._display()
