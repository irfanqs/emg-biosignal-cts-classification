import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from imblearn.over_sampling import SMOTE
import tensorflow as tf
from tensorflow.keras.callbacks import (
    EarlyStopping, ReduceLROnPlateau, ModelCheckpoint, 
    LambdaCallback, ProgbarLogger, Callback
)
from tensorflow.keras.optimizers import Adam
import sys
from models import ModelBuilder
from datetime import datetime
import json
import time
import pickle
import os
import warnings
warnings.filterwarnings('ignore')
import importlib


class EnhancedTrainingCallback(Callback):
    """
    Enhanced callback untuk menampilkan progress training yang sangat jelas
    dengan grafik ASCII real-time
    """
    
    def __init__(self, epochs, model_name="Model", save_plots=True, plots_dir=None):
        super().__init__()
        self.epochs = epochs
        self.model_name = model_name
        self.save_plots = save_plots
        self.plots_dir = plots_dir
        self.epoch_start_time = None
        self.history_tracker = {
            'loss': [],
            'accuracy': [],
            'val_loss': [],
            'val_accuracy': []
        }
    
    def on_train_begin(self, logs=None):
        print(f"\n{'='*80}")
        print(f"Training {self.model_name}")
        print(f"{'='*80}")
        print(f"Total epochs: {self.epochs}")
        print(f"{'='*80}\n")
    
    def on_epoch_begin(self, epoch, logs=None):
        self.epoch_start_time = time.time()
        print(f"\n{'─'*80}")
        print(f"Epoch {epoch+1}/{self.epochs}")
        print(f"{'─'*80}")
    
    def on_epoch_end(self, epoch, logs=None):
        elapsed = time.time() - self.epoch_start_time
        
        # Get metrics
        loss = logs.get('loss', 0)
        acc = logs.get('accuracy', 0)
        val_loss = logs.get('val_loss', 0)
        val_acc = logs.get('val_accuracy', 0)
        opt = self.model.optimizer
        if hasattr(opt, "lr"):
            lr = float(tf.keras.backend.get_value(opt.lr))
        else:
            lr = float(tf.keras.backend.get_value(opt.learning_rate))

        # Store history
        self.history_tracker['loss'].append(loss)
        self.history_tracker['accuracy'].append(acc)
        self.history_tracker['val_loss'].append(val_loss)
        self.history_tracker['val_accuracy'].append(val_acc)
        
        # Progress bar
        progress = (epoch + 1) / self.epochs
        bar_length = 50
        filled = int(progress * bar_length)
        bar = '█' * filled + '▒' * (bar_length - filled)
        
        print(f"\n[{bar}] {progress*100:.1f}%")
        print(f"{'─'*80}")
        print(f"Metrics:")
        print(f"Loss: {loss:.4f} │ Accuracy: {acc:.4f} ({acc*100:.2f}%)")
        print(f"Val Loss: {val_loss:.4f} │ Val Accuracy: {val_acc:.4f} ({val_acc*100:.2f}%)")
        print(f"{'─'*80}")
        print(f"Time: {elapsed:.1f}s │ LR: {lr:.2e}")
        
        # Best model indicator
        if val_acc == max(self.history_tracker['val_accuracy']):
            print(f"Best validation accuracy so far!")
        
        # Overfitting warning
        if acc - val_acc > 0.15:
            print(f" Warning: Possible overfitting detected (train-val gap: {(acc-val_acc)*100:.1f}%)")
        
        print(f"{'─'*80}")
    
    def on_train_end(self, logs=None):
        print(f"\n{'='*80}")
        print(f"Training {self.model_name} completed!")
        print(f"{'='*80}")
        
        # Final summary
        best_val_acc = max(self.history_tracker['val_accuracy'])
        best_epoch = self.history_tracker['val_accuracy'].index(best_val_acc) + 1
        
        print(f"\nFinal Summary:")
        print(f"   Best Validation Accuracy: {best_val_acc:.4f} ({best_val_acc*100:.2f}%)")
        print(f"   Best Epoch: {best_epoch}")
        print(f"   Final Training Accuracy: {self.history_tracker['accuracy'][-1]:.4f}")
        print(f"{'='*80}\n")
        
        # Save training plots
        if self.save_plots and self.plots_dir:
            self._save_training_plots()
    
    def _save_training_plots(self):
        """Save training history plots"""
        try:
            from utils import PlotUtils
            
            plot_path = os.path.join(self.plots_dir, f'{self.model_name}_training_history.png')
            PlotUtils.plot_training_history(self.history_tracker, save_path=plot_path)
        except Exception as e:
            print(f" Could not save training plots: {str(e)}")

class CustomProgressBar(tf.keras.callbacks.Callback):
    """
    Custom progress bar untuk menampilkan training progress dengan jelas
    """
    
    def __init__(self, epochs, verbose=1):
        super().__init__()
        self.epochs = epochs
        self.verbose = verbose
        self.current_epoch = 0
    
    def on_epoch_begin(self, epoch, logs=None):
        self.current_epoch = epoch + 1
        if self.verbose > 0:
            print(f"\n{'='*70}")
            print(f"Epoch {self.current_epoch}/{self.epochs}")
            print(f"{'='*70}")
    
    def on_batch_end(self, batch, logs=None):
        if self.verbose > 0 and batch % 10 == 0:
            loss = logs.get('loss', 0)
            acc = logs.get('accuracy', 0)
            sys.stdout.write(f"\rBatch {batch}: loss={loss:.4f}, acc={acc:.4f}")
            sys.stdout.flush()
    
    def on_epoch_end(self, epoch, logs=None):
        if self.verbose > 0:
            val_loss = logs.get('val_loss', 0)
            val_acc = logs.get('val_accuracy', 0)
            train_loss = logs.get('loss', 0)
            train_acc = logs.get('accuracy', 0)
            
            print(f"\n{'-'*70}")
            print(f"EPOCH {self.current_epoch} SUMMARY:")
            print(f"  Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f}")
            print(f"  Val Loss:   {val_loss:.4f} | Val Acc:   {val_acc:.4f}")
            print(f"{'-'*70}")

class DataSplitter:
    """
    Class untuk splitting data dengan stratification
    FIXED: 70% train, 15% validation, 15% test (sesuai jurnal)
    """
    
    def __init__(self, train_size=0.70, val_size=0.15, test_size=0.15, random_state=42):
        """
        Initialize data splitter with exact ratios
        
        Args:
            train_size (float): Training set size (default: 0.70 = 70%)
            val_size (float): Validation set size (default: 0.15 = 15%)
            test_size (float): Test set size (default: 0.15 = 15%)
            random_state (int): Random state for reproducibility
        """
        # Validate that sizes sum to 1.0
        total = train_size + val_size + test_size
        if not np.isclose(total, 1.0):
            raise ValueError(f"Split sizes must sum to 1.0, got {total}")
        
        self.train_size = train_size
        self.val_size = val_size
        self.test_size = test_size
        self.random_state = random_state
    
    def apply_smote(self, X, y, k_neighbors=3, sampling_strategy='auto'):
        """
        Apply SMOTE oversampling to balance minority classes
        
        Args:
            X (array): Features (raw signals or spectrograms)
            y (array): Labels
            k_neighbors (int): Number of nearest neighbors for SMOTE (default: 3)
            sampling_strategy (str or dict): Sampling strategy
                - 'auto': balance all classes to majority
                - dict: {class: n_samples} for custom balancing
        
        Returns:
            tuple: (X_resampled, y_resampled)
        """
        print(f"\n{'='*70}")
        print("SMOTE OVERSAMPLING")
        print(f"{'='*70}")
        
        # Original distribution
        unique, counts = np.unique(y, return_counts=True)
        print(f"\nOriginal class distribution:")
        for cls, count in zip(unique, counts):
            print(f"  Class {cls}: {count} samples ({count/len(y)*100:.1f}%)")
        
        # Flatten features if needed for SMOTE (it expects 2D)
        original_shape = X[0].shape if hasattr(X[0], 'shape') else None
        if original_shape and len(original_shape) > 1:
            # Flatten spectrograms/images to 1D for SMOTE
            X_flat = np.array([x.flatten() for x in X])
        else:
            X_flat = np.array(X)
        
        # Apply SMOTE
        try:
            smote = SMOTE(random_state=self.random_state, k_neighbors=k_neighbors, sampling_strategy=sampling_strategy)
            X_resampled_flat, y_resampled = smote.fit_resample(X_flat, y)
            
            # Reshape back if needed
            if original_shape and len(original_shape) > 1:
                X_resampled = X_resampled_flat.reshape(-1, *original_shape)
            else:
                X_resampled = X_resampled_flat
            
            # New distribution
            unique, counts = np.unique(y_resampled, return_counts=True)
            print(f"\nAfter SMOTE class distribution:")
            for cls, count in zip(unique, counts):
                print(f"  Class {cls}: {count} samples ({count/len(y_resampled)*100:.1f}%)")
            
            print(f"\nTotal samples: {len(y)} → {len(y_resampled)} (+{len(y_resampled)-len(y)} synthetic)")
            print(f"{'='*70}\n")
            
            return X_resampled, y_resampled
            
        except Exception as e:
            print(f"\n⚠ SMOTE failed: {e}")
            print("Continuing with original data...\n")
            return X, y
    
    def split_data(self, X, y, stratify=True, apply_smote=False, smote_k_neighbors=3):
        """
        Split data into train (70%), validation (15%), and test (15%) sets
        with optional SMOTE oversampling on training data only
        
        Args:
            X (array): Features
            y (array): Labels
            stratify (bool): Whether to stratify splits
            apply_smote (bool): Whether to apply SMOTE on training data
            smote_k_neighbors (int): K neighbors for SMOTE
            
        Returns:
            tuple: (X_train, X_val, X_test, y_train, y_val, y_test)
        """
        if stratify:
            stratify_param = y
        else:
            stratify_param = None
        
        # First split: separate test set (15%)
        X_temp, X_test, y_temp, y_test = train_test_split(
            X, y, 
            test_size=self.test_size,  # 15%
            random_state=self.random_state,
            stratify=stratify_param
        )
        
        val_size_adjusted = self.val_size / (self.train_size + self.val_size)
        
        if stratify:
            stratify_param = y_temp
        else:
            stratify_param = None
        
        X_train, X_val, y_train, y_val = train_test_split(
            X_temp, y_temp,
            test_size=val_size_adjusted,
            random_state=self.random_state,
            stratify=stratify_param
        )
        
        # Apply SMOTE to training data only (not val/test!)
        if apply_smote:
            X_train, y_train = self.apply_smote(X_train, y_train, k_neighbors=smote_k_neighbors)
        
        # Verify actual split ratios
        total_samples = len(y)
        actual_train_pct = len(y_train) / total_samples
        actual_val_pct = len(y_val) / total_samples
        actual_test_pct = len(y_test) / total_samples
        
        print(f"\nData Split Verification:")
        print(f"  Target: Train={self.train_size*100:.0f}%, Val={self.val_size*100:.0f}%, Test={self.test_size*100:.0f}%")
        print(f"  Actual: Train={actual_train_pct*100:.1f}%, Val={actual_val_pct*100:.1f}%, Test={actual_test_pct*100:.1f}%")
        
        return X_train, X_val, X_test, y_train, y_val, y_test
    
    def get_split_info(self, y_train, y_val, y_test):
        """Get information about data splits"""
        total_samples = len(y_train) + len(y_val) + len(y_test)
        
        info = {
            'total_samples': total_samples,
            'train_samples': len(y_train),
            'val_samples': len(y_val),
            'test_samples': len(y_test),
            'train_percentage': len(y_train) / total_samples * 100,
            'val_percentage': len(y_val) / total_samples * 100,
            'test_percentage': len(y_test) / total_samples * 100,
            'unique_classes': len(np.unique(np.concatenate([y_train, y_val, y_test]))),
            'class_distribution_train': np.bincount(y_train),
            'class_distribution_val': np.bincount(y_val),
            'class_distribution_test': np.bincount(y_test)
        }
        
        return info


class ModelTrainer:
    """Class untuk training deep learning models - UPDATED untuk 3 models"""
    
    def __init__(self, model_type='standard_cnn', architecture='standard', **model_params):
        self.model_type = model_type
        self.architecture = architecture
        self.model_params = model_params
        self.model = None
        self.history = None
        self.class_weights = None
    
    def prepare_input_shape(self, X):
        """Prepare input shape based on model type - UPDATED"""
        if self.model_type == 'standard_cnn':
            if len(X.shape) == 3:
                return X.shape[1:] + (1,)
            elif len(X.shape) == 4:
                return X.shape[1:]
            else:
                raise ValueError(f"Invalid input shape for CNN: {X.shape}")
        
        elif self.model_type == 'standard_lstm':
            if len(X.shape) == 3:
                return (X.shape[1] * X.shape[2],)
            elif len(X.shape) == 4:
                return (X.shape[1] * X.shape[2] * X.shape[3],)
            else:
                raise ValueError(f"Invalid input shape for LSTM: {X.shape}")
        
        elif self.model_type == 'standard_cnn_lstm':
            if len(X.shape) == 3:
                return X.shape[1:] + (1,)
            elif len(X.shape) == 4:
                return X.shape[1:]
            else:
                raise ValueError(f"Invalid input shape for CNN-LSTM: {X.shape}")
    
    def reshape_input_data(self, X):
        """Reshape input data based on model type - UPDATED"""
        if self.model_type == 'standard_cnn':
            if len(X.shape) == 3:
                X = np.expand_dims(X, axis=-1)
        
        elif self.model_type == 'standard_lstm':
            if len(X.shape) == 3:
                X = X.reshape(X.shape[0], -1)
            elif len(X.shape) == 4:
                X = X.reshape(X.shape[0], -1)
        
        elif self.model_type == 'standard_cnn_lstm':
            if len(X.shape) == 3:
                X = np.expand_dims(X, axis=-1)
        
        return X
    
    def build_model(self, input_shape, num_classes):
        """Build model based on specified type - FIXED: Remove conflicting params"""
        from models import ModelBuilder  
        

        clean_params = {k: v for k, v in self.model_params.items() 
                       if k not in ['input_shape', 'num_classes']}
        
        if self.model_type == 'standard_cnn':
            self.model = ModelBuilder.build_standard_cnn(
                input_shape, num_classes, **clean_params
            )
        
        elif self.model_type == 'standard_lstm':
            self.model = ModelBuilder.build_standard_lstm(
                input_shape, num_classes, **clean_params
            )
        
        elif self.model_type == 'standard_cnn_lstm':
            self.model = ModelBuilder.build_standard_cnn_lstm(
                input_shape, num_classes, **clean_params
            )
        
        else:
            raise ValueError(f"Unknown model type: {self.model_type}")
        
        return self.model
    
    def compute_class_weights(self, y):
        """Compute class weights untuk imbalanced data"""
        unique_classes = np.unique(y)
        weights = compute_class_weight('balanced', classes=unique_classes, y=y)
        # Multiply by 1.2 for moderate emphasis on minority classes
        weights = weights * 1.2
        self.class_weights = dict(zip(unique_classes, weights))
        
        print(f"\nClass Weights:")
        for cls, weight in self.class_weights.items():
            print(f"   Class {cls}: {weight:.4f}")
        
        return self.class_weights
    
    def train_model(self, X_train, y_train, X_val=None, y_val=None,
               epochs=30, batch_size=3, use_class_weights=True,  
               verbose=1, early_stopping_patience=10, learning_rate=0.005,  
               optimizer='adam', save_plots=True, plots_dir=None, 
               use_focal_loss=True):
        """Train model dengan enhanced configuration + Focal Loss\"\"\"
        
        # Reshape data
        X_train_reshaped = self.reshape_input_data(X_train)
        X_val_reshaped = self.reshape_input_data(X_val) if X_val is not None else None
        
        # Build model
        input_shape = self.prepare_input_shape(X_train)
        num_classes = len(np.unique(y_train))
        
        self.build_model(input_shape, num_classes)
        
        # Calculate class counts for Focal Loss
        unique, counts = np.unique(y_train, return_counts=True)
        class_counts = dict(zip(unique, counts))
        
        # Compile with Focal Loss if enabled
        if use_focal_loss:
            print(f\"\\n\u2713 Enabling Focal Loss (gamma=2.0) for imbalanced classification\")
            self.model.compile_model(
                learning_rate=learning_rate,
                optimizer_type=optimizer,
                use_focal_loss=True,
                class_counts=class_counts
            )
        else:
            # Standard compilation
            if optimizer == 'sgd':
                opt = tf.keras.optimizers.SGD(learning_rate=learning_rate, momentum=0.9, nesterov=True)
            else:
                opt = tf.keras.optimizers.Adam(learning_rate=learning_rate, beta_1=0.9, beta_2=0.999)
            
            opt = tf.keras.optimizers.get({
                'class_name': opt.__class__.__name__,
                'config': {**opt.get_config(), 'clipnorm': 1.0}
            })
            self.model.model.compile(
                optimizer=opt,
                loss='sparse_categorical_crossentropy',
                metrics=['accuracy']
            )
        
        print(f"[DEBUG] Model compiled successfully")
        print(f"[DEBUG] Model summary:")
        self.model.model.summary()

        # Compute class weights dengan balanced strategy
        class_weight = None
        if use_class_weights:
            class_weight = self.compute_class_weights(y_train)
        
        # Prepare enhanced callbacks
        callbacks = [
            # Enhanced training callback
            EnhancedTrainingCallback(
                epochs=epochs,
                model_name=self.model_type,
                save_plots=save_plots,
                plots_dir=plots_dir
            ),
            
            # Early stopping dengan lebih banyak patience
            tf.keras.callbacks.EarlyStopping(
                monitor='val_accuracy',
                patience=early_stopping_patience,
                restore_best_weights=True,
                verbose=1,
                mode='max'
            ),
            
            # Reduce LR dengan factor yang lebih agresif
            tf.keras.callbacks.ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=5,
                min_lr=1e-5,
                verbose=1,
                mode='min'
            ),
            tf.keras.callbacks.CSVLogger(
                f'{self.model_type}_training_log.csv',
                separator=',',
                append=False
        ),
            # Model checkpoint
            tf.keras.callbacks.ModelCheckpoint(
                f'best_{self.model_type}.h5',
                monitor='val_accuracy',
                save_best_only=True,
                mode='max',
                verbose=1
            ),
            
            # TensorBoard untuk monitoring
            tf.keras.callbacks.TensorBoard(
                log_dir=f'logs/{self.model_type}',
                histogram_freq=1,
                write_graph=True,
                write_images=True
            ) if plots_dir else None
        ]
        if save_plots:
            callbacks.append(
                tf.keras.callbacks.ModelCheckpoint(
                    f'best_{self.model_type}_journal.h5',
                    monitor='val_accuracy',
                    save_best_only=True,
                    mode='max',
                    verbose=1
                )
            )
        # Remove None callbacks
        callbacks = [cb for cb in callbacks if cb is not None]
        
        # Train model dengan validation
        validation_data = (X_val_reshaped, y_val) if X_val is not None else None
        print(f"\n[DEBUG] Starting training...")
        print(f"  Training samples: {len(X_train_reshaped)}")
        print(f"  Validation samples: {len(X_val_reshaped) if X_val is not None else 0}")
        print(f"  Batch size: {batch_size}")
        print(f"  Epochs: {epochs}")
        # Calculate steps per epoch untuk better progress tracking
        steps_per_epoch = None
        if len(X_train) > batch_size:
            steps_per_epoch = max(1, len(X_train) // batch_size)
        
        print(f"\nTraining Configuration:")
        print(f"  Model: {self.model_type}")
        print(f"  Input shape: {input_shape}")
        print(f"  Classes: {num_classes}")
        print(f"  Learning rate: {learning_rate}")
        print(f"  Batch size: {batch_size}")
        print(f"  Epochs: {epochs}")
        print(f"  Training samples: {len(X_train)}")
        if validation_data:
            print(f"  Validation samples: {len(X_val)}")
        if class_weight:
            print(f"  Using class weights")
        
        try:
            self.history = self.model.model.fit(
                X_train_reshaped, y_train,
                batch_size=batch_size,
                epochs=epochs,
                validation_data=validation_data,
                callbacks=callbacks,
                class_weight=class_weight,
                verbose=1,
                shuffle=True
            )
            
            print(f"\n[DEBUG] Training completed successfully")
            return self.history
        
        except Exception as e:
            print(f"\n[ERROR] Training failed: {str(e)}")
            print(f"  Trying with batch size 8 as fallback...")
            try:
                self.history = self.model.model.fit(
                    X_train_reshaped, y_train,
                    batch_size=8,  # Fallback batch size
                    epochs=epochs,
                    validation_data=validation_data,
                    callbacks=callbacks,
                    class_weight=class_weight,
                    verbose=1,
                    shuffle=True
                )
                print(f"  Fallback training successful")
                return self.history
            except Exception as e2:
                print(f"  Fallback also failed: {str(e2)}")
                import traceback
                traceback.print_exc()
                raise
            
    def save_model(self, filepath):
        """Save trained model"""
        if self.model and hasattr(self.model, 'model'):
            os.makedirs(os.path.dirname(filepath), exist_ok=True)
            self.model.model.save(filepath)
            print(f"Model saved: {filepath}")
            
            # Save metadata
            metadata_path = filepath.replace('.h5', '_metadata.pkl')
            metadata = {
                'model_type': self.model_type,
                'architecture': self.architecture,
                'model_params': self.model_params,
                'class_weights': self.class_weights
            }
            with open(metadata_path, 'wb') as f:
                pickle.dump(metadata, f)


def train_multiple_models(X, y, class_names, models_config,
                         save_dir='trained_models', use_smote=True, use_focal_loss=True, **train_params):
    """Train multiple models with progress tracking, SMOTE oversampling, and Focal Loss"""
    from utils import ProgressBar

    print("\n" + "="*80)
    print("DATA VALIDATION BEFORE TRAINING")
    print("="*80)
    os.makedirs(save_dir, exist_ok=True)

    print(f"X shape: {X.shape}")
    print(f"y shape: {y.shape}")

    if len(X.shape) >= 3:
        spec_size = X.shape[1:3]
        if spec_size != (64, 64):
            print(f"WARNING: Spectrogram size {spec_size} != (64, 64)")
        else:
            print(f"Spectrogram size correct: 64x64")

    unique_labels, counts = np.unique(y, return_counts=True)
    print(f"\nClass distribution:")
    for label, count in zip(unique_labels, counts):
        class_name = class_names[label] if label < len(class_names) else f"Class_{label}"
        percentage = (count / len(y)) * 100
        print(f"  {class_name}: {count} samples ({percentage:.1f}%)")

    if np.any(np.isnan(X)):
        print("WARNING: X contains NaN values")
        X = np.nan_to_num(X)
    
    if np.any(np.isinf(X)):
        print(" WARNING: X contains Inf values")
        X = np.nan_to_num(X)
    
    print(f"\nData validation completed")
    print("="*80)

    # Validate data
    if not isinstance(y, np.ndarray):
        y = np.array(y)
    
    if len(X) != len(y):
        raise ValueError(f"X and y length mismatch: {len(X)} vs {len(y)}")
    
    # Verify spectrograms
    if len(X.shape) >= 3:
        if X.shape[1:3] != (64, 64):
            raise ValueError(f"Expected 64x64 spectrograms, got {X.shape[1:3]}")
    
    # Split data with SMOTE oversampling
    splitter = DataSplitter(train_size=0.70, val_size=0.15, test_size=0.15)
    X_train, X_val, X_test, y_train, y_val, y_test = splitter.split_data(
        X, y, 
        apply_smote=use_smote, 
        smote_k_neighbors=3
    )
    
    # Calculate class counts for Focal Loss (from ORIGINAL y_train before any weighting)
    unique, counts = np.unique(y_train, return_counts=True)
    class_counts = dict(zip(unique, counts))
    print(f"\nClass counts for Focal Loss: {class_counts}")
    
    # Store split info
    split_data = {
        'X_train': X_train, 'y_train': y_train,
        'X_val': X_val, 'y_val': y_val,
        'X_test': X_test, 'y_test': y_test,
        'class_names': class_names
    }
    
    results = {}
    n_models = len(models_config)
    
    print(f"\n{'='*80}")
    print(f"TRAINING {n_models} MODEL(S)")
    print(f"{'='*80}")
    
    plots_dir = os.path.join(save_dir, '..', 'plots')
    os.makedirs(plots_dir, exist_ok=True)
    
    for i, config in enumerate(models_config):
        model_name = config['name']
        print(f"\n{'='*80}")
        print(f"Model {i+1}/{n_models}: {model_name.upper()}")
        print(f"{'='*80}")
        
        trainer = ModelTrainer(
            model_type=config['type'],
            architecture=config['architecture'],
            **config.get('params', {})
        )
        
        try:
            history = trainer.train_model(
                X_train, y_train, X_val, y_val,
                save_plots=True,
                plots_dir=plots_dir,
                **train_params
            )
            
            # Save model
            model_path = os.path.join(save_dir, f"{config['name']}.h5")
            trainer.save_model(model_path)
            
            results[config['name']] = {
                'trainer': trainer,
                'history': history,
                'config': config,
                'model_path': model_path
            }
            
            print(f"\n{config['name']} training completed successfully!")
            
        except Exception as e:
            print(f"\n{config['name']} training failed: {str(e)}")
            results[config['name']] = {'error': str(e), 'config': config}
    
    # Save split data
    split_data_path = os.path.join(save_dir, 'split_data.pkl')
    with open(split_data_path, 'wb') as f:
        pickle.dump(split_data, f)
    
    print(f"\nSplit data saved: {split_data_path}")
    
    return results, split_data


def cross_validate_models(X, y, class_names, models_config,
                         cv_epochs=50, save_dir='cv_results', **cv_params):
    """Cross validation wrapper - simplified for now"""
    print("\n]Cross validation is optional and takes significant time.")
    print("Skipping for now - focus on main training.")
    
    return {}, None
    
class CrossValidator:
    """
    Enhanced Cross Validator with comprehensive data validation
    FIXED: Proper data handling and validation
    """
    
    def __init__(self, n_splits=5, random_state=42):
        self.n_splits = n_splits
        self.random_state = random_state
        self.skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
        self.all_cv_results = {}
    
    def cross_validate_model(self, X, y, model_config, class_names, 
                       epochs=50, batch_size=32, verbose=0, save_dir=None):
        """Cross validation with progress tracking"""
        from utils import ProgressBar
        
        if save_dir is None:
            save_dir = f"cv_results_{model_config['name']}"
        
        os.makedirs(save_dir, exist_ok=True)
        
        if not isinstance(y, np.ndarray):
            y = np.array(y)
        
        if len(X) != len(y):
            raise ValueError(f"X and y length mismatch: {len(X)} vs {len(y)}")
        
        if len(X.shape) >= 3:
            if X.shape[1:3] != (256, 256):
                raise ValueError(f"Expected 256x256 spectrograms, got {X.shape[1:3]}")
        
        fold_results = []
        fold_accuracies = []
        fold_predictions = []
        fold_true_labels = []
        
        print(f"\n 5-Fold CV for {model_config['name']}")
        progress_bar = ProgressBar(5, desc=f"  {model_config['name']}", unit="fold")
        
        for fold_idx, (train_idx, val_idx) in enumerate(self.skf.split(X, y)):
            fold_num = fold_idx + 1
            
            try:
                X_train_fold = X[train_idx]
                y_train_fold = y[train_idx]
                X_val_fold = X[val_idx]
                y_val_fold = y[val_idx]
                
                trainer = ModelTrainer(
                    model_type=model_config['type'],
                    architecture=model_config['architecture'],
                    **model_config.get('params', {})
                )
                
                history = trainer.train_model(
                    X_train_fold, y_train_fold,
                    X_val_fold, y_val_fold,
                    epochs=epochs,
                    batch_size=batch_size,
                    verbose=0
                )
                
                X_val_reshaped = trainer.reshape_input_data(X_val_fold)
                y_pred = trainer.model.model.predict(X_val_reshaped, verbose=0)
                y_pred_classes = np.argmax(y_pred, axis=1)
                
                fold_accuracy = accuracy_score(y_val_fold, y_pred_classes)
                fold_accuracies.append(fold_accuracy)
                fold_predictions.extend(y_pred_classes)
                fold_true_labels.extend(y_val_fold)
                
                fold_result = {
                    'fold': fold_num,
                    'train_samples': len(X_train_fold),
                    'val_samples': len(X_val_fold),
                    'accuracy': fold_accuracy
                }
                fold_results.append(fold_result)
                
                # Save model
                if hasattr(trainer.model, 'model'):
                    fold_model_path = os.path.join(save_dir, f"{model_config['name']}_fold_{fold_num}.h5")
                    trainer.model.model.save(fold_model_path)
                
            except Exception as e:
                print(f"\nFold {fold_num} error: {str(e)}")
                fold_result = {
                    'fold': fold_num,
                    'error': str(e),
                    'accuracy': 0.0
                }
                fold_results.append(fold_result)
            
            progress_bar.update(1)
        
        progress_bar.close()
        
        cv_results = self._calculate_cv_metrics(
            fold_results, fold_accuracies, fold_predictions,
            fold_true_labels, class_names, model_config
        )
        
        self.all_cv_results[model_config['name']] = cv_results
        
        # Save results
        results_path = os.path.join(save_dir, f"{model_config['name']}_cv_results.pkl")
        with open(results_path, 'wb') as f:
            pickle.dump(cv_results, f)
        
        return cv_results


    
    def _calculate_cv_metrics(self, fold_results, fold_accuracies, 
                            fold_predictions, fold_true_labels, 
                            class_names, model_config):
        """Calculate cross validation metrics"""
        valid_accuracies = [acc for acc in fold_accuracies if acc is not None]
        
        if len(valid_accuracies) == 0:
            return {
                'model_config': model_config,
                'fold_results': fold_results,
                'mean_accuracy': 0,
                'std_accuracy': 0,
                'error': 'All folds failed'
            }
        
        mean_accuracy = np.mean(valid_accuracies)
        std_accuracy = np.std(valid_accuracies)
        
        overall_classification_report = classification_report(
            fold_true_labels, fold_predictions,
            target_names=class_names,
            output_dict=True,
            zero_division=0
        )
        
        overall_confusion_matrix = confusion_matrix(
            fold_true_labels, fold_predictions
        )
        
        cv_results = {
            'model_config': model_config,
            'fold_results': fold_results,
            'mean_accuracy': mean_accuracy,
            'std_accuracy': std_accuracy,
            'fold_accuracies': valid_accuracies,
            'overall_classification_report': overall_classification_report,
            'overall_confusion_matrix': overall_confusion_matrix
        }
        
        return cv_results
    
    def display_final_cv_results(self):
        """Display comprehensive cross validation results"""
        print("\n" + "="*80)
        print("FINAL CROSS VALIDATION RESULTS (5-FOLD)")
        print("="*80)
        
        if not self.all_cv_results:
            print("No cross validation results available.")
            return
        
        summary_data = []
        for model_name, results in self.all_cv_results.items():
            if 'error' not in results:
                summary_data.append({
                    'Model': model_name,
                    'Mean_Accuracy': results['mean_accuracy'],
                    'Std_Accuracy': results['std_accuracy']
                })
        
        summary_data.sort(key=lambda x: x['Mean_Accuracy'], reverse=True)
        
        print("\nMODEL PERFORMANCE RANKING:")
        print("-" * 80)
        print(f"{'Rank':<4} {'Model':<25} {'Mean±Std':<15}")
        print("-" * 80)
        
        for i, data in enumerate(summary_data):
            mean_std = f"{data['Mean_Accuracy']:.4f}±{data['Std_Accuracy']:.4f}"
            print(f"{i+1:<4} {data['Model']:<25} {mean_std:<15}")
        
        print("\n" + "="*80)


def validate_pipeline_data(X, y, stage_name=""):
    """
    Comprehensive validation function to use at any stage
    
    Usage:
        validate_pipeline_data(X, y, "Before Training")
        validate_pipeline_data(X, y, "Before Cross Validation")
    """
    print(f"\n{'='*70}")
    print(f"DATA VALIDATION: {stage_name}")
    print(f"{'='*70}")
    
    # Check X
    print(f"X Information:")
    print(f"  Type: {type(X)}")
    print(f"  Shape: {X.shape}")
    print(f"  Dtype: {X.dtype}")
    print(f"  Min: {np.min(X):.4f}, Max: {np.max(X):.4f}")
    print(f"  Contains NaN: {np.any(np.isnan(X))}")
    print(f"  Contains Inf: {np.any(np.isinf(X))}")
    
    # Check y
    print(f"\ny Information:")
    print(f"  Type: {type(y)}")
    if hasattr(y, 'shape'):
        print(f"  Shape: {y.shape}")
    print(f"  Length: {len(y)}")
    if isinstance(y, np.ndarray):
        print(f"  Dtype: {y.dtype}")
        print(f"  Unique values: {np.unique(y)}")
        print(f"  Value counts: {np.bincount(y)}")
    
    # Validation checks
    issues = []
    
    # Check 1: Same length
    if len(X) != len(y):
        issues.append(f" Length mismatch: X={len(X)}, y={len(y)}")
    else:
        print(f"\nLength match: {len(X)} samples")
    
    # Check 2: Spectrogram size
    if len(X.shape) >= 3:
        spec_size = X.shape[1:3]
        if spec_size != (256, 256):
            issues.append(f" Wrong spectrogram size: {spec_size}, expected (256, 256)")
        else:
            print(f"Correct spectrogram size: 256x256")
    
    # Check 3: Data quality
    if np.any(np.isnan(X)):
        issues.append(" X contains NaN values")
    else:
        print(f"No NaN values in X")
    
    if np.any(np.isinf(X)):
        issues.append(" X contains Inf values")
    else:
        print(f"No Inf values in X")
    
    # Check 4: y is numpy array
    if not isinstance(y, np.ndarray):
        issues.append(f"  y is {type(y)}, should be numpy array")
    else:
        print(f"y is numpy array")
    
    # Report issues
    if issues:
        print(f"\n{'='*70}")
        print("VALIDATION ISSUES FOUND:")
        for issue in issues:
            print(f"  {issue}")
        print(f"{'='*70}")
        return False
    else:
        print(f"\n{'='*70}")
        print("ALL VALIDATIONS PASSED!")
        print(f"{'='*70}")
        return True


def cross_validate_models(X, y, class_names, models_config, 
                         cv_epochs=50, save_dir='cv_results', **cv_params):
    """
    Cross validation function
    FIXED: Added comprehensive validation
    """
    os.makedirs(save_dir, exist_ok=True)
    
    # === FIX: Validate input data ===
    print("\n" + "="*80)
    print("CROSS VALIDATION DATA VALIDATION")
    print("="*80)
    print(f"X shape: {X.shape}")
    print(f"y shape: {y.shape if hasattr(y, 'shape') else len(y)}")
    
    # Ensure y is numpy array
    if not isinstance(y, np.ndarray):
        print(f"Converting y to numpy array")
        y = np.array(y)
    
    # Verify 256x256 spectrograms
    if len(X.shape) >= 3:
        spec_size = X.shape[1:3]
        if spec_size != (256, 256):
            raise ValueError(
                f" ERROR: Expected 256x256 spectrograms, got {spec_size}\n"
                f"   Full X shape: {X.shape}\n"
                f"   Please fix feature extraction to output correct size!"
            )
    
    # Verify length match
    if len(X) != len(y):
        raise ValueError(f" X and y length mismatch: {len(X)} vs {len(y)}")
    
    print(f"Input validation passed")
    print(f"Ready for cross validation with {len(X)} samples")
    print("="*80)
    
    cv = CrossValidator(n_splits=5, random_state=42)
    
    print("\n" + "="*80)
    print("STARTING 5-FOLD CROSS VALIDATION")
    print("="*80)
    
    for config in models_config:
        try:
            print(f"\nProcessing: {config['name']}")
            cv_results = cv.cross_validate_model(
                X, y, config, class_names, 
                epochs=cv_epochs, 
                save_dir=save_dir, 
                **cv_params
            )
        except Exception as e:
            print(f"✗ Error in cross validation for {config['name']}: {str(e)}")
            import traceback
            traceback.print_exc()
            cv.all_cv_results[config['name']] = {'error': str(e)}
    
    cv.display_final_cv_results()
    
    return cv.all_cv_results, cv


def train_multiple_models(X, y, class_names, models_config, 
                         save_dir='trained_models', **train_params):
    """Train multiple models with progress tracking"""
    from utils import ProgressBar
    
    os.makedirs(save_dir, exist_ok=True)
    
    # Validate data
    if not isinstance(y, np.ndarray):
        y = np.array(y)
    
    if len(X) != len(y):
        raise ValueError(f"X and y length mismatch: {len(X)} vs {len(y)}")
    
    # Verify spectrograms
    if len(X.shape) >= 3:
        if X.shape[1:3] != (256, 256):
            raise ValueError(f"Expected 256x256 spectrograms, got {X.shape[1:3]}")
    
    # Split data
    splitter = DataSplitter(train_size=0.70, val_size=0.15, test_size=0.15)
    X_train, X_val, X_test, y_train, y_val, y_test = splitter.split_data(X, y)
    
    split_info = splitter.get_split_info(y_train, y_val, y_test)
    
    print(f"\n Data Split (70-15-15):")
    print(f"   Train: {split_info['train_samples']} ({split_info['train_percentage']:.1f}%)")
    print(f"   Val:   {split_info['val_samples']} ({split_info['val_percentage']:.1f}%)")
    print(f"   Test:  {split_info['test_samples']} ({split_info['test_percentage']:.1f}%)")
    
    results = {}
    n_models = len(models_config)
    
    progress_bar = ProgressBar(n_models, desc="Training models", unit="model")
    
    for i, config in enumerate(models_config):
        model_name = config['name']
        print(f"\n Training {i+1}/{n_models}: {model_name}")
        
        trainer = ModelTrainer(
            model_type=config['type'],
            architecture=config['architecture'],
            **config.get('params', {})
        )
        
        try:
            history = trainer.train_model(
                X_train, y_train, X_val, y_val, **train_params
            )
            
            # Save model
            model_path = os.path.join(save_dir, f"{config['name']}.h5")
            trainer.save_model(model_path)
            
            results[config['name']] = {
                'trainer': trainer,
                'history': history,
                'config': config,
                'model_path': model_path
            }
            
            print(f"    {config['name']} completed")
            
        except Exception as e:
            print(f"    {config['name']} failed: {str(e)}")
            results[config['name']] = {'error': str(e), 'config': config}
        
        progress_bar.update(1)
    
    progress_bar.close()
    
    split_data = {
        'X_train': X_train, 'y_train': y_train,
        'X_val': X_val, 'y_val': y_val,
        'X_test': X_test, 'y_test': y_test,
        'class_names': class_names,
        'split_info': split_info
    }
    
    split_data_path = os.path.join(save_dir, 'split_data.pkl')
    with open(split_data_path, 'wb') as f:
        pickle.dump(split_data, f)
    
    print(f"\nSplit data saved: {split_data_path}")
    
    return results, split_data


# ===========================================================================
# HELPER FUNCTIONS
# ===========================================================================

def evaluate_trained_models(results, split_data, save_dir='evaluation_results'):
    """
    Evaluate all trained models on test set
    
    Args:
        results: Dictionary of trained models from train_multiple_models()
        split_data: Split data dictionary containing test set
        save_dir: Directory to save evaluation results
    """
    os.makedirs(save_dir, exist_ok=True)
    
    X_test = split_data['X_test']
    y_test = split_data['y_test']
    class_names = split_data['class_names']
    
    print("\n" + "="*70)
    print("EVALUATING MODELS ON TEST SET")
    print("="*70)
    
    evaluation_results = {}
    
    for model_name, model_result in results.items():
        if 'error' in model_result:
            print(f"\n{model_name}: Skipped (training error)")
            continue
        
        print(f"\nEvaluating: {model_name}")
        
        try:
            trainer = model_result['trainer']
            
            # Reshape test data
            X_test_reshaped = trainer.reshape_input_data(X_test)
            
            # Predictions
            y_pred = trainer.model.model.predict(X_test_reshaped, verbose=0)
            y_pred_classes = np.argmax(y_pred, axis=1)
            
            # Calculate metrics
            accuracy = accuracy_score(y_test, y_pred_classes)
            cm = confusion_matrix(y_test, y_pred_classes)
            report = classification_report(y_test, y_pred_classes, 
                                          target_names=class_names,
                                          output_dict=True,
                                          zero_division=0)
            
            evaluation_results[model_name] = {
                'accuracy': accuracy,
                'confusion_matrix': cm,
                'classification_report': report,
                'predictions': y_pred_classes,
                'probabilities': y_pred
            }
            
            print(f"  Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
            
        except Exception as e:
            print(f"  Error: {str(e)}")
            evaluation_results[model_name] = {'error': str(e)}
    
    # Save evaluation results
    eval_path = os.path.join(save_dir, 'evaluation_results.pkl')
    with open(eval_path, 'wb') as f:
        pickle.dump(evaluation_results, f)
    
    print(f"\nEvaluation results saved to: {eval_path}")
    
    # Display summary
    print("\n" + "="*70)
    print("TEST SET EVALUATION SUMMARY")
    print("="*70)
    print(f"{'Model':<30} {'Accuracy':<15}")
    print("-"*70)
    
    sorted_results = sorted(
        [(k, v) for k, v in evaluation_results.items() if 'error' not in v],
        key=lambda x: x[1]['accuracy'],
        reverse=True
    )
    
    for model_name, metrics in sorted_results:
        print(f"{model_name:<30} {metrics['accuracy']:.4f} ({metrics['accuracy']*100:.2f}%)")
    
    print("="*70)
    
    return evaluation_results


def compare_cv_and_test_results(cv_results, evaluation_results):
    """
    Compare cross-validation and test set results
    
    Args:
        cv_results: Results from cross_validate_models()
        evaluation_results: Results from evaluate_trained_models()
    """
    print("\n" + "="*80)
    print("CROSS-VALIDATION vs TEST SET COMPARISON")
    print("="*80)
    print(f"{'Model':<25} {'CV Mean±Std':<20} {'Test Accuracy':<15} {'Difference':<10}")
    print("-"*80)
    
    for model_name in cv_results.keys():
        if 'error' in cv_results[model_name]:
            continue
        
        cv_mean = cv_results[model_name]['mean_accuracy']
        cv_std = cv_results[model_name]['std_accuracy']
        
        if model_name in evaluation_results and 'error' not in evaluation_results[model_name]:
            test_acc = evaluation_results[model_name]['accuracy']
            diff = test_acc - cv_mean
            diff_str = f"{diff:+.4f}"
            
            print(f"{model_name:<25} {cv_mean:.4f}±{cv_std:.4f}      {test_acc:.4f}         {diff_str}")
        else:
            print(f"{model_name:<25} {cv_mean:.4f}±{cv_std:.4f}      N/A             N/A")
    
    print("="*80)
class DetailedProgressCallback(Callback):
    """
    Custom callback untuk menampilkan progress training yang jelas
    """
    
    def __init__(self, epochs, model_name="Model"):
        super().__init__()
        self.epochs = epochs
        self.model_name = model_name
        self.epoch_start_time = None
        
    def on_train_begin(self, logs=None):
        print(f"\n{'='*70}")
        print(f"Training {self.model_name}")
        print(f"{'='*70}")
    
    def on_epoch_begin(self, epoch, logs=None):
        self.epoch_start_time = time.time()
        print(f"\nEpoch {epoch+1}/{self.epochs}")
        print("-" * 70)
    
    def on_epoch_end(self, epoch, logs=None):
        elapsed = time.time() - self.epoch_start_time
        
        # Get metrics
        loss = logs.get('loss', 0)
        acc = logs.get('accuracy', 0)
        val_loss = logs.get('val_loss', 0)
        val_acc = logs.get('val_accuracy', 0)
        opt = self.model.optimizer
        if hasattr(opt, "lr"):
            lr = float(opt.lr.numpy())
        else:
            lr = float(opt.learning_rate.numpy())

        # Progress bar
        progress = (epoch + 1) / self.epochs
        bar_length = 40
        filled = int(progress * bar_length)
        bar = '█' * filled + '░' * (bar_length - filled)
        
        print(f"\n[{bar}] {progress*100:.1f}%")
        print(f"Loss: {loss:.4f} - Acc: {acc:.4f} | Val Loss: {val_loss:.4f} - Val Acc: {val_acc:.4f}")
        print(f"Time: {elapsed:.1f}s - LR: {lr:.2e}")
        
        # Best model indicator
        if val_acc == max([logs.get(f'val_accuracy', 0) for _ in range(epoch+1)]):
            print("Best model so far!")
    
    def on_train_end(self, logs=None):
        print(f"\n{'='*70}")
        print(f"Training {self.model_name} completed!")
        print(f"{'='*70}\n")


class BatchProgressCallback(Callback):
    """
    Callback untuk menampilkan progress per batch (untuk epoch yang lama)
    """
    
    def __init__(self, total_batches):
        super().__init__()
        self.total_batches = total_batches
        self.batch_count = 0
        
    def on_batch_end(self, batch, logs=None):
        self.batch_count += 1
        if self.batch_count % max(1, self.total_batches // 10) == 0:  # Update setiap 10%
            progress = self.batch_count / self.total_batches
            bar_length = 30
            filled = int(progress * bar_length)
            bar = '█' * filled + '░' * (bar_length - filled)
            
            loss = logs.get('loss', 0)
            acc = logs.get('accuracy', 0)
            
            print(f"\r  Batch [{bar}] {progress*100:.0f}% | Loss: {loss:.4f} Acc: {acc:.4f}", 
                  end='', flush=True)
        
        if self.batch_count == self.total_batches:
            print()  # Newline after epoch
            self.batch_count = 0


def get_enhanced_callbacks(model_name, epochs, batch_size, train_size, 
                          patience=25, factor=0.5, min_lr=1e-8):
    """
    Get callbacks dengan progress bar yang visible
    
    Args:
        model_name: Nama model
        epochs: Jumlah epochs
        batch_size: Batch size
        train_size: Ukuran training set
        patience: Early stopping patience
        factor: ReduceLR factor
        min_lr: Minimum learning rate
    """
    
    total_batches = train_size // batch_size
    
    callbacks = [
        # Custom progress callback
        DetailedProgressCallback(epochs, model_name),
        
        # Batch progress (untuk dataset besar)
        BatchProgressCallback(total_batches) if train_size > 1000 else None,
        
        # Early stopping
        tf.keras.callbacks.EarlyStopping(
            monitor='val_accuracy',
            patience=patience,
            restore_best_weights=True,
            verbose=1,  # PENTING: verbose=1
            mode='max'
        ),
        
        # Reduce LR
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=factor,
            patience=patience//3,
            min_lr=min_lr,
            verbose=1,  # PENTING: verbose=1
            cooldown=5
        ),
        
        # Model checkpoint
        tf.keras.callbacks.ModelCheckpoint(
            f'best_{model_name}.h5',
            monitor='val_accuracy',
            save_best_only=True,
            save_weights_only=False,
            mode='max',
            verbose=1  # PENTING: verbose=1
        )
    ]
    
    # Remove None callbacks
    callbacks = [cb for cb in callbacks if cb is not None]
    
    return callbacks