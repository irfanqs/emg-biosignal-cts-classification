import numpy as np
from scipy.signal import stft
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
import tensorflow as tf
from tensorflow.keras import layers, models
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')
class EMGHeaderParser:
    """
    Parser untuk extract metadata dari EMG file headers
    Support Motor (CVM) dan Sensory (CVS) formats
    """
    
    def __init__(self):
        self.metadata = {}
        self.signal_type_mapping = {
            'CVM': 'Motor',
            'CVS': 'Sensory'
        }
    
    def parse_header(self, file_path):
        """
        Parse header dari EMG file
        Extract: signal type, channels, sensitivity, filters, sampling info
        """
        metadata = {
            'file_path': file_path,
            'filename': os.path.basename(file_path),
            'signal_type': None,
            'test_name': None,
            'test_item': None,
            'patient_name': None,
            'patient_id': None,
            'test_date': None,
            'channels': [],
            'sensitivities': [],
            'hi_cut': None,
            'lo_cut': None,
            'ms_per_sample': None,
            'num_samples': None,
            'sampling_rate': None,
        }
        
        try:
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                lines = f.readlines()
            
            # Parse header lines (usually first 15-20 lines)
            for i, line in enumerate(lines[:50]):
                line = line.strip()
                
                if not line or line.startswith('%') or line.startswith('#'):
                    continue
                
                # === Extract metadata ===
                
                # Test Name
                if 'Test Name' in line or line.startswith('NCV'):
                    parts = line.split('\t')
                    if len(parts) > 1:
                        metadata['test_name'] = parts[1].strip()
                
                # Test Item
                if 'Test Item' in line:
                    parts = line.split('\t')
                    if len(parts) > 1:
                        test_item = parts[1].strip()
                        metadata['test_item'] = test_item
                        
                        # Determine signal type dari test item
                        if 'CVM' in test_item or 'Motor' in test_item:
                            metadata['signal_type'] = 'Motor'
                        elif 'CVS' in test_item or 'Sensory' in test_item:
                            metadata['signal_type'] = 'Sensory'
                
                # Patient info
                if 'Patient Name' in line:
                    parts = line.split('\t')
                    if len(parts) > 1:
                        metadata['patient_name'] = parts[1].strip()
                
                if 'Patient ID' in line:
                    parts = line.split('\t')
                    if len(parts) > 1:
                        metadata['patient_id'] = parts[1].strip()
                
                if 'Test Date' in line:
                    parts = line.split('\t')
                    if len(parts) > 1:
                        metadata['test_date'] = parts[1].strip()
                
                # === Parse Trace Label (Channels) ===
                if 'Trace Label' in line:
                    # Line format: "Trace Label     2nd :          3rd :          4th :          4th uln :"
                    channels = self._extract_channel_names(line)
                    metadata['channels'] = channels
                
                # === Parse Sensitivity (µV/Div) ===
                if 'Sensitivity' in line and 'µV' in line:
                    sensitivities = self._extract_sensitivities(line)
                    metadata['sensitivities'] = sensitivities
                
                # === Parse Hi-Cut (Hz) ===
                if 'Hicut' in line or 'Hi-Cut' in line or 'High Cut' in line:
                    hi_cut = self._extract_frequency(line)
                    if hi_cut:
                        metadata['hi_cut'] = hi_cut
                
                # === Parse Lo-Cut (Hz) ===
                if 'Locut' in line or 'Lo-Cut' in line or 'Low Cut' in line:
                    lo_cut = self._extract_frequency(line)
                    if lo_cut:
                        metadata['lo_cut'] = lo_cut
                
                # === Parse ms/Sample ===
                if 'ms/Sample' in line:
                    ms_per_sample = self._extract_float(line)
                    if ms_per_sample:
                        metadata['ms_per_sample'] = ms_per_sample
                        # Calculate sampling rate
                        metadata['sampling_rate'] = 1000.0 / ms_per_sample
                
                # === Parse Samples ===
                if 'Samples' in line and 'Trace Data' not in line:
                    num_samples = self._extract_int(line)
                    if num_samples:
                        metadata['num_samples'] = num_samples
        
        except Exception as e:
            print(f"Error parsing header: {str(e)}")
        
        return metadata
    
    def _extract_channel_names(self, line):
        """Extract channel names dari Trace Label line"""
        # Remove "Trace Label" prefix
        line = line.replace('Trace Label', '').strip()
        
        # Split by ':' dan clean up
        channels = []
        parts = line.split(':')
        for part in parts:
            part = part.strip()
            if part and part not in ['']:
                channels.append(part)
        
        return channels
    
    def _extract_sensitivities(self, line):
        """Extract sensitivity values (µV/Div)"""
        # Find all numbers followed by .00
        sensitivities = []
        
        # Pattern: number.00 followed by µV or mV
        pattern = r'(\d+(?:\.\d+)?)\s*(?:µV|mV|uV|mV)'
        matches = re.findall(pattern, line)
        
        for match in matches:
            try:
                sensitivities.append(float(match))
            except:
                pass
        
        return sensitivities
    
    def _extract_frequency(self, line):
        """Extract frequency value in Hz"""
        # Pattern: number followed by .00 and Hz or just number
        pattern = r'(\d+(?:\.\d+)?)\s*(?:Hz|hz)?'
        matches = re.findall(pattern, line)
        
        if matches:
            try:
                return float(matches[-1]) 
            except:
                pass
        
        return None
    
    def _extract_float(self, line):
        """Extract float value"""
        pattern = r'(\d+\.\d+)'
        matches = re.findall(pattern, line)
        
        if matches:
            try:
                return float(matches[-1])
            except:
                pass
        
        return None
    
    def _extract_int(self, line):
        """Extract integer value"""
        pattern = r'(\d+)(?:\.\d+)?'
        matches = re.findall(pattern, line)
        
        if matches:
            try:
                return int(matches[-1])
            except:
                pass
        
        return None

class SpectrogramExtractor:
    """
    Class untuk ekstraksi fitur spektogram dari EMG signals
    menggunakan Short-Time Fourier Transform (STFT)
    FIXED: Spectrogram resize to proper 256x256 dimensions
    """
    
    def __init__(self, sampling_rate=1000, target_size=256, n_frames=20):
        self.sampling_rate = sampling_rate
        self.target_size = target_size
        self.n_frames = n_frames
    
    def compute_stft(self, signal, nperseg=512, noverlap=256, window='hann'):
        """Compute STFT with proper error handling"""
        try:
            # Adjust parameters based on signal length
            signal_length = len(signal)
            
            # Ensure nperseg is not larger than signal
            nperseg = min(nperseg, signal_length // 4)
            noverlap = min(noverlap, nperseg // 2)
            
            # Make sure we have enough data
            if signal_length < nperseg:
                # Pad signal if too short
                pad_length = nperseg - signal_length
                signal = np.pad(signal, (0, pad_length), mode='constant')
            
            f, t, Zxx = stft(signal, 
                           fs=self.sampling_rate,
                           window=window,
                           nperseg=nperseg,
                           noverlap=noverlap)
            
            return f, t, Zxx
            
        except Exception as e:
            print(f"Error computing STFT: {str(e)}")
            return None, None, None
    
    def stft_to_spectrogram(self, signal, representation='power', 
                          nperseg=512, noverlap=256, window='hann'):
        """Convert STFT to spectrogram with different representations"""
        f, t, Zxx = self.compute_stft(signal, nperseg, noverlap, window)
        
        if Zxx is None:
            return None, None, None
        
        if representation == 'magnitude':
            spectrogram = np.abs(Zxx)
        elif representation == 'power':
            spectrogram = np.abs(Zxx) ** 2
        elif representation == 'power_spectral_density':
            spectrogram = np.abs(Zxx) ** 2
        elif representation == 'log_magnitude':
            magnitude = np.abs(Zxx)
            spectrogram = 20 * np.log10(magnitude + np.finfo(float).eps)
        elif representation == 'log_power':
            power = np.abs(Zxx) ** 2
            spectrogram = 10 * np.log10(power + np.finfo(float).eps)
        else:
            spectrogram = np.abs(Zxx) ** 2
        
        return f, t, spectrogram
    
    def normalize_spectrogram(self, spectrogram, method='minmax'):
        """Normalize spectrogram"""
        if method == 'minmax':
            min_val = np.min(spectrogram)
            max_val = np.max(spectrogram)
            if max_val > min_val:
                normalized = (spectrogram - min_val) / (max_val - min_val)
            else:
                normalized = spectrogram
        elif method == 'zscore':
            mean_val = np.mean(spectrogram)
            std_val = np.std(spectrogram)
            if std_val > 0:
                normalized = (spectrogram - mean_val) / std_val
            else:
                normalized = spectrogram
        elif method == 'robust':
            p25, p75 = np.percentile(spectrogram, [25, 75])
            median_val = np.median(spectrogram)
            if p75 > p25:
                normalized = (spectrogram - median_val) / (p75 - p25)
            else:
                normalized = spectrogram
        else:
            normalized = spectrogram
        
        return normalized
    
    def resize_spectrogram(self, spectrogram, target_shape):
        """
        Resize spectrogram to target shape using zoom
        FIXED: Properly handle 2D to 2D resizing
        """
        try:
            from scipy.ndimage import zoom
            
            # Ensure target_shape is tuple of 2 elements
            if isinstance(target_shape, (list, tuple)):
                if len(target_shape) > 2:
                    target_shape = (target_shape[0], target_shape[1])
                target_shape = tuple(target_shape)
            else:
                target_shape = (target_shape, target_shape)
            
            # Get current shape
            current_shape = spectrogram.shape
            
            # Verify spectrogram is 2D
            if len(current_shape) != 2:
                print(f"Warning: Unexpected spectrogram shape: {current_shape}")
                # Try to reshape to 2D if possible
                if len(current_shape) == 3:
                    # Take first channel if multi-channel
                    spectrogram = spectrogram[:, :, 0]
                    current_shape = spectrogram.shape
                else:
                    print(f"Error: Cannot process {len(current_shape)}D spectrogram")
                    return spectrogram
            
            # Calculate zoom factors
            zoom_factors = (
                float(target_shape[0]) / float(current_shape[0]),
                float(target_shape[1]) / float(current_shape[1])
            )
            
            # Perform zoom/resize
            resized = zoom(spectrogram, zoom_factors, order=1)
            
            # Verify output shape
            if resized.shape != target_shape:
                print(f"Warning: Resized shape {resized.shape} != target {target_shape}")
                # Force exact size if close
                if abs(resized.shape[0] - target_shape[0]) <= 1 and \
                   abs(resized.shape[1] - target_shape[1]) <= 1:
                    # Crop or pad to exact size
                    resized = self._force_exact_size(resized, target_shape)
            
            return resized
            
        except Exception as e:
            print(f"Error resizing spectrogram: {str(e)}")
            print(f"  Current shape: {spectrogram.shape}")
            print(f"  Target shape: {target_shape}")
            return spectrogram
    
    def _force_exact_size(self, array, target_shape):
        """Force array to exact target shape by cropping or padding"""
        current_shape = array.shape
        result = np.zeros(target_shape)
        
        # Calculate how much to copy
        h_copy = min(current_shape[0], target_shape[0])
        w_copy = min(current_shape[1], target_shape[1])
        
        # Copy data
        result[:h_copy, :w_copy] = array[:h_copy, :w_copy]
        
        return result
    
    def extract_spectrogram_features(self, signal, representation='power',
                               normalize=True, normalize_method='minmax',
                               target_shape=None, nperseg=512, noverlap=256, **kwargs):
        """
        Extract spectrogram features with GUARANTEED 256x256 output
        
        Args:
            signal: Input EMG signal
            representation: Type of spectrogram representation
            normalize: Whether to normalize the spectrogram
            normalize_method: Method for normalization
            target_shape: Target shape for resizing (default: 256x256)
            nperseg: Length of each segment for STFT
            noverlap: Number of points to overlap between segments
        """
        try:
            # CRITICAL: Always use 256x256 target
            if target_shape is None:
                target_shape = (256, 256)
            
            # Ensure target_shape is (256, 256)
            if isinstance(target_shape, (list, tuple)):
                if len(target_shape) >= 2:
                    target_shape = (256, 256)  # Force 256x256
            
            # Compute STFT and convert to spectrogram
            f, t, spectrogram = self.stft_to_spectrogram(
                signal, representation, nperseg, noverlap
            )

            if spectrogram is None:
                print("Warning: STFT computation returned None")
                # Return zero spectrogram instead of None
                return {
                    'spectrogram': np.zeros((256, 256)),
                    'frequencies': None,
                    'times': None,
                    'shape': (256, 256),
                    'representation': representation,
                    'normalized': normalize,
                    'sampling_rate': self.sampling_rate
                }

            # Normalize if requested
            if normalize:
                spectrogram = self.normalize_spectrogram(spectrogram, normalize_method)

            # CRITICAL: Resize to exact 256x256
            spectrogram = self.resize_spectrogram(spectrogram, (256, 256))
            
            # FINAL VERIFICATION
            if spectrogram.shape != (256, 256):
                print(f"WARNING: Final shape {spectrogram.shape} != (256, 256), forcing resize")
                # Force exact size
                spectrogram = self._force_exact_size(spectrogram, (256, 256))

            # Prepare features dictionary
            features = {
                'spectrogram': spectrogram,
                'frequencies': f,
                'times': t,
                'shape': spectrogram.shape,
                'representation': representation,
                'normalized': normalize,
                'sampling_rate': self.sampling_rate
            }

            return features

        except Exception as e:
            print(f"Error extracting spectrogram features: {str(e)}")
            import traceback
            traceback.print_exc()
            # Return zero spectrogram instead of None
            return {
                'spectrogram': np.zeros((256, 256)),
                'frequencies': None,
                'times': None,
                'shape': (256, 256),
                'representation': representation,
                'normalized': normalize,
                'sampling_rate': self.sampling_rate
            }


def extract_batch_spectrograms(segments, labels, extractor, 
                              target_shape=(256, 256), **kwargs):
    """Extract spectrograms with progress tracking"""
    from utils import ProgressBar
    
    all_spectrograms = []
    valid_labels = []
    metadata = []
    
    total_segments = len(segments)
    print(f"\nExtracting spectrograms from {total_segments} segments...")
    print(f"   Target shape: {target_shape}")
    
    progress_bar = ProgressBar(total_segments, desc="Extracting spectrograms", unit="segment")
    
    for i, (segment, label) in enumerate(zip(segments, labels)):
        try:
            features = extractor.extract_spectrogram_features(
                segment, 
                target_shape=target_shape,
                **kwargs
            )
            
            if features is not None:
                spec = features['spectrogram']
                
                if spec.shape != (256, 256):
                    from scipy.ndimage import zoom
                    zoom_factors = (256.0 / spec.shape[0], 256.0 / spec.shape[1])
                    spec = zoom(spec, zoom_factors, order=1)
                
                all_spectrograms.append(spec)
                valid_labels.append(label)
                metadata.append({
                    'original_index': i,
                    'shape': spec.shape,
                    'representation': features['representation']
                })
                
        except Exception as e:
            pass
        
        progress_bar.update(1)
    
    progress_bar.close()
    
    spectrograms = np.array(all_spectrograms)
    valid_labels = np.array(valid_labels)
    
    print(f"✅ Generated {len(spectrograms)} spectrograms of shape {spectrograms.shape[1:]}")
    
    return spectrograms, valid_labels, metadata


class EMGSpectrogramClassifier:
    """
    Class untuk ekstraksi spektrogram dan klasifikasi EMG signals
    """
    
    def __init__(self, sampling_rate=1000, model_type='cnn', input_shape=(256, 256, 1)):
        self.sampling_rate = sampling_rate
        self.model_type = model_type
        self.input_shape = input_shape
        self.model = None
        self.history = None
        self.class_names = None
        
        self.extractor = SpectrogramExtractor(sampling_rate, target_size=256, n_frames=20)
    
    def extract_spectrogram_features(self, *args, **kwargs):
        """Delegate to SpectrogramExtractor"""
        return self.extractor.extract_spectrogram_features(*args, **kwargs)
    
    def create_cnn_model(self, num_classes):
        """Create basic CNN model"""
        model = models.Sequential([
            layers.Conv2D(32, (5, 5), activation='relu', padding='same',
                         input_shape=self.input_shape),
            layers.BatchNormalization(),
            layers.MaxPooling2D((2, 2)),
            layers.Dropout(0.25),
            
            layers.Conv2D(64, (5, 5), activation='relu', padding='same'),
            layers.BatchNormalization(),
            layers.MaxPooling2D((2, 2)),
            layers.Dropout(0.25),
            
            layers.Conv2D(128, (3, 3), activation='relu', padding='same'),
            layers.BatchNormalization(),
            layers.MaxPooling2D((2, 2)),
            layers.Dropout(0.25),
            
            layers.Conv2D(256, (3, 3), activation='relu', padding='same'),
            layers.BatchNormalization(),
            layers.MaxPooling2D((2, 2)),
            layers.Dropout(0.3),
            
            layers.Conv2D(512, (3, 3), activation='relu', padding='same'),
            layers.BatchNormalization(),
            layers.GlobalAveragePooling2D(),
            layers.Dropout(0.5),
            
            layers.Dense(1024, activation='relu'),
            layers.BatchNormalization(),
            layers.Dropout(0.5),
            
            layers.Dense(512, activation='relu'),
            layers.BatchNormalization(),
            layers.Dropout(0.4),
            
            layers.Dense(256, activation='relu'),
            layers.Dropout(0.3),
            
            layers.Dense(num_classes, activation='softmax')
        ])
        
        optimizer = tf.keras.optimizers.Adam(
            learning_rate=0.001,
            beta_1=0.9,
            beta_2=0.999,
            decay=1e-6
        )
        
        model.compile(
            optimizer=optimizer,
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy', 'precision', 'recall']
        )
        
        return model
    
    def prepare_data(self, spectrograms, labels, test_size=0.2, validation_size=0.2):
        """Prepare data for training"""
        self.class_names = np.unique(labels)
        num_classes = len(self.class_names)
        
        print(f"Classes found: {self.class_names}")
        print(f"Number of classes: {num_classes}")
        
        label_counts = np.bincount(labels)
        print(f"Label distribution: {dict(zip(self.class_names, label_counts))}")
        
        X_train, X_test, y_train, y_test = train_test_split(
            spectrograms, labels, test_size=test_size, 
            random_state=42, stratify=labels
        )
        
        X_train, X_val, y_train, y_val = train_test_split(
            X_train, y_train, test_size=validation_size, 
            random_state=42, stratify=y_train
        )
        
        if self.model_type == 'cnn':
            X_train = X_train.reshape(X_train.shape[0], 
                                    X_train.shape[1], 
                                    X_train.shape[2], 1)
            X_val = X_val.reshape(X_val.shape[0], 
                                X_val.shape[1], 
                                X_val.shape[2], 1)
            X_test = X_test.reshape(X_test.shape[0], 
                                  X_test.shape[1], 
                                  X_test.shape[2], 1)
        
        elif self.model_type in ['rf', 'svm']:
            X_train = X_train.reshape(X_train.shape[0], -1)
            X_val = X_val.reshape(X_val.shape[0], -1)
            X_test = X_test.reshape(X_test.shape[0], -1)
        
        print(f"Training set: {X_train.shape}")
        print(f"Validation set: {X_val.shape}")
        print(f"Test set: {X_test.shape}")
        
        return X_train, X_val, X_test, y_train, y_val, y_test
    
    def train_model(self, X_train, y_train, X_val=None, y_val=None, 
                   epochs=100, batch_size=16, enhanced_model=False):
        """Train model"""
        num_classes = len(np.unique(y_train))
        
        if self.model_type == 'cnn':
            self.model = self.create_cnn_model(num_classes)
            
            early_stopping = tf.keras.callbacks.EarlyStopping(
                monitor='val_accuracy', 
                patience=15, 
                restore_best_weights=True,
                mode='max'
            )
            
            reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(
                monitor='val_accuracy', 
                factor=0.5, 
                patience=8, 
                min_lr=1e-7,
                mode='max'
            )
            
            model_checkpoint = tf.keras.callbacks.ModelCheckpoint(
                'best_model.h5',
                monitor='val_accuracy',
                save_best_only=True,
                mode='max'
            )
            
            validation_data = (X_val, y_val) if X_val is not None else None
            
            self.history = self.model.fit(
                X_train, y_train,
                batch_size=batch_size,
                epochs=epochs,
                validation_data=validation_data,
                callbacks=[early_stopping, reduce_lr, model_checkpoint],
                verbose=1
            )
            
        elif self.model_type == 'rf':
            self.model = RandomForestClassifier(
                n_estimators=200, 
                max_depth=20,
                min_samples_split=5,
                min_samples_leaf=2,
                random_state=42, 
                n_jobs=-1
            )
            self.model.fit(X_train, y_train)
            
        elif self.model_type == 'svm':
            self.model = SVC(
                kernel='rbf', 
                C=10,
                gamma='scale',
                random_state=42, 
                probability=True
            )
            self.model.fit(X_train, y_train)
        
        print(f"{self.model_type.upper()} model training completed!")
    
    def predict(self, X):
        """Make predictions"""
        if self.model is None:
            raise ValueError("Model belum dilatih!")
        
        predictions = self.model.predict(X)
        
        if self.model_type == 'cnn':
            probabilities = self.model.predict(X)
            predictions = np.argmax(probabilities, axis=1)
        else:
            probabilities = self.model.predict_proba(X)
        
        return predictions, probabilities
    
    def evaluate_model(self, X_test, y_test):
        """Evaluate model"""
        predictions, probabilities = self.predict(X_test)
        
        accuracy = accuracy_score(y_test, predictions)
        
        print(f"Test Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
        print("\nClassification Report:")
        print(classification_report(y_test, predictions, 
                                  target_names=[f"Class {c}" for c in self.class_names]))
        
        cm = confusion_matrix(y_test, predictions)
        
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                   xticklabels=[f"Class {c}" for c in self.class_names],
                   yticklabels=[f"Class {c}" for c in self.class_names])
        plt.title(f'Confusion Matrix - Accuracy: {accuracy:.4f}')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.tight_layout()
        plt.show()
        
        return {
            'accuracy': accuracy,
            'confusion_matrix': cm,
            'predictions': predictions,
            'probabilities': probabilities
        }


def create_classification_pipeline(spectrograms, labels, model_type='cnn', enhanced_model=False):
    """Create complete classification pipeline"""
    print(f"Starting EMG Spectrogram Classification Pipeline")
    print(f"Model type: {model_type.upper()}")
    print(f"Data shape: {spectrograms.shape}")
    print(f"Number of samples: {len(labels)}")
    
    classifier = EMGSpectrogramClassifier(
        model_type=model_type,
        input_shape=(*spectrograms.shape[1:], 1)
    )
    
    X_train, X_val, X_test, y_train, y_val, y_test = classifier.prepare_data(
        spectrograms, labels
    )
    
    if model_type == 'cnn':
        classifier.train_model(X_train, y_train, X_val, y_val, 
                              epochs=150, batch_size=8, enhanced_model=enhanced_model)
    else:
        classifier.train_model(X_train, y_train, X_val, y_val)
    
    print(f"\nEvaluating model on test set...")
    results = classifier.evaluate_model(X_test, y_test)
    
    accuracy_percentage = results['accuracy'] * 100
    target_met = accuracy_percentage >= 60.0
    
    print(f"\n{'='*50}")
    print(f"FINAL RESULTS")
    print(f"{'='*50}")
    print(f"Model Type: {model_type.upper()}")
    print(f"Final Accuracy: {accuracy_percentage:.2f}%")
    print(f"Target (60%): {'ACHIEVED' if target_met else 'NOT MET'}")
    print(f"{'='*50}")
    
    return classifier, results


if __name__ == "__main__":
    print("Testing Feature Extraction Module...")
    print("Module loaded successfully!")