"""
Data Preprocessing untuk EMG Biosignal Classification
STFT Spectrogram Generation dan Data Augmentation
"""
import numpy as np
from scipy.signal import butter, filtfilt, iirnotch, stft
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg
import warnings
warnings.filterwarnings('ignore')

class EMGPreprocessor:
    """
    Class untuk preprocessing EMG signals:
    - STFT Spectrogram Generation (256x256 pixels)
    - Data Augmentation dengan Gaussian Noise
    - Filtering dan Normalisasi
    """
    def __init__(self, sampling_rate=12804, segment_length=0.5, window_size=256, n_frames=20):
        """
        Initialize preprocessor
        Args:
            sampling_rate (int): Sampling rate dalam Hz
            segment_length (float): Panjang segment dalam DETIK (bisa float untuk ms)
            window_size (int): Window size untuk STFT
            n_frames (int): Number of frames
        """
        self.sampling_rate = sampling_rate
        self.segment_length = segment_length
        self.segment_samples = int(segment_length * sampling_rate)
        print(f"\n[INFO] Preprocessor initialized:")
        print(f"  Sampling rate: {sampling_rate} Hz")
        print(f"  Segment length: {segment_length}s = {self.segment_samples} samples")
        print(f"  Segment duration: {segment_length*1000:.1f} ms")
        if segment_length < 1:
            self.segment_samples = int(segment_length * sampling_rate)
            print(f"Short segment detected: {segment_length}s = {self.segment_samples} samples ({segment_length*1000:.1f} ms)")
        else:
            self.segment_samples = int(segment_length * sampling_rate)
        
        self.window_size = window_size
        self.n_frames = n_frames
        self.spectrogram_size = 64  # Changed from 256 to 64 for short signals
            
    def bandpass_filter(self, data, lowcut=20, highcut=450, filter_order=4):
        """
        Apply bandpass filter untuk EMG signals
        
        Args:
            data (array): Raw signal data
            lowcut (float): Low frequency cutoff (Hz)
            highcut (float): High frequency cutoff (Hz)
            filter_order (int): Filter order
            
        Returns:
            array: Filtered signal
        """
        try:
            nyquist = 0.5 * self.sampling_rate
            low = lowcut / nyquist
            high = min(highcut / nyquist, 0.99)
            
            b, a = butter(filter_order, [low, high], btype='band')
            filtered_data = filtfilt(b, a, data)
            
            return filtered_data
            
        except Exception as e:
            print(f"Warning: Bandpass filter failed - {str(e)}, using original data")
            return data
    
    def notch_filter(self, data, notch_freq=50, quality_factor=30):
        """
        Apply notch filter untuk menghilangkan power line interference
        
        Args:
            data (array): Signal data
            notch_freq (float): Notch frequency (50Hz atau 60Hz)
            quality_factor (float): Quality factor
            
        Returns:
            array: Notch filtered signal
        """
        try:
            nyquist = 0.5 * self.sampling_rate
            freq = notch_freq / nyquist
            
            b, a = iirnotch(freq, quality_factor)
            notched_data = filtfilt(b, a, data)
            
            return notched_data
            
        except Exception as e:
            print(f"Warning: Notch filter failed - {str(e)}, using original data")
            return data
    
    def normalize_signal(self, data, method='rms'):
        """
        Normalize EMG signal
        
        Args:
            data (array): Signal data
            method (str): Normalization method ('rms', 'zscore', 'minmax')
            
        Returns:
            array: Normalized signal
        """
        data = data - np.mean(data)
        
        if method == 'rms':
            rms = np.sqrt(np.mean(data**2))
            if rms > 0:
                normalized_data = data / rms
            else:
                normalized_data = data
                
        elif method == 'zscore':
            std = np.std(data)
            if std > 0:
                normalized_data = data / std
            else:
                normalized_data = data
                
        elif method == 'minmax':
            min_val, max_val = np.min(data), np.max(data)
            if max_val > min_val:
                normalized_data = (data - min_val) / (max_val - min_val)
                normalized_data = 2 * normalized_data - 1
            else:
                normalized_data = data
        else:
            normalized_data = data
            
        return normalized_data
    
    def generate_stft_spectrogram(self, signal):
        """
        Generate STFT spectrogram - OPTIMIZED VERSION for 1.0s segments
        """
        try:
            target_shape = (self.spectrogram_size, self.spectrogram_size)  # (64, 64)
            signal_length = len(signal)
            
            # === OPTIMAL PARAMETERS for SHORT EMG signals (20-50ms) ===
            # Sensorik: ~640 samples at 31949 Hz (20ms)
            # Motorik: ~640 samples at 12804 Hz (50ms)
            
            if signal_length >= 500:  # For 50ms+ signals
                nperseg = 64  # Smaller window for short signals
                noverlap = 48  # 75% overlap
            elif signal_length >= 300:  # For 20-50ms signals  
                nperseg = 32
                noverlap = 24
            else:  # For very short signals
                nperseg = min(16, signal_length // 2)
                noverlap = int(nperseg * 0.75)
            
            # === Ensure minimum signal length ===
            min_signal_length = nperseg * 2
            if signal_length < min_signal_length:
                pad_length = min_signal_length - signal_length
                signal = np.pad(signal, (0, pad_length), mode='reflect')
                signal_length = len(signal)
            
            # === Compute STFT dengan parameter optimal ===
            f, t, Zxx = stft(
                signal, 
                fs=self.sampling_rate,
                nperseg=nperseg,
                noverlap=noverlap,
                window='hann',
                boundary='zeros',
                padded=True
            )
            
            # === Power Spectrogram ===
            spectrogram = np.abs(Zxx) ** 2
            
            # Tambahkan small value untuk menghindari log(0)
            spectrogram_db = 10 * np.log10(spectrogram + 1e-6)
            
            # Gunakan 1st dan 99th percentile untuk lebih robust
            p1, p99 = np.percentile(spectrogram_db, [1, 99])
            spectrogram_clipped = np.clip(spectrogram_db, p1, p99)
            
            spec_min = np.min(spectrogram_clipped)
            spec_max = np.max(spectrogram_clipped)
            
            if spec_max > spec_min:
                spectrogram_normalized = (spectrogram_clipped - spec_min) / (spec_max - spec_min)
            else:
                spectrogram_normalized = spectrogram_clipped
            
            # === Resize to 256x256 ===
            from scipy.ndimage import zoom
            current_shape = spectrogram_normalized.shape
            
            if current_shape != target_shape:
                zoom_factors = (
                    target_shape[0] / current_shape[0],
                    target_shape[1] / current_shape[1]
                )
                spectrogram_resized = zoom(spectrogram_normalized, zoom_factors, order=1)  # order=1 untuk bilinear
                
                # Ensure exact shape
                if spectrogram_resized.shape != target_shape:
                    final_spec = np.zeros(target_shape, dtype=np.float32)
                    min_h = min(spectrogram_resized.shape[0], target_shape[0])
                    min_w = min(spectrogram_resized.shape[1], target_shape[1])
                    final_spec[:min_h, :min_w] = spectrogram_resized[:min_h, :min_w]
                    spectrogram_resized = final_spec
            else:
                spectrogram_resized = spectrogram_normalized
            
            # === Validasi output ===
            if np.any(np.isnan(spectrogram_resized)):
                print("Warning: Spectrogram contains NaN, replacing with zeros")
                spectrogram_resized = np.nan_to_num(spectrogram_resized)
            
            return spectrogram_resized.astype(np.float32)
            
        except Exception as e:
            print(f"Error generating spectrogram: {str(e)}")
            return np.zeros((64, 64), dtype=np.float32)
    
    def add_gaussian_noise(self, signal, noise_threshold=0.05): 
        """
        Add Gaussian noise untuk data augmentation
        Conservative approach untuk medical signals
        """
        # Pastikan signal tidak nol
        signal_std = np.std(signal)
        if signal_std < 1e-6:
            signal_std = 0.1
        
        # SNR-based noise addition
        noise = np.random.normal(0, noise_threshold * signal_std, signal.shape)
        noisy_signal = signal + noise
        return noisy_signal
    
    def add_time_shift_augmentation(self, signal, max_shift=0.1):
        """
        Time shift augmentation
        Shift signal by ±10% of length
        """
        shift_samples = int(len(signal) * max_shift * np.random.uniform(-1, 1))
        shifted_signal = np.roll(signal, shift_samples)
        return shifted_signal

    def add_amplitude_scaling(self, signal, scale_range=(0.9, 1.1)):
        """
        Amplitude scaling augmentation
        Scale amplitude by 90-110%
        """
        scale_factor = np.random.uniform(*scale_range)
        scaled_signal = signal * scale_factor
        return scaled_signal
    
    def segment_signal(self, data, overlap=0.5, min_segment_samples=100):
        """
        Segment signal dengan overlap and optional limit
        
        Args:
            data (array): Signal data
            overlap (float): Overlap ratio (0-1)
            max_segments (int): Maximum number of segments (None = unlimited)
            
        Returns:
            array: Array of segments
        """
        data_length = len(data)
    
        print(f"\n[DEBUG] Processing NCV signal:")
        print(f"  Data length: {data_length} samples")
        print(f"  Signal duration: {data_length/self.sampling_rate*1000:.1f} ms")
        
        # For NCV data: Each file is ONE complete test (639 samples)
        # NO SEGMENTATION NEEDED - use entire signal as-is
        if data_length >= 200:  # Minimum viable signal length
            print(f"  Using entire signal as one segment (NCV test)")
            segment = data.astype(np.float32)
            return np.array([segment], dtype=np.float32)
        
        # Fallback for very short signals (shouldn't happen with NCV data)
        print(f"  WARNING: Signal too short ({data_length} samples < 200)")
        if data_length >= min_segment_samples:
            segment = data.astype(np.float32)
            return np.array([segment], dtype=np.float32)
        
        # Last resort: zero-pad
        print(f"  Zero-padding to {min_segment_samples} samples")
        padded_data = np.zeros(min_segment_samples, dtype=np.float32)
        padded_data[:data_length] = data
        return np.array([padded_data], dtype=np.float32)
    
        print(f"  Generated {len(segments)} segments")
        return np.array(segments, dtype=np.float32)
        
    def process_signal_to_spectrogram(self, raw_signal, apply_bandpass=True, 
                                apply_notch=True, normalize_method='zscore',
                                segment_overlap=0.5, augment_with_noise=True,
                                augment_time_shift=True, augment_amplitude=True, 
                                **kwargs):
        """
        Complete preprocessing pipeline: Signal -> STFT Spectrogram - OPTIMIZED
        """
        try:
            # Handle kwargs parameters
            window_size = kwargs.get('window_size', self.window_size)
            n_frames = kwargs.get('n_frames', self.n_frames)
            
            signal = raw_signal.copy().astype(np.float32)
            
            # === Preprocessing Pipeline ===
            
            # 1. Remove DC offset
            signal = signal - np.mean(signal)
            
            # 2. Bandpass filter (optimized for EMG)
            if apply_bandpass:
                # Gunakan range yang lebih sesuai untuk EMG
                signal = self.bandpass_filter(signal, lowcut=20, highcut=500)
            
            # 3. Notch filter untuk powerline noise
            if apply_notch:
                signal = self.notch_filter(signal, notch_freq=50)
            
            # 4. Normalization (zscore works best for EMG)
            signal = self.normalize_signal(signal, method=normalize_method)
            
            # 5. Segmentasi dengan validasi
            segments = self.segment_signal(signal, overlap=segment_overlap)
            
            if len(segments) == 0:
                print("Warning: No segments generated")
                return np.array([]), None
            
            spectrograms = []
            augmented_spectrograms = []
            
            for segment in segments:
                # Validasi segment
                if len(segment) < 256:  # Minimum samples
                    continue
                
                # Generate spectrogram
                spectrogram = self.generate_stft_spectrogram(segment)
                
                # Validasi spectrogram
                if np.all(spectrogram == 0) or np.any(np.isnan(spectrogram)):
                    continue
                
                spectrograms.append(spectrogram)
                
                if augment_with_noise:
                    # Moderate noise level for balanced augmentation
                    noisy_segment = self.add_gaussian_noise(segment, noise_threshold=0.06)
                    noisy_spec = self.generate_stft_spectrogram(noisy_segment)
                    if noisy_spec is not None and not np.all(noisy_spec == 0):
                        augmented_spectrograms.append(noisy_spec)
                
                if augment_time_shift:
                    # Moderate time shift
                    shifted_segment = self.add_time_shift_augmentation(segment, max_shift=0.07)
                    shifted_spec = self.generate_stft_spectrogram(shifted_segment)
                    if shifted_spec is not None and not np.all(shifted_spec == 0):
                        augmented_spectrograms.append(shifted_spec)
                
                if augment_amplitude:
                    # Moderate amplitude scaling
                    scaled_segment = self.add_amplitude_scaling(segment, scale_range=(0.9, 1.1))
                    scaled_spec = self.generate_stft_spectrogram(scaled_segment)
                    if scaled_spec is not None and not np.all(scaled_spec == 0):
                        augmented_spectrograms.append(scaled_spec)
            
            # Konversi ke numpy array
            if len(spectrograms) > 0:
                spectrograms_array = np.array(spectrograms, dtype=np.float32)
            else:
                spectrograms_array = np.array([], dtype=np.float32)
            
            if len(augmented_spectrograms) > 0:
                augmented_array = np.array(augmented_spectrograms, dtype=np.float32)
            else:
                augmented_array = None
            
            return spectrograms_array, augmented_array
            
        except Exception as e:
            print(f"Error in signal processing: {str(e)}")
            import traceback
            traceback.print_exc()
            return None, None

    
    def process_signal(self, raw_signal, apply_bandpass=True, apply_notch=True, 
                      normalize_method='rms', segment_overlap=0.5):
        """
        Legacy method untuk backward compatibility
        """
        spectrograms, _ = self.process_signal_to_spectrogram(
            raw_signal, apply_bandpass, apply_notch, 
            normalize_method, segment_overlap, augment_with_noise=False)
        return spectrograms
    
    def get_signal_stats(self, signal):
        """
        Get basic statistics dari signal
        
        Args:
            signal (array): Signal data
            
        Returns:
            dict: Signal statistics
        """
        stats = {
            'length': len(signal),
            'duration': len(signal) / self.sampling_rate,
            'mean': np.mean(signal),
            'std': np.std(signal),
            'min': np.min(signal),
            'max': np.max(signal),
            'rms': np.sqrt(np.mean(signal**2))
        }
        return stats

def preprocess_batch_signals_to_spectrograms(signals, labels, preprocessor, 
                                            augment_data=True, **kwargs):
        """
        Preprocess batch with progress tracking and MEMORY OPTIMIZATION
        
        Args:
            signals: List of raw signals
            labels: List of labels
            preprocessor: EMGPreprocessor instance
            augment_data: Enable data augmentation
            
        Returns:
            (spectrograms, labels) as numpy arrays with optimized dtype
        """
        from utils import ProgressBar
        
        all_spectrograms = []
        all_labels = []
        
        total_signals = len(signals)
        print(f"\nProcessing {total_signals} signals to spectrograms...")
        
        signal_length = len(signals[0]) if len(signals) > 0 else 0
        segment_samples = preprocessor.segment_samples
        segment_overlap = kwargs.get('segment_overlap', 0.5)
        
        if segment_samples > 0:
            step = int(segment_samples * (1 - segment_overlap))
            estimated_segments_per_signal = max(1, (signal_length - segment_samples) // step + 1)
            
            if augment_data:
                estimated_segments_per_signal *= 2
            
            total_estimated_spectrograms = total_signals * estimated_segments_per_signal
            
            # Memory estimate (MB)
            memory_per_spec = 256 * 256 * 4  # float32 = 4 bytes
            total_memory_mb = (total_estimated_spectrograms * memory_per_spec) / (1024 * 1024)
            
            print(f"  Signal length: {signal_length} samples ({signal_length/1000:.1f}s)")
            print(f"  Segment size: {segment_samples} samples ({segment_samples/1000:.3f}s)")
            print(f"  Estimated segments per signal: ~{estimated_segments_per_signal}")
            print(f"  Estimated total spectrograms: ~{total_estimated_spectrograms:,}")
            print(f"  Estimated memory: ~{total_memory_mb:.1f} MB")
            
            if total_memory_mb > 4000:  # 4GB warning
                print(f"\nWARNING: High memory usage detected!")
                print(f"     Consider:")
                print(f"       - Increasing segment_length_seconds in config.json")
                print(f"       - Reducing segment_overlap")
                print(f"       - Disabling augmentation temporarily")
                print(f"       - Processing data in batches\n")
        
        progress_bar = ProgressBar(total_signals, desc="Processing", unit="signal")
        
        for i, (signal, label) in enumerate(zip(signals, labels)):
            spectrograms, augmented_spectrograms = preprocessor.process_signal_to_spectrogram(
                signal, augment_with_noise=augment_data, **kwargs)
            
            if spectrograms is not None and len(spectrograms) > 0:
                spectrograms = spectrograms.astype(np.float32)
                all_spectrograms.extend(spectrograms)
                all_labels.extend([label] * len(spectrograms))
                
                if augment_data and augmented_spectrograms is not None:
                    augmented_spectrograms = augmented_spectrograms.astype(np.float32)
                    all_spectrograms.extend(augmented_spectrograms)
                    all_labels.extend([label] * len(augmented_spectrograms))
            
            progress_bar.update(1)
        
        progress_bar.close()
        
        spectrograms_array = np.array(all_spectrograms, dtype=np.float32)
        labels_array = np.array(all_labels, dtype=np.int32)
        
        total_spectrograms = len(spectrograms_array)
        actual_memory_mb = (spectrograms_array.nbytes) / (1024 * 1024)
        
        augmentation_status = "enabled" if augment_data else "disabled"
        print(f"\n\Generated {total_spectrograms:,} spectrograms")
        print(f"   Shape: {spectrograms_array.shape}")
        print(f"   Dtype: {spectrograms_array.dtype}")
        print(f"   Memory: {actual_memory_mb:.1f} MB")
        print(f"   Augmentation: {augmentation_status}")
        
        return spectrograms_array, labels_array



def preprocess_batch_signals(signals, labels, preprocessor, **kwargs):
    """
    Legacy function untuk backward compatibility
    
    Args:
        signals (list): List of raw signals
        labels (list): List of labels  
        preprocessor (EMGPreprocessor): Preprocessor instance
        **kwargs: Additional arguments including augment_data
        
    Returns:
        tuple: (spectrograms, spectrogram_labels)
    """
    # Extract augment_data from kwargs if present, otherwise use default False for backward compatibility
    augment_data = kwargs.pop('augment_data', False)
    
    return preprocess_batch_signals_to_spectrograms(
        signals, labels, preprocessor, 
        augment_data=augment_data, 
        **kwargs
    )

if __name__ == "__main__":
    print("Testing EMG Preprocessor with STFT Spectrogram...")
    
    fs = 1000
    duration = 5
    t = np.linspace(0, duration, fs * duration)
    
    emg_signal = (np.random.randn(len(t)) * 0.1 + 
                  0.5 * np.sin(2*np.pi*100*t) +
                  0.2 * np.sin(2*np.pi*50*t))
    
    preprocessor = EMGPreprocessor(sampling_rate=fs, segment_length=2, 
                                 window_size=128, n_frames=None)
    
    original_stats = preprocessor.get_signal_stats(emg_signal)
    print("\nOriginal Signal Stats:")
    for key, value in original_stats.items():
        print(f"  {key}: {value:.4f}")
    
    spectrograms, augmented_spectrograms = preprocessor.process_signal_to_spectrogram(
        emg_signal, augment_with_noise=True)
    
    if spectrograms is not None:
        print(f"\nProcessing Results:")
        print(f"  Original signal length: {len(emg_signal)} samples")
        print(f"  Number of original spectrograms: {len(spectrograms)}")
        print(f"  Number of augmented spectrograms: {len(augmented_spectrograms) if augmented_spectrograms is not None else 0}")
        print(f"  Spectrogram shape: {spectrograms[0].shape}")
        print(f"  Total dataset size: {len(spectrograms) + (len(augmented_spectrograms) if augmented_spectrograms is not None else 0)}")
    
    print("\nPreprocessor test completed!")