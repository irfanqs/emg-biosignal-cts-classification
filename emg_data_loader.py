"""
EMG Data Loader - FIXED VERSION
Memperbaiki parsing untuk format data horizontal (multi-channel per row)
"""

import os
import numpy as np
import pandas as pd
import re
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')


class EMGFileReader:
    """
    Complete EMG file reader dengan header parsing
    Support multi-channel EMG data dengan CSV header format
    """
    
    def __init__(self):
        self.metadata = {}
        self.data = None
    
    def read_emg_file(self, filepath):
        """
        Read EMG file lengkap: header + data
        
        Args:
            filepath (str): Path ke file EMG
            
        Returns:
            tuple: (data_array, metadata_dict)
        """
        print(f"Reading: {os.path.basename(filepath)}")
        
        # Parse header
        metadata = self._parse_header(filepath)
        
        # Load data
        data = self._load_trace_data(filepath, metadata)
        
        # Validate
        if data is not None:
            print(f"Loaded: {data.shape} | Channels: {metadata['n_channels']} | SR: {metadata['sampling_rate']} Hz")
        else:
            print(f"Failed to load data")
        
        return data, metadata
    
    def _parse_header(self, filepath):
        """Parse CSV header untuk extract metadata"""
        metadata = {
            'filepath': filepath,
            'filename': os.path.basename(filepath),
            'test_name': None,
            'test_item': None,
            'patient_name': None,
            'patient_id': None,
            'test_date': None,
            'channels': [],
            'sensitivity': [],
            'hicut': None,
            'locut': None,
            'ms_per_sample': None,
            'samples_per_channel': None,
            'sampling_rate': 1000,  # Default
            'n_channels': 0,
            'signal_type': 'Unknown',
            'header_lines': 0
        }
        
        try:
            with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
                lines = f.readlines()
            
            # Find "Trace Data" line
            trace_data_idx = 0
            for idx, line in enumerate(lines):
                if 'Trace Data' in line:
                    trace_data_idx = idx
                    metadata['header_lines'] = idx + 1  # +1 to skip "Trace Data" line
                    break
            
            # Parse each header line
            for i in range(trace_data_idx):
                line = lines[i].strip()
                if not line:
                    continue
                
                # Split by comma
                parts = [p.strip().strip('"') for p in line.split(',')]
                
                if len(parts) < 2:
                    continue
                
                key = parts[0]
                values = parts[1:]
                
                # === Extract fields ===
                
                if 'Test Name' in key:
                    metadata['test_name'] = values[0] if values else None
                
                elif 'Test Item' in key:
                    metadata['test_item'] = values[0] if values else None
                    # Determine signal type dari test item
                    test_item = values[0] if values else ''
                    if 'Motor' in test_item or 'CVM' in test_item:
                        metadata['signal_type'] = 'Motorik'
                    elif 'Sensory' in test_item or 'CVS' in test_item or 'Anti Sensory' in test_item:
                        metadata['signal_type'] = 'Sensorik'
                
                elif 'Patient Name' in key:
                    metadata['patient_name'] = values[0] if values else None
                
                elif 'Patient ID' in key:
                    metadata['patient_id'] = values[0] if values else None
                
                elif 'Test Date' in key:
                    metadata['test_date'] = values[0] if values else None
                
                elif 'Trace Label' in key:
                    # Extract channel names (remove empty strings)
                    channels = [v.strip(' :') for v in values if v.strip(' :')]
                    metadata['channels'] = channels
                    metadata['n_channels'] = len(channels)
                
                elif 'Sensitivity' in key:
                    # Extract sensitivity values (µV/Div)
                    try:
                        sens_values = [float(v) for v in values if v and v.replace('.', '').replace('-', '').isdigit()]
                        metadata['sensitivity'] = sens_values
                    except:
                        pass
                
                elif 'Hicut' in key:
                    try:
                        metadata['hicut'] = float(values[0]) if values else None
                    except:
                        pass
                
                elif 'Locut' in key:
                    try:
                        metadata['locut'] = float(values[0]) if values else None
                    except:
                        pass
                
                elif 'ms/Sample' in key:
                    try:
                        ms_per_sample = float(values[0]) if values else None
                        metadata['ms_per_sample'] = ms_per_sample
                        # Calculate sampling rate
                        if ms_per_sample and ms_per_sample > 0:
                            metadata['sampling_rate'] = 1000.0 / ms_per_sample
                    except:
                        pass
                
                elif 'Samples' in key:
                    try:
                        metadata['samples_per_channel'] = int(values[0]) if values else None
                    except:
                        pass
        
        except Exception as e:
            print(f"Header parsing warning: {str(e)}")
        
        return metadata
    
    def _load_trace_data(self, filepath, metadata):
        """
        Load trace data setelah header
        FORMAT: Setiap baris = 1 sample dengan N channels (horizontal)
        
        Contoh untuk 3 channels:
        Trace Data (µV),-26758.64,-21.47,-97.28    ← Sample 0: [Ch1, Ch2, Ch3]
                       ,15079.60,-143.27,-18.96     ← Sample 1: [Ch1, Ch2, Ch3]
                       ,140.47,-259.54,-174.49      ← Sample 2: [Ch1, Ch2, Ch3]
        
        Args:
            filepath: Path ke file
            metadata: Metadata dict dari parse_header
            
        Returns:
            numpy array: (n_samples, n_channels)
        """
        try:
            header_lines = metadata.get('header_lines', 12)
            n_channels = metadata.get('n_channels', 1)
            
            # Read all lines
            with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
                lines = f.readlines()
            
            # Parse data starting from header_lines
            data_rows = []
            
            for line_idx in range(header_lines, len(lines)):
                line = lines[line_idx].strip()
                
                if not line:
                    continue
                
                # Split by comma
                parts = line.split(',')
                
                # Extract numeric values only (skip empty first element if exists)
                row_values = []
                for part in parts:
                    part = part.strip()
                    if not part:  # Skip empty strings
                        continue
                    
                    try:
                        value = float(part)
                        row_values.append(value)
                    except ValueError:
                        # Skip non-numeric values
                        continue
                
                # Validate row
                if len(row_values) == 0:
                    continue
                
                # Handle channel count mismatch
                if len(row_values) > n_channels:
                    # Take last n_channels values (in case there's extra column)
                    row_values = row_values[-n_channels:]
                elif len(row_values) < n_channels:
                    # Pad with zeros if missing channels
                    row_values.extend([0.0] * (n_channels - len(row_values)))
                
                data_rows.append(row_values)
            
            if len(data_rows) == 0:
                print(f"   No data extracted from file")
                return None
            
            # Convert to numpy array: shape = (n_samples, n_channels)
            data = np.array(data_rows, dtype=np.float32)
            
            print(f"Extracted: {data.shape[0]} samples × {data.shape[1]} channels")
            
            # Validate expected samples
            expected_samples = metadata.get('samples_per_channel')
            if expected_samples:
                actual_samples = data.shape[0]
                if abs(actual_samples - expected_samples) > 10:
                    print(f"Sample count mismatch: got {actual_samples}, expected {expected_samples}")
            
            return data
        
        except Exception as e:
            print(f"  Data loading error: {str(e)}")
            import traceback
            traceback.print_exc()
            
            # Final fallback: pandas with skiprows
            try:
                print(f"Trying pandas fallback...")
                
                df = pd.read_csv(
                    filepath,
                    skiprows=header_lines,
                    header=None,
                    encoding='utf-8',
                    on_bad_lines='skip'
                )
                
                # Remove empty columns
                df = df.dropna(axis=1, how='all')
                
                # Convert to numpy
                data = df.values.astype(np.float32)
                
                # Filter rows with all NaN
                data = data[~np.isnan(data).all(axis=1)]
                
                print(f"  Pandas fallback successful: {data.shape}")
                
                return data
            
            except Exception as e2:
                print(f"  Pandas fallback also failed: {str(e2)}")
                return None


class EMGDataLoader:
    """
    Data loader untuk EMG dataset dengan struktur folder:
    - Full_Data/
    - Motorik/
    - Sensorik/
    """
    
    def __init__(self, data_directory):
        self.data_directory = data_directory
        self.reader = EMGFileReader()
        self.class_mapping = {
            'non_cts': 0,
            'mild': 1,
            'moderate': 2,
            'severe': 3
        }
    
    def scan_emg_files(self):
        """
        Scan EMG files dalam struktur folder
        
        Returns:
            dict: {class_name: [list of file paths]}
        """
        file_paths = {}
        
        for class_name in self.class_mapping.keys():
            class_dir = os.path.join(self.data_directory, class_name)
            
            if os.path.exists(class_dir):
                # Find all .txt and .csv files
                files = [
                    f for f in os.listdir(class_dir)
                    if f.endswith(('.txt', '.csv', '.TXT', '.CSV'))
                ]
                
                file_paths[class_name] = [
                    os.path.join(class_dir, f) for f in files
                ]
            else:
                file_paths[class_name] = []
        
        return file_paths
    
    def load_single_file(self, filepath):
        """
        Load single EMG file dengan header parsing
        
        Args:
            filepath (str): Path ke file EMG
            
        Returns:
            tuple: (data, metadata) atau (None, None) jika gagal
        """
        try:
            data, metadata = self.reader.read_emg_file(filepath)
            return data, metadata
        
        except Exception as e:
            print(f"   Error loading {os.path.basename(filepath)}: {str(e)}")
            return None, None
    
    def load_all_emg_data(self, max_files_per_class=None, use_metadata=True):
        """
        Load semua EMG data dari folder
        
        Args:
            max_files_per_class (int): Limit jumlah file per class (untuk testing)
            use_metadata (bool): Store metadata atau tidak
            
        Returns:
            tuple: (signals_list, labels_array, info_list)
        """
        file_paths = self.scan_emg_files()
        
        all_signals = []
        all_labels = []
        all_info = []
        
        print(f"\n{'='*70}")
        print(f"LOADING EMG DATA FROM: {self.data_directory}")
        print(f"{'='*70}\n")
        
        for class_name, files in file_paths.items():
            if max_files_per_class:
                files = files[:max_files_per_class]
            
            class_label = self.class_mapping[class_name]
            
            print(f"Class: {class_name.upper()} ({len(files)} files)")
            print(f"{'─'*70}")
            
            for filepath in files:
                data, metadata = self.load_single_file(filepath)
                
                if data is not None and len(data) > 0:
                    all_signals.append(data)
                    all_labels.append(class_label)
                    
                    info = {
                        'filepath': filepath,
                        'class': class_name,
                        'label': class_label,
                        'shape': data.shape,
                        'n_channels': data.shape[1] if len(data.shape) > 1 else 1,
                        'n_samples': len(data),
                        'duration_sec': len(data) / metadata.get('sampling_rate', 1000)
                    }
                    
                    if use_metadata:
                        info['metadata'] = metadata
                    
                    all_info.append(info)
            
            print()
        
        # Summary
        print(f"{'='*70}")
        print(f"LOADING SUMMARY")
        print(f"{'='*70}")
        print(f"Total files loaded: {len(all_signals)}")
        print(f"Class distribution:")
        
        unique_labels, counts = np.unique(all_labels, return_counts=True)
        for label, count in zip(unique_labels, counts):
            class_name = [k for k, v in self.class_mapping.items() if v == label][0]
            print(f"   {class_name}: {count} files")
        
        if all_signals:
            # Check if multi-channel
            first_shape = all_signals[0].shape
            if len(first_shape) > 1 and first_shape[1] > 1:
                print(f"Multi-channel data detected: {first_shape[1]} channels per signal")
            else:
                print(f"Single-channel data")
        
        print(f"{'='*70}\n")
        
        return all_signals, np.array(all_labels), all_info
    
    def get_dataset_statistics(self, signals, labels, info_list):
        """
        Get comprehensive statistics tentang dataset
        
        Returns:
            dict: Dataset statistics
        """
        stats = {
            'total_files': len(signals),
            'total_samples': sum(len(s) for s in signals),
            'class_distribution': {},
            'signal_shapes': {},
            'sampling_rates': [],
            'durations': [],
            'n_channels_list': []
        }
        
        # Class distribution
        for label in np.unique(labels):
            class_name = [k for k, v in self.class_mapping.items() if v == label][0]
            stats['class_distribution'][class_name] = int(np.sum(labels == label))
        
        # Signal properties
        for info in info_list:
            if 'metadata' in info:
                meta = info['metadata']
                stats['sampling_rates'].append(meta.get('sampling_rate', 1000))
            
            stats['durations'].append(info['duration_sec'])
            stats['n_channels_list'].append(info['n_channels'])
        
        # Aggregate
        if stats['sampling_rates']:
            stats['mean_sampling_rate'] = np.mean(stats['sampling_rates'])
            stats['std_sampling_rate'] = np.std(stats['sampling_rates'])
        
        if stats['durations']:
            stats['mean_duration'] = np.mean(stats['durations'])
            stats['total_duration'] = np.sum(stats['durations'])
        
        if stats['n_channels_list']:
            stats['max_channels'] = max(stats['n_channels_list'])
            stats['min_channels'] = min(stats['n_channels_list'])
        
        return stats
    
    def print_dataset_info(self, signals, labels, info_list):
        """Print comprehensive dataset information"""
        stats = self.get_dataset_statistics(signals, labels, info_list)
        
        print(f"\n{'='*70}")
        print(f"DATASET INFORMATION")
        print(f"{'='*70}")
        print(f"Total files: {stats['total_files']}")
        print(f"Total samples: {stats['total_samples']:,}")
        print(f"Total duration: {stats.get('total_duration', 0):.1f} seconds")
        print(f"Sampling rate: {stats.get('mean_sampling_rate', 1000):.0f} Hz")
        print(f"Channels: {stats.get('min_channels', 1)}-{stats.get('max_channels', 1)}")
        
        print(f"\nClass Distribution:")
        for class_name, count in stats['class_distribution'].items():
            percentage = (count / stats['total_files']) * 100
            bar = '█' * int(percentage / 2)
            print(f"   {class_name:10s}: {count:3d} files ({percentage:5.1f}%) {bar}")
        
        print(f"{'='*70}\n")


# Standalone testing
if __name__ == "__main__":
    print("Testing EMG Data Loader (FIXED VERSION)...")
    
    # Test dengan sample file
    test_dir = "data/Motorik"  # Adjust path
    
    if os.path.exists(test_dir):
        loader = EMGDataLoader(test_dir)
        
        # Scan files
        files = loader.scan_emg_files()
        print(f"\nFound files:")
        for class_name, file_list in files.items():
            print(f"  {class_name}: {len(file_list)} files")
        
        # Load all data
        signals, labels, info = loader.load_all_emg_data(max_files_per_class=2)
        
        # Print info
        loader.print_dataset_info(signals, labels, info)
        
        # Verify first signal shape
        if len(signals) > 0:
            print(f"\nFirst signal shape: {signals[0].shape}")
            print(f"First 3 samples:")
            print(signals[0][:3])
        
        print("\nData loader test completed!")
    else:
        print(f"\nTest directory not found: {test_dir}")
        print("Please update test_dir path in __main__ section")