EMG CTS Classification System
Sistem klasifikasi tingkat keparahan Carpal Tunnel Syndrome dari sinyal EMG menggunakan deep learning. Straightforward, no BS.

Ringkasan Cepat
Input: Raw EMG signal files (.txt/.csv)
Output: Classification ke 4 kelas (non-CTS, mild, moderate, severe)
Target: Val accuracy ≥60% (realistic untuk medical 4-class)
Pipeline:
Raw EMG → Filtering → Segmentation → STFT → Spectrogram → CNN/LSTM → Prediction

Requirements
Hardware

RAM: 16GB (32GB ideal)
GPU: NVIDIA 6GB+ VRAM (strongly recommended)
Storage: 10GB free

Software
bashPython 3.8-3.10
tensorflow >= 2.10.0
numpy, scipy, scikit-learn
pandas, matplotlib, seaborn

Quick Start

1. Install Dependencies
   bash# Create virtual environment
   python -m venv venv
   source venv/bin/activate # Linux/Mac

# atau

venv\Scripts\activate # Windows

# Install packages

pip install tensorflow numpy scipy scikit-learn pandas matplotlib seaborn

```

### 2. Prepare Your Data

Organize files like this (EXACT structure required):
```

data/
├── Motorik/ # atau Full_Data atau Sensorik
│ ├── non_cts/
│ │ ├── patient1.txt
│ │ └── patient2.txt
│ ├── mild/
│ ├── moderate/
│ └── severe/
Data format: EMG files with header (auto-parsed) or plain text numbers.
Requirements:

Minimum 10 files per class
Sampling rate: 12,804 Hz (auto-detected)
Duration: 20-50ms per signal (very short!)

3. Configure
   Edit config.json:
   json{
   "data": {
   "base_directory": "D:\\path\\to\\your\\data" // UPDATE THIS
   },

"preprocessing": {
"segment_length_seconds": 0.04, // 40ms - CRITICAL!
"segment_overlap": 0.5,
"bandpass_lowcut": 10,
"bandpass_highcut": 2500,
"normalization_method": "zscore",
"augmentation": {
"gaussian_noise": {
"threshold": 0.15
}
}
},

"feature_extraction": {
"stft": {
"window_size": 64, // 5ms window
"overlap": 48
}
},

"training": {
"hyperparameters": {
"epochs": 150,
"batch_size": 16,
"learning_rate": 0.001
}
}
} 4. Run
Interactive menu:
bashpython main_menu.py
Pick option 2 (Motorik) or 3 (Sensorik), select model, done.
Auto-comparison (Motorik vs Sensorik):
bashpython run_model_comparison.py

```

---

## Project Structure
```

emg-classification/
├── config.json # Main configuration
├── main_menu.py # Interactive interface
├── run_model_comparison.py # Auto Motorik vs Sensorik
│
├── emg_data_loader.py # EMG file parser
├── data_preprocessing.py # STFT spectrogram generation
├── models.py # CNN, LSTM, CNN-LSTM
├── train.py # Training loop
├── evaluate.py # Metrics & confusion matrix
├── utils.py # Helpers
│
├── data/ # Your EMG files here
└── experiments/ # Output (auto-generated)
└── exp_YYYYMMDD_HHMMSS/
├── models/ # Trained .h5 files
├── plots/ # Confusion matrices
├── results/ # CSV metrics
└── FINAL_REPORT.md # Summary

Understanding Configuration
Critical Parameters (MUST get right!)
segment_length_seconds: 0.04

Your data is 20-50ms long
Using 1-2 seconds = 95% padding zeros
Model learns noise, not features
Solution: 40ms matches actual signal duration

window_size: 64

For 40ms signal, window 512 = only 1-2 STFT frames
Window 64 = ~8-12 frames with temporal info
Better time-frequency resolution

batch_size: 16

Batch size 3 = unstable training
16 = sweet spot for stability vs memory

gaussian_noise: 0.15

80% noise destroys EMG features
15% = safe augmentation level

Why These Differ from Defaults
Original config had:

segment_length: 1.0 second → WAY too long
window_size: 512 → Too large for short signals
batch_size: 3 → Too small for stable gradients
noise: 0.8 → Destroys signal integrity

Result: ~25% accuracy (random guess) with old config
Expected: 60-75% accuracy with new config

Workflow Examples
Example 1: Quick Test (30 min)
Test with limited data first:
bash# Edit config.json
"max_files_per_class": 5
"epochs": 20

# Run

python main_menu.py

# Pick option 2, model 1 (CNN only)

Example 2: Single Model Full Training (2-3 hours)
bash# config.json with full settings
"max_files_per_class": null # Use all data
"epochs": 150

python main_menu.py

# Option 2 or 3, pick CNN

Example 3: Compare All Models (8-10 hours)
bashpython main_menu.py

# Option 2 or 3, pick option 4 (ALL MODELS)

Example 4: Motorik vs Sensorik (6-8 hours)
bashpython run_model_comparison.py

```

Trains 3 models on Motorik, 3 on Sensorik, generates comparison report.

---

## Output & Results

### What You Get
```

experiments/motorik_classification_20241204_143022/
├── FINAL_REPORT.md # Read this first
├── models/
│ ├── motorik_standard_cnn.h5
│ ├── motorik_standard_lstm.h5
│ └── motorik_standard_cnn_lstm.h5
├── plots/
│ ├── all_confusion_matrices.png
│ └── training_history.png
└── results/
└── model_comparison.csv # Excel-friendly metrics

```

### Reading Confusion Matrix
```

              Predicted
           NC  Mild  Mod  Sev

Actual NC [42 3 1 0] Good: 91% correct
Mild[ 2 38 5 1] OK: some Mod confusion
Mod[ 1 4 40 3] OK: mild-moderate overlap natural
Sev[ 0 1 2 45] Excellent: 94% correct

```

**Look for:**
- High diagonal = good
- NC vs Severe never confused = critical distinction works
- Mild vs Moderate overlap = expected (borderline cases)

### Performance Targets

| Metric | Minimum | Good | Excellent |
|--------|---------|------|-----------|
| Accuracy | 60% | 70% | 75%+ |
| Precision | 0.60 | 0.70 | 0.75+ |
| Recall | 0.60 | 0.70 | 0.75+ |
| F1-Score | 0.60 | 0.70 | 0.75+ |

**Reality check:** 4-class medical = hard. 60-70% is respectable, 75%+ is publication-worthy.

---

## Troubleshooting

### Out of Memory
```

ResourceExhaustedError: OOM when allocating tensor
Fix:
json// In config.json
"batch_size": 8 // or even 4
Or train one model at a time, or force CPU:
pythonimport os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

```

### Data Not Found
```

ERROR: Motorik directory not found

```

**Fix:**
- Use double backslash in Windows: `"D:\\data\\folder"`
- Verify folder structure: `dir data\Motorik` should show `non_cts, mild, moderate, severe`
- Use absolute path, not relative

### Config Not Being Read

If training shows:
```

Segment Length: 1.0s = 12804 samples // WRONG!
STFT Window: 512 samples // WRONG!
Problem: Code using defaults, ignoring config.
Fix: Update main_menu.py line 283-286:
pythonpreprocessor = EMGPreprocessor(
sampling_rate=config.get('signal_processing', {}).get('sampling_rate', 12804),
segment_length=config.get('preprocessing', {}).get('segment_length_seconds', 0.04),
window_size=config.get('feature_extraction', {}).get('stft', {}).get('window_size', 64)
)

```

### Accuracy Stuck at 25% (Random Guess)

**Causes:**
1. Config not read (see above)
2. Still using 1-2 second segments
3. STFT window still 512

**Verify with print output:**
```

Should show:
Segment Length: 0.04s = 512 samples
STFT Window: 64 samples = 5.00 ms

If shows:
Segment Length: 1.0s = 12804 samples
STFT Window: 512 samples = 40.00 ms
If wrong = config not being read. Fix code first.
Training Too Slow
Check GPU:
pythonimport tensorflow as tf
print(tf.config.list_physical_devices('GPU'))
If empty list = GPU not detected.
Options:

Install CUDA Toolkit + cuDNN
Use Google Colab (free T4 GPU)
Reduce epochs for testing

Technical Details
Signal Processing
Bandpass Filter: 10-2500 Hz (4th order Butterworth)

Wider than standard 20-450 Hz
CTS features often <20 Hz
Muscle activity up to 2000+ Hz

Notch Filter: 50 Hz (or 60 Hz for US)

Removes power line interference
Quality factor: 30

Normalization: Z-score

(signal - mean) / std
Better than RMS for classification
Preserves relative power across frequencies

STFT Spectrogram
ParameterValueReasonWindow64 samples5ms at 12,804 HzOverlap48 samples75% overlapWindow typeHannSmooth, minimal leakageOutput256×256Standard CNN input
Why 64 not 512?

40ms signal = 512 samples total
Window 512 = 1 frame (no temporal info)
Window 64 = 8 frames (captures dynamics)

Model Architectures
Standard CNN:

4 conv blocks (64→128→256→512 filters)
Global average pooling
2 dense layers (512→256)
Gradual dropout (0.2→0.4)
~5M parameters

Standard LSTM:

2 Bidirectional LSTM (64 units)
2 dense layers (256→128)
Dropout 0.3
~2M parameters

Standard CNN-LSTM:

CNN feature extraction (3 blocks)
LSTM temporal modeling (2 layers)
~3.5M parameters

Data Augmentation
Applied during preprocessing:

Gaussian noise: +15% SNR
Time shift: ±10% signal length
Amplitude scaling: ×0.9 to ×1.1

Results in 4× data (1 original + 3 augmented per segment).

Advanced Usage
Custom Model
Add to models.py:
pythonclass YourCustomModel:
def **init**(self, input_shape, num_classes): # Your architecture
pass

    def build_model(self):
        # Build and return model
        pass

Register in config.json:
json"models": [
{
"name": "your_model",
"type": "your_custom",
"params": {...}
}
]
Cross-Validation
Enable in config:
json"cross_validation": {
"enabled": true,
"n_splits": 5
}
Channel Selection (Multi-channel Data)
If your data has 10 channels, automatic statistical selection happens in preprocessing. Check channel_analysis/ in output.

Best Practices
Development

Start with 10 files per class
Verify config being read (check print outputs)
Train one model first (CNN)
Check confusion matrix for obvious issues
Iterate on parameters

Production

Use full dataset
Train all 3 models
Enable cross-validation
Document all changes
Save everything (models, plots, logs)

Publication

Run 5-fold CV
Report mean ± std
Statistical comparison between models
Show failure cases (confusion matrix analysis)
Ablation study (test each preprocessing step)

FAQ
Q: Can I use different sampling rate?
A: Yes, update config.json. Auto-detect usually works.
Q: My data is 2-class (healthy vs CTS), not 4?
A: Remove mild/moderate/severe folders, update class_mapping in code.
Q: Training on CPU?
A: Works but 10-20× slower. Use Colab for free GPU.
Q: Model overfitting (90% train, 50% val)?
A: Increase dropout, more augmentation, or reduce model size.
Q: Can I use VGG or ResNet?
A: Yes, but overkill for 256×256. Standard CNN sufficient.
Q: Results not reproducible?
A: Set random_seed: 42 in config, but GPU ops have some randomness.

Known Issues

Config sometimes not read - Fixed in latest main_menu.py
Memory leak after 3-4 models - Restart kernel between runs
Slow on HDD - Use SSD for data folder
TensorFlow warnings - Cosmetic, ignore

Citation
bibtex@software{emg_cts_classification_2024,
title = {EMG Classification for CTS Severity Detection},
author = {Your Name},
year = {2024},
url = {https://github.com/yourusername/emg-classification}
}

Support
Stuck? Check in order:

This README troubleshooting section
experiments/\*/logs/experiment.log
Verify config values in print output
Test with small dataset first

License
MIT License - do whatever you want with it.

Version: 2.0 (December 2024)
Status: Production-ready after config fixes
Expected accuracy: 60-75% for 4-class CTS classification
