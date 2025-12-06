import os
import numpy as np

files_to_check = [
    'data/data/Sensorik/severe/170451_Median Sensory_L.csv',
    'data/data/Motorik/severe/245527_Median Motor_L.csv', 
    'data/data/Sensorik/non_cts/141135_Median Sensory_R.csv',
    'data/data/Motorik/mild/108198_Median Motor_L.csv',
    'data/data/Sensorik/mild/1013777_Median Sensory_R.csv'
]

print('='*80)
print('EMG DATA FILE ANALYSIS')
print('='*80)

for filepath in files_to_check:
    if not os.path.exists(filepath):
        print(f'\nFile not found: {filepath}')
        continue
        
    lines = open(filepath, 'r', encoding='latin-1').readlines()
    
    # Parse header
    ms_per_sample = float(lines[9].split(',')[1])
    samples = int(lines[10].split(',')[1])
    
    # Calculate properties
    duration_ms = samples * ms_per_sample
    duration_s = duration_ms / 1000
    sampling_rate = 1000 / ms_per_sample
    
    # Count data values
    data_lines = [l.strip() for l in lines[11:]]
    values = []
    for line in data_lines:
        values.extend([float(x) for x in line.split(',') if x and x.replace('-','').replace('.','').isdigit()])
    
    print(f'\n{os.path.basename(filepath)}:')
    print(f'  Samples per channel: {samples}')
    print(f'  ms/sample: {ms_per_sample} ms')
    print(f'  Sampling rate: {sampling_rate:.1f} Hz')
    print(f'  Duration: {duration_ms:.2f} ms ({duration_s:.4f} seconds)')
    print(f'  Total data values: {len(values)}')
    print(f'  Estimated channels: {len(values) // samples}')

print('\n' + '='*80)
print('SUMMARY:')
print('='*80)
print('PROBLEM: Signal duration is TOO SHORT!')
print('  - Sensorik (Sensory): ~20 ms (0.020 seconds)')
print('  - Motorik (Motor): ~50 ms (0.050 seconds)')
print('  - Config setting: 1.0 second segment')
print('\nIMPACT: Signal is 20-50x SHORTER than expected segment length!')
print('='*80)
