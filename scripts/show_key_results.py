#!/usr/bin/env python3
"""Read recorded held-out CSV metrics; this script does NOT train or evaluate a model."""
from pathlib import Path
import csv
ROOT = Path(__file__).resolve().parents[1]
FILE = ROOT / 'results' / 'novelty' / 'heldout_detection_results.csv'
CONDITIONS = ['voc_test', 'voc_syn_lowlight_s3', 'rtts_test', 'voc_syn_haze_s3', 'voc_syn_sandstorm_s3']
MODELS = ['original_baseline', 'original_combined', 'minority_lowlight_best']
with FILE.open(newline='', encoding='utf-8') as f:
    data = {(row['candidate'], row['clean_or_shifted']): row for row in csv.DictReader(f)}
print('Held-out detection mAP@0.5 (from recorded CSV):')
print(f"{'Condition':29} " + ' '.join(f'{x:>23}' for x in MODELS))
for condition in CONDITIONS:
    scores=[]
    for model in MODELS:
        row=data.get((model,condition))
        scores.append(f"{float(row['mAP50']):.3f}" if row else 'n/a')
    print(f"{condition:29} " + ' '.join(f'{x:>23}' for x in scores))
