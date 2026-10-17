"""
Aggregates Q1 evaluation results (results_json/q1_*.json)
and generates LaTeX tables for the paper.
"""
import glob
import json
import os
import re
from collections import defaultdict
import numpy as np

DATASETS = ['Kvasir', 'ClinicDB', 'ColonDB', 'ETIS', 'CVC-300']

def load_results():
    res = defaultdict(list)
    for f in sorted(glob.glob('results_json/q1_*.json')):
        base = os.path.basename(f)
        m = re.match(r'q1_(\w+?)_s(\d+)(.*)\.json', base)
        if m:
            cfg = m.group(1) + m.group(3) # e.g. 'full', 'full_tta', 'baseline', etc.
            with open(f) as fh:
                res[cfg].append(json.load(fh))
    return res

def format_ms(runs, ds, key, scale=1.0, prec=3):
    v = np.array([r[ds][key] for r in runs]) * scale
    if len(v) == 1:
        return f'{v[0]:.{prec}f}'
    return f'{v.mean():.{prec}f}$\\pm${v.std(ddof=1):.{prec}f}'

def main():
    res = load_results()
    print("=== Found Configurations ===")
    for k, v in res.items():
        print(f"  {k:15s} (runs={len(v)})")

    # Print summary table
    print("\n" + "="*80)
    print(f"{'Config':18s} | " + " | ".join(f"{d:8s}" for d in DATASETS) + " | Average")
    print("="*80)
    for cfg in ['baseline', 'bcssm_only', 'bpe_only', 'full', 'full_tta']:
        if cfg in res:
            runs = res[cfg]
            dices = [np.mean([r[d]['dice'] for r in runs]) for d in DATASETS]
            avg = np.mean(dices)
            row_str = f"{cfg:18s} | " + " | ".join(f"{d:.4f}  " for d in dices) + f" | {avg:.4f}"
            print(row_str)
    print("="*80)

if __name__ == '__main__':
    main()
