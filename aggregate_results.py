"""
Aggregates results_json/clean_<cfg>_s<seed>.json into mean +- std LaTeX tables
(paper/tables/main_results.tex and paper/tables/ablation.tex).
"""
import glob
import json
import os
import re
from collections import defaultdict

import numpy as np

DATASETS = ['Kvasir', 'ClinicDB', 'ColonDB', 'ETIS', 'CVC-300']
CFG_ORDER = [('baseline', r'\ding{55}', r'\ding{55}'), ('deepsup', r'\ding{55}', r'\ding{51}'),
             ('edge', r'\ding{51}', r'\ding{55}'), ('full', r'\ding{51}', r'\ding{51}')]


def load():
    res = defaultdict(list)  # cfg -> list of per-seed dicts
    for f in sorted(glob.glob('results_json/clean_*_s*.json')):
        m = re.match(r'clean_(\w+?)_s(\d+)\.json', os.path.basename(f))
        if m:
            with open(f) as fh:
                res[m.group(1)].append(json.load(fh))
    return res


def ms(runs, ds, key, scale=1.0, prec=3):
    v = np.array([r[ds][key] for r in runs]) * scale
    if len(v) == 1:
        return f'{v[0]:.{prec}f}'
    return f'{v.mean():.{prec}f}$\\pm${v.std(ddof=1):.{prec}f}'


def main():
    res = load()
    if not res:
        raise SystemExit('No results_json/clean_*.json found yet.')
    os.makedirs('paper/tables', exist_ok=True)

    print('=== Per-config summary (mean over seeds) ===')
    for cfg, runs in res.items():
        print(f'{cfg:9s} seeds={len(runs)} ' + ' '.join(
            f'{d}:{np.mean([r[d]["dice"] for r in runs]):.4f}' for d in DATASETS))

    # Ablation table: mDice / Boundary-IoU on every dataset
    lines = []
    for cfg, bc, ds_ in CFG_ORDER:
        if cfg not in res:
            continue
        runs = res[cfg]
        cells = [f'{ms(runs, d, "dice")} / {ms(runs, d, "biou")}' for d in DATASETS]
        lines.append(f'{bc} & {ds_} & ' + ' & '.join(cells) + r' \\')
    with open('paper/tables/ablation_rows.tex', 'w') as f:
        f.write('\n'.join(lines) + '\n')

    # Full B-Mamba row for the main comparison table (mDice, mIoU)
    if 'full' in res:
        runs = res['full']
        row = ' & '.join(f'{ms(runs, d, "dice")} & {ms(runs, d, "iou")}' for d in DATASETS)
        with open('paper/tables/bmamba_row.tex', 'w') as f:
            f.write(row + '\n')
        print('\nB-Mamba row (main table):\n' + row)
        print('\nBoundary metrics (full): ' + ', '.join(
            f'{d}: BIoU {ms(runs, d, "biou")}, HD95 {ms(runs, d, "hd95", prec=1)}' for d in DATASETS))
    print('\nWrote paper/tables/ablation_rows.tex and paper/tables/bmamba_row.tex')


if __name__ == '__main__':
    main()
