"""Shared, read-only helpers for AQUALOC paper figures."""
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

DETECTORS = ['ORB', 'AKAZE_MLDB', 'SIFT']
PREPS = ['raw', 'clahe', 'adaptive_v2']
SCOPES = ['all_pairs', 'local_gap_le_100', 'nonlocal_gap_ge_200']

def style():
    plt.rcParams.update({'figure.dpi': 120, 'savefig.dpi': 300,
        'font.size': 10, 'axes.titlesize': 12, 'axes.labelsize': 10,
        'axes.spines.top': False, 'axes.spines.right': False,
        'grid.alpha': .18, 'pdf.fonttype': 42, 'svg.fonttype': 'none'})

def export(fig, out, name):
    out = Path(out); out.mkdir(parents=True, exist_ok=True)
    fig.savefig(out / (name + '.png'), bbox_inches='tight', dpi=300, facecolor='white')
    plt.close(fig)
    print('[PNG]', out / (name + '.png'))

def read(path):
    if not Path(path).is_file():
        raise FileNotFoundError(f'Missing source: {path}')
    return pd.read_csv(path, low_memory=False)

def locate(root, *candidates):
    for relative in candidates:
        path = Path(root) / relative
        if path.is_file(): return path
    raise FileNotFoundError('None of these source files exists:\n' + '\n'.join(str(Path(root)/x) for x in candidates))

def summary(root, kind='mutual'):
    folder = 'sequence6_mutual_matching_results' if kind == 'mutual' else 'sequence6_calibrated_feature_results'
    name = 'sequence6_mutual_feature_summary.csv' if kind == 'mutual' else 'sequence6_reference_feature_summary.csv'
    apath = ('sequence6_adaptive_v2_mutual_matching_results' if kind == 'mutual' else 'sequence6_adaptive_v2_calibrated_reference_results') + '/' + name
    base = locate(root, f'raw_data/{folder}/{name}')
    adaptive = locate(root, apath)
    df = pd.concat([read(base), read(adaptive)], ignore_index=True)
    needed = ['detector', 'preprocessing', 'scope', 'sampson_threshold_px', 'f1']
    needed.append('minimum_mutual_inliers' if kind == 'mutual' else 'minimum_reference_inliers')
    require(df, needed, str(base))
    df['preprocessing'] = df['preprocessing'].astype(str).str.lower()
    for k in ['sampson_threshold_px','minimum_mutual_inliers' if kind=='mutual' else 'minimum_reference_inliers']:
        df[k] = pd.to_numeric(df[k], errors='raise')
    return df

def require(df, cols, source):
    absent = [c for c in cols if c not in df.columns]
    if absent: raise ValueError(f'{source}: missing expected columns {absent}. Present: {list(df.columns)}')

def point(df, detector, preprocessing, metric, scope='all_pairs', threshold=2, minimum=8, kind='mutual'):
    mincol = 'minimum_mutual_inliers' if kind == 'mutual' else 'minimum_reference_inliers'
    require(df, [metric], 'summary')
    x = df[(df.detector == detector) & (df.preprocessing == preprocessing) & (df.scope == scope) &
           np.isclose(df.sampson_threshold_px, threshold) & (df[mincol] == minimum)]
    if len(x) != 1:
        raise ValueError(f'Expected exactly 1 operating point, got {len(x)} for {kind}/{detector}/{preprocessing}/{scope}/{threshold}/{minimum}/{metric}')
    return float(x.iloc[0][metric])

def output(root, category):
    p = Path(root) / 'paper_research_draft' / 'figures' / category
    p.mkdir(parents=True, exist_ok=True)
    return p

def warn(msg): print('[SKIP]', msg)
