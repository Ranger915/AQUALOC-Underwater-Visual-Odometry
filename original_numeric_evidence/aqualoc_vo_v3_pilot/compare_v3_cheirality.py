#!/usr/bin/env python3
"""Compare V3 cheirality thresholds using jointly fresh COLMAP keyframes.
No reference information is used by the estimator. Exploratory, not independent GT ATE.
"""
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def quat_rotation(q):
    x, y, z, w = (float(v) for v in q)
    n = np.linalg.norm([x, y, z, w])
    if n < 1e-12:
        raise ValueError('Reference quaternion has zero norm')
    x, y, z, w = np.array([x, y, z, w]) / n
    return np.array([
        [1-2*(y*y+z*z), 2*(x*y-z*w), 2*(x*z+y*w)],
        [2*(x*y+z*w), 1-2*(x*x+z*z), 2*(y*z-x*w)],
        [2*(x*z-y*w), 2*(y*z+x*w), 1-2*(x*x+y*y)]])


def load_reference(path, convention):
    table = pd.read_csv(path, sep=r'[\s,]+', engine='python', header=None, comment='#')
    if table.shape[1] < 8:
        raise ValueError('Reference requires frame tx ty tz qx qy qz qw columns')
    ref = {}
    for row in table.itertuples(index=False, name=None):
        frame = int(round(float(row[0])))
        t = np.asarray(row[1:4], dtype=float)
        q = np.asarray(row[4:8], dtype=float)
        if not np.all(np.isfinite(t)) or not np.all(np.isfinite(q)):
            continue
        ref[frame] = t if convention == 'twc' else -quat_rotation(q).T @ t
    return ref


def load_vo(path):
    df = pd.read_csv(path)
    required = {'frame', 'x', 'y', 'z', 'tracking_success'}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f'{path}: missing columns {sorted(missing)}')
    if df['frame'].duplicated().any():
        raise ValueError(f'{path}: duplicate frame identifiers')
    df = df.copy()
    for c in ['frame', 'x', 'y', 'z', 'tracking_success']:
        df[c] = pd.to_numeric(df[c], errors='coerce')
    df = df.dropna(subset=['frame', 'x', 'y', 'z', 'tracking_success'])
    poses = {}
    for row in df.itertuples(index=False):
        f = int(row.frame)
        if int(row.tracking_success) == 1:
            poses[f] = np.array([row.x, row.y, row.z], dtype=float)
    return df, poses


def sim3_fit(A, B):
    if len(A) < 3:
        raise ValueError('Sim3 needs at least 3 points')
    Ac = A - A.mean(axis=0)
    Bc = B - B.mean(axis=0)
    variance = np.sum(Ac**2) / len(A)
    if variance < 1e-12:
        raise ValueError('Degenerate estimated trajectory: no position variance')
    cov = Bc.T @ Ac / len(A)
    U, singular, Vt = np.linalg.svd(cov)
    diag = np.eye(3)
    if np.linalg.det(U @ Vt) < 0:
        diag[-1,-1] = -1
    R = U @ diag @ Vt
    scale = np.sum(singular * np.diag(diag)) / variance
    if not np.isfinite(scale) or scale <= 0:
        raise ValueError('Invalid Sim3 scale')
    t = B.mean(axis=0) - scale*(R @ A.mean(axis=0))
    return scale, R, t


def aligned(X, model):
    s, R, t = model
    return s*(R @ X.T).T + t


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root', required=True)
    p.add_argument('--reference-convention', required=True, choices=('twc','tcw'))
    p.add_argument('--reference')
    p.add_argument('--min-train', type=int, default=5)
    p.add_argument('--min-test', type=int, default=3)
    p.add_argument('--out')
    a = p.parse_args()
    root = Path(a.root)
    reference_path = Path(a.reference) if a.reference else root/'colmap_traj_sequence_6.txt'
    ref = load_reference(reference_path, a.reference_convention)
    base = root/'paper_research_draft'
    paths = {
      'threshold_0p06': base/'vo_pilot_dual_source'/'vo_sequence6_orb_clahe_v3.csv',
      'threshold_0p03': base/'vo_pilot_cheirality_ablation'/'vo_sequence6_orb_clahe_v3.csv'
    }
    data, tracks = {}, {}
    for label, path in paths.items():
        if not path.is_file(): raise FileNotFoundError(str(path))
        data[label], tracks[label] = load_vo(path)
    common = sorted(set(ref).intersection(*(set(v) for v in tracks.values())))
    out = Path(a.out) if a.out else base/'cheirality_threshold_comparison'
    out.mkdir(parents=True, exist_ok=True)
    audit = {'reference_file': str(reference_path), 'reference_convention_assumed': a.reference_convention,
             'reference_frames': len(ref), 'jointly_fresh_frames': len(common),
             'common_frame_ids': ','.join(map(str,common))}
    for label in paths:
        audit[label+'_accepted_transitions'] = int((data[label]['tracking_success']==1).sum())
        audit[label+'_fresh_reference_frames'] = len(set(ref).intersection(tracks[label]))
    split = int(len(common)*0.6)
    train, test = common[:split], common[split:]
    audit['train_n'], audit['test_n'] = len(train), len(test)
    audit['train_frame_ids'] = ','.join(map(str,train))
    audit['test_frame_ids'] = ','.join(map(str,test))
    if len(train)<a.min_train or len(test)<a.min_test:
        audit['status']='INSUFFICIENT_COMMON_COVERAGE'
        pd.DataFrame([audit]).to_csv(out/'comparison_summary.csv', index=False)
        print('[STOP] Insufficient jointly fresh reference poses for a defensible held-out comparison.')
        print('Common:',len(common),'train:',len(train),'test:',len(test))
        print('Wrote audit:',out/'comparison_summary.csv')
        return
    Btrain = np.array([ref[f] for f in train])
    Btest = np.array([ref[f] for f in test])
    rows=[]
    fig, ax = plt.subplots(figsize=(9,4.5))
    for label in paths:
        model = sim3_fit(np.array([tracks[label][f] for f in train]), Btrain)
        pred = aligned(np.array([tracks[label][f] for f in test]), model)
        errors = np.linalg.norm(pred-Btest,axis=1)
        audit[label+'_holdout_rmse'] = float(np.sqrt(np.mean(errors**2)))
        audit[label+'_holdout_median'] = float(np.median(errors))
        audit[label+'_holdout_p95'] = float(np.percentile(errors,95))
        audit[label+'_sim3_scale'] = float(model[0])
        for frame,e in zip(test,errors):rows.append({'frame':frame, 'method':label,'holdout_translation_error_colmap_units':e})
        ax.plot(test,errors,marker='o',label=label)
    audit['status']='COMPUTED_EXPLORATORY'
    pd.DataFrame([audit]).to_csv(out/'comparison_summary.csv',index=False)
    pd.DataFrame(rows).to_csv(out/'heldout_frame_errors.csv',index=False)
    ax.set_xlabel('AQUALOC frame ID')
    ax.set_ylabel('Aligned translation error (COLMAP reconstruction units)')
    ax.set_title('Held-out Sim(3) trajectory consistency: cheirality thresholds')
    ax.legend(); ax.grid(alpha=.3)
    fig.tight_layout(); fig.savefig(out/'heldout_error_comparison.png',dpi=300); plt.close(fig)
    print('[COMPLETE] Jointly fresh:',len(common),'train:',len(train),'test:',len(test))
    for label in paths:
        print(label,'holdout RMSE:',round(audit[label+'_holdout_rmse'],6),
              'median:',round(audit[label+'_holdout_median'],6))
    print('[CAUTION] Not independently measured ground-truth ATE. Reference convention must be verified.')
    print('Saved:',out)


if __name__=='__main__':
    main()
