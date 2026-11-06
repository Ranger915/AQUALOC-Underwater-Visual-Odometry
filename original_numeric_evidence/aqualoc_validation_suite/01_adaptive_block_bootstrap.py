#!/usr/bin/env python3
"""Paired MOVING BLOCK bootstrap across frame-1 time, not independent pair rows.
Only python standard library required. Preserves original CSV files.
"""
import argparse, csv, json, math, random, statistics
from collections import defaultdict, Counter
from pathlib import Path


def read_rows(path):
    with Path(path).open(newline='', encoding='utf-8-sig') as f:
        return list(csv.DictReader(f))

def detector(s): return s.strip().upper().replace('AKAZE_MLDB', 'AKAZE')
def val(x):
    try:
        f=float(x)
        return f if math.isfinite(f) else None
    except (ValueError, TypeError): return None

def frame_key(r):
    def fetch(*names):
        for k in names:
            if r.get(k) not in (None,''): return int(r[k])
        raise ValueError('Missing one of '+str(names))
    return (detector(r['detector']),fetch('frame_id_1','frame_a_id'),fetch('frame_id_2','frame_b_id'),fetch('frame_stride','pair_stride'))

def load(path, fixed_original=False):
    data={}
    for r in read_rows(path):
        if fixed_original and r['preprocessing'].strip().lower()!='clahe': continue
        key=frame_key(r)
        if key in data: raise ValueError(f'Duplicate key {key} in {path}')
        data[key]=r
    return data

def is_success(r): return int(r.get('status','')=='ok' and (val(r.get('ransac_inliers')) or 0)>=8)

def metrics(r):
    return {'success':float(is_success(r)),
            'ransac_inliers':val(r.get('ransac_inliers')),
            'ratio_matches':val(r.get('ratio_matches')),
            'inlier_ratio_all':val(r.get('inlier_ratio')),
            'epipolar_px':val(r.get('mean_symmetric_epipolar_error_px')) if is_success(r) else None}

def calc(rows, indices, metric):
    # Mean paired adaptive-minus-control difference. For epipolar, condition
    # on both methods succeeding and both errors being defined.
    pairs=[(rows[i][0][metric],rows[i][1][metric]) for i in indices]
    pairs=[(a,b) for a,b in pairs if a is not None and b is not None]
    return (sum(a-b for a,b in pairs)/len(pairs),len(pairs)) if pairs else (None,0)

def percentile_sorted(x,p):
    if not x: return None
    z=(len(x)-1)*p; lo=int(z); hi=min(lo+1,len(x)-1)
    return x[lo]*(hi-z)+x[hi]*(z-lo)

def bootstrap_group(rows, block, reps, seed):
    rng=random.Random(seed)
    # Time index is the FIRST frame in each pair. Resample contiguous temporal
    # anchor blocks, keeping all pair-stride entries from the selected anchors.
    by_frame=defaultdict(list)
    for i,(a,b,k) in enumerate(rows): by_frame[k[1]].append(i)
    frames=sorted(by_frame)
    if not frames:return []
    # Preserves meaningful temporal blocks using integer frame IDs, including
    # any missing frame IDs; circular blocks to avoid edge undercoverage.
    metrics_names=('success','ransac_inliers','ratio_matches','inlier_ratio_all','epipolar_px')
    all_indices=list(range(len(rows)))
    observed={m:calc(rows,all_indices,m) for m in metrics_names}
    samples={m:[] for m in metrics_names}
    length=len(frames)
    draws=math.ceil(length/block)
    for _ in range(reps):
        selected=[]
        for __ in range(draws):
            start=rng.randrange(length)
            selected.extend(frames[(start+j)%length] for j in range(block))
        # Equal frame-anchor representation; last block trimmed to N anchors.
        chosen=[i for f in selected[:length] for i in by_frame[f]]
        for m in metrics_names:
            v,n=calc(rows,chosen,m)
            if v is not None:samples[m].append(v)
    output=[]
    for m in metrics_names:
        estimate,n=observed[m]
        draws=sorted(samples[m])
        output.append({'metric':m,'paired_mean_difference':estimate,'n_valid_pairs':n,
                       'bootstrap_replicates_valid':len(draws),
                       'ci95_lower':percentile_sorted(draws,.025),
                       'ci95_upper':percentile_sorted(draws,.975),
                       'bootstrap_probability_positive':sum(v>0 for v in draws)/len(draws) if draws else None,
                       'conditioning':('both methods harmonized-successful' if m=='epipolar_px' else 'all matched pairs')})
    return output

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--fixed',required=True);p.add_argument('--adaptive',required=True)
    p.add_argument('--original',default='');p.add_argument('--output',required=True)
    p.add_argument('--reps',type=int,default=1000);p.add_argument('--blocks',type=int,nargs='+',default=[10,50,100])
    p.add_argument('--seed',type=int,default=606)
    args=p.parse_args()
    if args.reps<10 or any(n<1 for n in args.blocks):raise ValueError('Invalid bootstrap settings')
    fixed=load(args.fixed);adapt=load(args.adaptive)
    if set(fixed)!=set(adapt):
        raise ValueError(f'Pair sets differ: only fixed={len(set(fixed)-set(adapt))}; only adaptive={len(set(adapt)-set(fixed))}')
    if len(fixed)!=30387:print('NOTE: number of pairs differs from full Sequence 06 expected 30387')
    if args.original:
        orig=load(args.original,True)
        if set(orig)!=set(fixed):raise ValueError('Original and fixed control pair keys differ')
        for det in ('ORB','AKAZE','SIFT'):
            keys=[k for k in fixed if k[0]==det]
            mm=sum(orig[k].get('ratio_matches')!=fixed[k].get('ratio_matches') for k in keys)
            print(f'Original/control ratio-match mismatches {det}: {mm}/{len(keys)}')
    output=Path(args.output);output.mkdir(parents=True,exist_ok=True)
    rows=[]
    keys=sorted(fixed)
    normalized=[(metrics(adapt[k]),metrics(fixed[k]),k) for k in keys]
    for det in ('ORB','AKAZE','SIFT'):
        for scope in ('all','stride_1','stride_5','stride_20'):
            subset=[r for r in normalized if r[2][0]==det and (scope=='all' or r[2][3]==int(scope.split('_')[1]))]
            if not subset:continue
            for block in args.blocks:
                for metric in bootstrap_group(subset,block,args.reps,args.seed+block+len(subset)):
                    rows.append({'detector':det,'scope':scope,'block_frames':block,'replicates_requested':args.reps,
                                 'difference':'adaptive_v2_minus_fixed_control','pair_count':len(subset),**metric})
                print(f'[DONE] {det} {scope} block={block} n={len(subset)}',flush=True)
    dest=output/'adaptive_block_bootstrap.csv'
    with dest.open('w',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=rows[0].keys());w.writeheader();w.writerows(rows)
    (output/'bootstrap_metadata.json').write_text(json.dumps({'method':'circular moving-block bootstrap on chronological frame_1 anchors; paired adaptive minus fixed; same sampled anchors for all strides','block_lengths_frame_anchors':args.blocks,'reps':args.reps,'seed':args.seed,'harmonized_success':'status ok AND >=8 RANSAC inliers','epipolar_estimand':'conditional mean on common successful pairs; selection bias possible','limitations':['Not independent sequences; within-sequence time-block uncertainty only','Blocks cannot replace external sequence generalization','Original and control SIFT differences should be disclosed','Frame-1-anchor bootstrap retains all chosen stride pairs but repeated second-frame dependence may extend beyond block']},indent=2),encoding='utf-8')
    print('[SAVED]',dest)
    print('PRIMARY: report block 100, include 10 and 50 as sensitivity analyses.')

if __name__=='__main__':main()
