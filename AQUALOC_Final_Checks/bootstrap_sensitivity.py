#!/usr/bin/env python3
"""Paired dyadic block bootstrap sensitivity for mutual overlap AP-count and F1.
Resamples endpoint/keyframe blocks with multinomial weights; each unordered pair
receives product of endpoint multiplicities. This is a sensitivity procedure, not
an independence-based confidence interval, and cannot eliminate within-sequence tuning.
"""
import argparse,csv,json,math
from pathlib import Path
import numpy as np
import pandas as pd


def prep(rows):
    return rows.sort_values(['frame_id_i','frame_id_j']).reset_index(drop=True)

def score(y, s, w, threshold=8):
    pos=(y==1); pred=(s>=threshold)
    tp=float(w[pos&pred].sum());fp=float(w[(~pos)&pred].sum());fn=float(w[pos&(~pred)].sum())
    f1=(2*tp/(2*tp+fp+fn)) if (2*tp+fp+fn)>0 else float('nan')
    posw=float(w[pos].sum())
    if posw<=0:return f1,float('nan')
    order=np.argsort(-s,kind='stable'); ss=s[order]; yy=y[order]; ww=w[order]
    tpw=np.cumsum(ww*yy); allw=np.cumsum(ww)
    # AP sum at END of each tie group; tie-aware ranking, weighted by TP gains
    ends=np.r_[np.flatnonzero(ss[:-1]!=ss[1:]),len(ss)-1]
    tpend=tpw[ends]; allend=allw[ends]
    delta=np.diff(np.r_[0.,tpend]); precision=np.divide(tpend,allend,out=np.zeros_like(tpend),where=allend>0)
    ap=float(np.dot(delta,precision)/posw)
    return f1,ap

def main():
    p=argparse.ArgumentParser();p.add_argument('--root',required=True);p.add_argument('--out',required=True)
    p.add_argument('--reps',type=int,default=400);p.add_argument('--seed',type=int,default=20261105)
    p.add_argument('--blocks',default='1,5,10,20');args=p.parse_args()
    root=Path(args.root);out=Path(args.out);out.mkdir(parents=True,exist_ok=True)
    paths={
      'original':root/'raw_data/sequence6_mutual_matching_results/sequence6_mutual_feature_metrics.csv',
      'adaptive':root/'sequence6_adaptive_v2_mutual_matching_results/sequence6_mutual_feature_metrics.csv'}
    for k,path in paths.items():
        if not path.exists():raise FileNotFoundError(f'MISSING {k} {path}')
    orig=pd.read_csv(paths['original'],low_memory=False)
    adap=pd.read_csv(paths['adaptive'],low_memory=False)
    need={'detector','preprocessing','frame_id_i','frame_id_j','overlap_label','mutual_inliers_2px'}
    for k,df in (('original',orig),('adaptive',adap)):
        if not need.issubset(df.columns):raise ValueError(f'{k} missing {need-set(df.columns)}')
    rng=np.random.default_rng(args.seed)
    output=[];issues=[]
    for det in ('ORB','AKAZE_MLDB','SIFT'):
        a=prep(orig[(orig.detector==det)&(orig.preprocessing=='clahe')].copy())
        b=prep(adap[(adap.detector==det)&(adap.preprocessing=='adaptive_v2')].copy())
        ids_a=a[['frame_id_i','frame_id_j']].to_numpy(); ids_b=b[['frame_id_i','frame_id_j']].to_numpy()
        if len(a)!=14365 or len(b)!=14365 or not np.array_equal(ids_a,ids_b):
            raise ValueError(f'Pair keys mismatch for {det}; cannot perform paired bootstrap')
        ya=a.overlap_label.to_numpy(dtype=int);yb=b.overlap_label.to_numpy(dtype=int)
        if not np.array_equal(ya,yb):raise ValueError(f'Label discrepancy {det}')
        sa=a.mutual_inliers_2px.to_numpy(dtype=float);sb=b.mutual_inliers_2px.to_numpy(dtype=float)
        if not (np.isfinite(sa).all() and np.isfinite(sb).all()):raise ValueError('Nonfinite scores')
        frame_i=ids_a[:,0];frame_j=ids_a[:,1]
        for scope in ('all_pairs','local_gap_le_100','nonlocal_gap_ge_200'):
            mask=np.ones(len(a),dtype=bool) if scope=='all_pairs' else ((frame_j-frame_i)<=100 if scope=='local_gap_le_100' else (frame_j-frame_i)>=200)
            yi=ya[mask];sa0=sa[mask];sb0=sb[mask];fi=frame_i[mask];fj=frame_j[mask]
            for block in [int(x) for x in args.blocks.split(',')]:
                if block<=0:raise ValueError('Block length must be positive')
                # Each block spans this many consecutive reference keyframes (step=20 images).
                bi=(fi//20)//block;bj=(fj//20)//block
                uniq=np.unique(np.r_[bi,bj]); reindex={v:i for i,v in enumerate(uniq)}
                ii=np.array([reindex[v] for v in bi],dtype=int)
                jj=np.array([reindex[v] for v in bj],dtype=int)
                n=len(uniq); results=[]
                wf=np.ones(len(yi),dtype=float)
                f0,a0=score(yi,sa0,wf);f1,a1=score(yi,sb0,wf)
                for _ in range(args.reps):
                    counts=rng.multinomial(n,[1/n]*n).astype(float)
                    w=counts[ii]*counts[jj]
                    # all-zero/one-class effective dataset is invalid and excluded, counted
                    if np.sum(w*yi)==0 or np.sum(w*(1-yi))==0:continue
                    bf,ba=score(yi,sa0,w);cf,ca=score(yi,sb0,w)
                    if all(np.isfinite(z) for z in (bf,ba,cf,ca)):results.append((cf-bf,ca-ba))
                r=np.asarray(results,dtype=float)
                for metric,idx,observed in [('f1',0,f1-f0),('ap_count',1,a1-a0)]:
                    vals=r[:,idx] if len(r)>0 else np.array([])
                    output.append(dict(detector=det,scope=scope,block_keyframes=block,metric=metric,
                         observed_adaptive_minus_fixed=observed,reps_requested=args.reps,reps_valid=len(vals),seed=args.seed,
                         ci95_low=float(np.quantile(vals,.025)) if len(vals)>0 else float('nan'),
                         ci95_high=float(np.quantile(vals,.975)) if len(vals)>0 else float('nan'),
                         bootstrap_p_difference_positive=float(np.mean(vals>0)) if len(vals)>0 else float('nan'),
                         n_pairs=int(len(yi)),n_reference_blocks=int(n)))
    pd.DataFrame(output).to_csv(out/'bootstrap_block_sensitivity.csv',index=False)
    info={'seed':args.seed,'replicates':args.reps,'block_keyframes':args.blocks,
          'scope_definitions':'local gap <=100 raw image frames; nonlocal gap >=200 raw image frames; all pairs',
          'method':'paired dyadic endpoint block bootstrap, pair multiplicity=count(block_i)*count(block_j); percentile 95% intervals; fixed same labels and pair keys; AP uses weighted tie blocks',
          'limitations':['heuristic dependency sensitivity; not conventional unbiased bootstrap for network/dyadic data',
                         'outcome labels originate from same COLMAP-derived reconstruction',
                         'does not correct within-Sequence-06 tuning or multiple comparisons',
                         'does not validate nominal 95% coverage']}
    (out/'bootstrap_methodology.json').write_text(json.dumps(info,indent=2),encoding='utf-8')
    print('[DONE]',out/'bootstrap_block_sensitivity.csv')
    print('[VALIDATED] matching pair IDs and labels for all 3 detectors')
if __name__=='__main__':main()
