#!/usr/bin/env python3
"""Held-out COLMAP-relative trajectory consistency on matched V3 pilot runs.

NOT metric GT ATE, not full SLAM, not independent testing if model choices
were informed by this sequence. Align on early poses and score later poses.
"""
import argparse,csv,json,sys
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def quat_R(x,y,z,w):
    q=np.array([x,y,z,w],float);q=q/np.linalg.norm(q);x,y,z,w=q
    return np.array([[1-2*(y*y+z*z),2*(x*y-z*w),2*(x*z+y*w)],
                     [2*(x*y+z*w),1-2*(x*x+z*z),2*(y*z-x*w)],
                     [2*(x*z-y*w),2*(y*z+x*w),1-2*(x*x+y*y)]])

def read_reference(path,convention):
    ref={}
    for i,line in enumerate(open(path,encoding='utf-8-sig'),1):
        if not line.strip() or line.lstrip().startswith('#'): continue
        a=[float(v) for v in line.replace(',',' ').split()]
        if len(a)!=8: raise ValueError(f'Reference line {i}: expected 8 columns')
        frame=int(a[0]); t=np.array(a[1:4])
        if abs(frame-a[0])>1e-6: raise ValueError('Reference IDs must be integers')
        c=t if convention=='twc' else -quat_R(*a[4:]).T@t
        ref[frame]=c
    return ref

def read_vo(path):
    d={};last_success=None
    with open(path,newline='',encoding='utf-8-sig') as f:
        for r in csv.DictReader(f):
            frame=int(float(r['frame'])); yes=r['tracking_success'].strip().lower() in ('1','true')
            if yes: last_success=frame
            d[frame]=(np.array([float(r[k]) for k in ('x','y','z')]),yes,
                      frame-last_success if last_success is not None else 10**9)
    return d

def sim3(A,B):
    if len(A)<3: raise ValueError('Not enough calibration poses')
    am=A.mean(0);bm=B.mean(0);a=A-am;b=B-bm
    var=np.mean(np.sum(a*a,axis=1))
    if var<1e-12: raise ValueError('Estimated trajectory degenerate')
    U,s,Vt=np.linalg.svd(b.T@a/len(a));sgn=np.ones(3)
    if np.linalg.det(U@Vt)<0: sgn[-1]=-1
    R=U@np.diag(sgn)@Vt
    scale=np.dot(s,sgn)/var
    return float(scale),R,bm-scale*R@am

def rmse(e):return float(np.sqrt(np.mean(np.square(e))))

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',required=True)
    p.add_argument('--reference',default=None)
    p.add_argument('--reference-convention',required=True,choices=['twc','tcw'])
    p.add_argument('--max-staleness',type=int,default=5)
    p.add_argument('--min-calibration',type=int,default=8)
    p.add_argument('--min-validation',type=int,default=5)
    p.add_argument('--out',default=None)
    args=p.parse_args()
    root=Path(args.root)
    ref=read_reference(Path(args.reference) if args.reference else root/'colmap_traj_sequence_6.txt',args.reference_convention)
    experiments={
        'unit':root/'paper_research_draft'/'vo_pilot_unit',
        'gap':root/'paper_research_draft'/'vo_pilot_gap',
    }
    output=Path(args.out) if args.out else root/'paper_research_draft'/'vo_pilot_comparison'
    output.mkdir(parents=True,exist_ok=True)
    summaries=[]
    for config in ('orb_raw','orb_clahe','akaze_raw','akaze_clahe'):
        data={}
        for label,folder in experiments.items():
            path=folder/f'vo_sequence6_{config}_v3.csv'
            if not path.exists():raise FileNotFoundError(f'Missing {path}; run both pilots first')
            data[label]=read_vo(path)
        common=sorted(set(ref)&set(data['unit'])&set(data['gap']))
        fresh=[f for f in common if all(d[f][2]<=args.max_staleness for d in data.values())]
        # Same camera frames in both candidates: do not reward differences
        # merely due to altered pose availability.
        split=int(len(fresh)*.60)
        train=fresh[:split]; test=fresh[split:]
        info=dict(configuration=config,associated=len(common),fresh_common=len(fresh),
                  train=len(train),test=len(test),max_staleness=args.max_staleness,
                  reference_convention=args.reference_convention)
        if len(train)<args.min_calibration or len(test)<args.min_validation:
            info['status']='insufficient_coverage';summaries.append(info)
            print('[SKIP]',config,info);continue
        B=np.array([ref[f] for f in train]); Bt=np.array([ref[f] for f in test]);
        for label in experiments:
            A=np.array([data[label][f][0] for f in train]);At=np.array([data[label][f][0] for f in test]);
            try:
                scale,R,t=sim3(A,B)
                e=np.linalg.norm(scale*(At@R.T)+t-Bt,axis=1)
                info[f'{label}_scale']=scale
                info[f'{label}_holdout_rmse']=rmse(e)
                info[f'{label}_holdout_median']=float(np.median(e))
                plt.plot(test,e,marker='.',label=label)
            except ValueError as exc:
                info[f'{label}_error']=str(exc)
        plt.xlabel('Frame ID (held-out later 40%)');plt.ylabel('Position error (COLMAP reconstruction units)')
        plt.title(f'{config}: held-out Sim(3)-aligned errors')
        plt.legend();plt.tight_layout();plt.savefig(output/f'{config}_heldout_errors.png',dpi=220);plt.close()
        info['status']='evaluated';summaries.append(info)
        print('[EVALUATED]',config,info)
    (output/'pilot_summary.json').write_text(json.dumps(summaries,indent=2),encoding='utf-8')
    columns=list(dict.fromkeys(k for row in summaries for k in row))
    with (output/'pilot_summary.csv').open('w',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=columns);w.writeheader();w.writerows(summaries)
    print('[SAVED]',output)
    print('[CAUTION] Uses COLMAP-derived reference; global alignment fitted ONLY on earlier 60% of jointly fresh keyframes.')
    print('[CAUTION] Raw frames can contain held poses up to max-staleness; inspect tracking gaps and orientation before publication.')
    print('[CAUTION] Testing many variants on Sequence 06 makes later selection exploratory, not external validation.')
if __name__=='__main__':main()
