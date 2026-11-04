#!/usr/bin/env python3
"""AQUALOC Sequence 06 VO V3: coverage-aware COLMAP-relative trajectory evaluation.

No ground-truth claim: COLMAP is a reconstruction-derived reference. No pose
convention inference from numbers; --reference-convention is required.
"""
import argparse
import csv
import json
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def quaternion_rotation(q):
    x, y, z, w = map(float, q)
    norm = np.linalg.norm([x,y,z,w])
    if norm < 1e-12: raise ValueError('Zero quaternion')
    x,y,z,w = np.array([x,y,z,w])/norm
    return np.array([
        [1-2*(y*y+z*z), 2*(x*y-z*w), 2*(x*z+y*w)],
        [2*(x*y+z*w), 1-2*(x*x+z*z), 2*(y*z-x*w)],
        [2*(x*z-y*w), 2*(y*z+x*w), 1-2*(x*x+y*y)]
    ])


def read_reference(path, convention):
    # Dataset text: frame_id tx ty tz qx qy qz qw, whitespace or CSV.
    out={}
    with path.open(encoding='utf-8-sig') as f:
        for lnum, line in enumerate(f,1):
            line=line.strip()
            if not line or line.startswith('#'): continue
            parts=line.replace(',', ' ').split()
            if len(parts)<8:
                raise ValueError(f'Expected 8 columns, line {lnum}: {line[:100]}')
            vals=list(map(float,parts[:8]))
            fid=int(round(vals[0]))
            if abs(fid-vals[0])>1e-6: raise ValueError('Reference first column is not integral frame ID')
            t=np.array(vals[1:4]); rot=quaternion_rotation(vals[4:8]);
            # Twc: t = camera center in reference world coordinates.
            # Tcw: t = world-to-camera translation: Cw = -Rcw.T @ tcw.
            center=t if convention=='twc' else -(rot.T @ t)
            if fid in out: raise ValueError(f'Duplicate reference frame ID {fid}')
            out[fid]=center
    if len(out)<3: raise ValueError('Reference contains fewer than 3 poses')
    return out


def sim3_fit(est, ref):
    if est.shape[0]<3: raise ValueError('Need at least three alignment poses')
    mu_e=est.mean(axis=0); mu_r=ref.mean(axis=0)
    e=est-mu_e; r=ref-mu_r
    var=np.sum(e*e)/len(est)
    if var < 1e-12: raise ValueError('Degenerate estimated positions: cannot align trajectory')
    U,s,Vt=np.linalg.svd(r.T@e/len(est))
    S=np.eye(3)
    if np.linalg.det(U@Vt)<0: S[2,2]=-1
    R=U@S@Vt
    scale=float(np.sum(s*np.diag(S))/var)
    if scale<=0 or not np.isfinite(scale): raise ValueError('Invalid Sim3 scale')
    trans=mu_r-scale*R@mu_e
    return scale,R,trans


def aligned(X, fit):
    s,R,t=fit
    return (s*(R@X.T)).T+t


def summaries(values):
    a=np.asarray(values,dtype=float)
    if not len(a): return {}
    return {'rmse':float(np.sqrt(np.mean(a*a))),
            'median':float(np.median(a)),
            'mean':float(np.mean(a)),
            'p95':float(np.percentile(a,95)),
            'max':float(np.max(a)), 'n':int(len(a))}


def savefig(path):
    plt.tight_layout()
    plt.savefig(path,dpi=300,bbox_inches='tight')
    plt.close()


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',required=True)
    p.add_argument('--trajectory',default=None)
    p.add_argument('--reference',default=None)
    p.add_argument('--reference-convention',required=True,choices=['twc','tcw'])
    p.add_argument('--out',default=None)
    p.add_argument('--train-fraction',type=float,default=0.60)
    p.add_argument('--min-train',type=int,default=10)
    p.add_argument('--min-test',type=int,default=5)
    p.add_argument('--max-staleness',type=int,default=0,
                   help='For separate staleness-qualified analysis only; 0 means fresh only')
    args=p.parse_args()
    if not 0.2<args.train_fraction<0.8: p.error('--train-fraction must be between 0.2 and 0.8')
    root=Path(args.root).resolve()
    traj=Path(args.trajectory) if args.trajectory else root/'paper_research_draft'/'vo_full_sequence_v3'/'vo_sequence6_orb_clahe_v3.csv'
    refpath=Path(args.reference) if args.reference else root/'colmap_traj_sequence_6.txt'
    out=Path(args.out) if args.out else root/'paper_research_draft'/'full_sequence_trajectory_evaluation'
    for f in [traj,refpath]:
        if not f.exists(): raise FileNotFoundError(f)
    out.mkdir(parents=True,exist_ok=True)
    ref=read_reference(refpath,args.reference_convention)
    df=pd.read_csv(traj)
    required={'frame','x','y','z','tracking_success'}
    if not required.issubset(df.columns): raise ValueError(f'Missing VO CSV columns {required-set(df.columns)}')
    for c in ['frame','x','y','z','tracking_success']:
        df[c]=pd.to_numeric(df[c],errors='raise')
    if df.frame.duplicated().any(): raise ValueError('Duplicate frame IDs in trajectory')
    df['frame']=df.frame.astype(int)
    df=df.sort_values('frame').reset_index(drop=True)
    rows=df.set_index('frame')
    frames=sorted(set(ref)&set(rows.index))
    fresh=[f for f in frames if int(rows.loc[f,'tracking_success'])==1]
    # Do not treat the origin as a fresh estimated motion; it is part of coverage only.
    split=int(np.floor(len(fresh)*args.train_fraction))
    train=fresh[:split]; test=fresh[split:]
    metrics={'status':'insufficient_coverage','reference_file':str(refpath),
             'estimated_file':str(traj),'reference_convention_assumed':args.reference_convention,
             'reference_count':len(ref),'estimated_rows':len(df),'associated_frames':len(frames),
             'fresh_reference_frames':len(fresh),
             'fresh_coverage_pct':round(100*len(fresh)/len(ref),3),
             'accepted_transitions':int((df.tracking_success==1).sum()),
             'total_transitions':max(0,len(df)-1),
             'accepted_pct':float(100*(df.tracking_success==1).sum()/max(1,len(df)-1)),
             'train_n':len(train),'test_n':len(test),'train_frames':train,
             'test_frames':test,'max_staleness_optional':args.max_staleness}
    if len(train)<args.min_train or len(test)<args.min_test:
        (out/'summary.json').write_text(json.dumps(metrics,indent=2),encoding='utf8')
        print('[INSUFFICIENT COVERAGE]',metrics)
        return
    A=np.array([rows.loc[f,['x','y','z']].to_numpy(dtype=float) for f in train])
    B=np.array([ref[f] for f in train])
    fit=sim3_fit(A,B)
    metrics['sim3_scale']=fit[0]
    def errors(fids):
        aa=np.array([rows.loc[f,['x','y','z']].to_numpy(dtype=float) for f in fids])
        bb=np.array([ref[f] for f in fids]); pred=aligned(aa,fit)
        ee=np.linalg.norm(pred-bb,axis=1)
        return aa,bb,pred,ee
    _,_,_,etr=errors(train)
    _,_,_,ete=errors(test)
    _,_,_,eall=errors(fresh)
    metrics['status']='COMPUTED_EXPLORATORY'
    metrics['training_error_colmap_units']=summaries(etr)
    metrics['heldout_error_colmap_units']=summaries(ete)
    metrics['all_fresh_error_colmap_units']=summaries(eall)
    pd.DataFrame({'frame':fresh,'split':['train' if f in train else 'test' for f in fresh],
                  'error_colmap_units':eall}).to_csv(out/'fresh_pose_errors.csv',index=False)
    # Optional carried-forward analysis. This is separate from accepted-pose metrics.
    if args.max_staleness>0:
        last=-10**9; fresh_at={}
        for row in df.itertuples(index=False):
            f=int(row.frame)
            if int(row.tracking_success)==1: last=f
            fresh_at[f]=f-last
        allowed=[f for f in frames if f in fresh_at and 0<=fresh_at[f]<=args.max_staleness]
        if allowed:
            _,_,_,estale=errors(allowed)
            metrics['staleness_qualified_diagnostic']={
                'max_staleness_frames':args.max_staleness,
                'associated_n':len(allowed),'errors_colmap_units':summaries(estale),
                'warning':'These are held/stale estimates; not a fresh tracking accuracy metric'}
    # Coverage over all reference frames, plus first frame as untracked.
    ref_frames=sorted(ref)
    plt.figure(figsize=(13,3))
    plt.scatter(ref_frames,[0]*len(ref_frames),s=18,label='Reference frame',alpha=.4)
    plt.scatter(fresh,[1]*len(fresh),s=26,label='Fresh accepted VO')
    plt.yticks([0,1],['COLMAP reference','Fresh VO update'])
    plt.xlabel('Image frame ID'); plt.title('Reference pose availability and fresh VO coverage')
    plt.legend(loc='upper right'); savefig(out/'01_reference_coverage.png')
    # Tracking success across all image frames
    plt.figure(figsize=(13,3))
    plt.plot(df.frame,df.tracking_success,linewidth=.6)
    plt.xlabel('Image frame ID'); plt.ylabel('Accepted translation (0/1)')
    plt.title('Full-sequence VO translation-acceptance timeline')
    savefig(out/'02_tracking_timeline.png')
    # Held-out accuracy with explicit split
    plt.figure(figsize=(10,4))
    plt.plot(train,etr,'o-',label=f'Alignment segment (n={len(train)})')
    plt.plot(test,ete,'o-',label=f'Held-out segment (n={len(test)})')
    plt.axvline((train[-1]+test[0])/2,linestyle='--',color='gray')
    plt.xlabel('Frame ID'); plt.ylabel('Position residual (COLMAP units)')
    plt.title('Chronological Sim(3) fit and held-out residuals')
    plt.legend(); savefig(out/'03_heldout_position_errors.png')
    # XY XZ YZ overlays; only fresh associated reference frames to prevent carried forward artifacts
    aa=np.array([rows.loc[f,['x','y','z']].to_numpy(dtype=float) for f in fresh]);
    bb=np.array([ref[f] for f in fresh]); pp=aligned(aa,fit)
    fig,axes=plt.subplots(1,3,figsize=(15,4))
    for ax,(i,j,lab) in zip(axes,[(0,1,'XY'),(0,2,'XZ'),(1,2,'YZ')]):
        ax.plot(bb[:,i],bb[:,j],'o-',label='COLMAP reference',markersize=3)
        ax.plot(pp[:,i],pp[:,j],'.-',label='Aligned monocular VO',markersize=3)
        ax.set_xlabel(lab[0]);ax.set_ylabel(lab[1]);ax.set_title(lab)
        ax.axis('equal');ax.legend(fontsize=7)
    fig.suptitle('Only fresh reference-associated estimates; Sim(3) fit on earlier poses')
    savefig(out/'04_trajectory_projections.png')
    plt.figure(figsize=(7,4))
    plt.hist(ete,bins=min(15,max(4,len(ete)//2)))
    plt.xlabel('Held-out position residual (COLMAP units)');plt.ylabel('Number of poses')
    plt.title('Held-out Sim(3) residual distribution');savefig(out/'05_holdout_histogram.png')
    ss=np.sort(ete)
    plt.figure(figsize=(7,4));plt.step(ss,np.arange(1,len(ss)+1)/len(ss),where='post')
    plt.xlabel('Held-out position residual (COLMAP units)');plt.ylabel('Empirical CDF')
    plt.title('Held-out trajectory residual CDF');savefig(out/'06_holdout_cdf.png')
    # Rejection reasons and parallax: purely diagnostic, never a validation of pose correctness.
    if 'status' in df.columns:
        subset=df[df.tracking_success!=1]
        grouped=subset[subset.frame!=df.frame.iloc[0]].status.astype(str).value_counts()
        if len(grouped):
            plt.figure(figsize=(9,4));plt.barh(grouped.index[::-1],grouped.values[::-1]);
            plt.xlabel('Frame transitions');plt.title('VO rejection reasons')
            savefig(out/'07_rejection_reasons.png')
    if 'median_parallax_deg' in df.columns:
        vals=pd.to_numeric(df.median_parallax_deg,errors='coerce')
        vals=vals.replace([np.inf,-np.inf],np.nan)
        plt.figure(figsize=(11,4))
        plt.scatter(df.frame,vals,c=df.tracking_success,s=6,alpha=.7)
        plt.xlabel('Frame ID');plt.ylabel('Median parallax (degrees)')
        plt.title('Estimated parallax (colored by acceptance)')
        savefig(out/'08_parallax_diagnostics.png')
    (out/'summary.json').write_text(json.dumps(metrics,indent=2),encoding='utf8')
    with (out/'summary_flat.csv').open('w',newline='',encoding='utf8') as f:
        fields=['reference_count','estimated_rows','associated_frames','fresh_reference_frames',
                'fresh_coverage_pct','accepted_transitions','total_transitions','accepted_pct',
                'train_n','test_n','sim3_scale','heldout_rmse','heldout_median','heldout_p95']
        writer=csv.DictWriter(f,fieldnames=fields);writer.writeheader()
        writer.writerow({**{k:metrics.get(k) for k in fields},
                         'heldout_rmse':metrics['heldout_error_colmap_units']['rmse'],
                         'heldout_median':metrics['heldout_error_colmap_units']['median'],
                         'heldout_p95':metrics['heldout_error_colmap_units']['p95']})
    print('[COMPLETED] Full-sequence COLMAP-relative evaluation')
    print('Fresh:',len(fresh),'/',len(ref),'| train:',len(train),'| held-out:',len(test))
    print('Held-out position RMSE (COLMAP units):',metrics['heldout_error_colmap_units']['rmse'])
    print('Held-out median:',metrics['heldout_error_colmap_units']['median'])
    print('Outputs:',out)
    print('[CAUTION] Reference convention assumed:',args.reference_convention,
          '| monocular arbitrary translation scale | COLMAP-derived reference, not independently measured GT')

if __name__=='__main__':
    main()
