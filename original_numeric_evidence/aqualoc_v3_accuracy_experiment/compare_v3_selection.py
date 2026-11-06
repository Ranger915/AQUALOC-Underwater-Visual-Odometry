#!/usr/bin/env python3
"""Compare quality versus descriptor-first V3 policies using jointly fresh COLMAP poses.
No independent metric ground truth; do not use test set to select thresholds repeatedly.
"""
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def rotation(q):
    x,y,z,w=np.asarray(q,dtype=float)
    norm=np.linalg.norm([x,y,z,w]);
    if norm<1.e-12: raise ValueError('Zero-norm quaternion')
    x,y,z,w=np.array([x,y,z,w])/norm
    return np.array([[1-2*(y*y+z*z),2*(x*y-z*w),2*(x*z+y*w)],
                     [2*(x*y+z*w),1-2*(x*x+z*z),2*(y*z-x*w)],
                     [2*(x*z-y*w),2*(y*z+x*w),1-2*(x*x+y*y)]])


def reference(path, convention):
    df=pd.read_csv(path,sep=r'\s+|,',engine='python',header=None,comment='#')
    if df.shape[1]<8: raise ValueError('Expected frame tx ty tz qx qy qz qw')
    out={}
    for values in df.iloc[:,:8].itertuples(index=False,name=None):
        fid=int(round(float(values[0]))); p=np.asarray(values[1:4],float)
        q=np.asarray(values[4:8],float)
        if np.isfinite(p).all() and np.isfinite(q).all():
            out[fid]=p if convention=='twc' else -(rotation(q).T@p)
    return out


def trajectory(path):
    df=pd.read_csv(path)
    cols={'frame','x','y','z','tracking_success','status','match_source'}
    miss=cols-set(df)
    if miss:raise ValueError(f'{path}: missing {sorted(miss)}')
    for col in ['frame','x','y','z','tracking_success']:
        df[col]=pd.to_numeric(df[col],errors='coerce')
    if df['frame'].isna().any() or df['frame'].duplicated().any():
        raise ValueError(f'{path}: invalid or repeated frame IDs')
    df=df.sort_values('frame').reset_index(drop=True)
    fresh={int(row.frame):np.array([row.x,row.y,row.z],float) for row in df.itertuples()
           if row.tracking_success==1 and np.isfinite([row.x,row.y,row.z]).all()}
    return df,fresh


def fit_sim3(A,B):
    if len(A)<5:raise ValueError('Require >=5 alignment poses')
    muA=A.mean(axis=0);muB=B.mean(axis=0)
    a=A-muA;b=B-muB
    spread=np.sum(a*a)/len(A)
    if spread<1e-10:raise ValueError('Degenerate estimated trajectory')
    U,D,Vt=np.linalg.svd(b.T@a/len(A));S=np.eye(3)
    if np.linalg.det(U@Vt)<0:S[-1,-1]=-1
    R=U@S@Vt;s=np.sum(D*np.diag(S))/spread
    if not np.isfinite(s) or s<=0:raise ValueError('Invalid Sim3 scale')
    t=muB-s*(R@muA)
    return s,R,t


def transform(X,model):
    s,R,t=model
    return (s*(R@X.T)).T+t


def plot_status(df,out):
    fig,ax=plt.subplots(figsize=(11,3))
    values=df['tracking_success'].fillna(0).to_numpy(dtype=float)
    ax.step(df['frame'],values,where='mid');ax.set_ylim(-0.1,1.15)
    ax.set_yticks([0,1],['Rejected','Accepted']);ax.set_xlabel('AQUALOC frame')
    ax.set_title('Fresh translation estimates (rejected frames are not new poses)')
    fig.tight_layout();fig.savefig(out,dpi=300);plt.close(fig)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',required=True)
    p.add_argument('--reference-convention',required=True,choices=['twc','tcw'])
    p.add_argument('--reference')
    p.add_argument('--min-train',type=int,default=5)
    p.add_argument('--min-test',type=int,default=3)
    p.add_argument('--out')
    args=p.parse_args()
    root=Path(args.root);paper=root/'paper_research_draft'
    refpath=Path(args.reference) if args.reference else root/'colmap_traj_sequence_6.txt'
    paths={
        'quality':paper/'vo_pilot_dual_source'/'vo_sequence6_orb_clahe_v3.csv',
        'descriptor_first':paper/'vo_pilot_descriptor_first'/'vo_sequence6_orb_clahe_v3.csv',
    }
    out=Path(args.out) if args.out else paper/'vo_selection_comparison'
    out.mkdir(parents=True,exist_ok=True)
    ref=reference(refpath,args.reference_convention)
    D={};T={}
    for name,path in paths.items():
        if not path.is_file():raise FileNotFoundError(str(path))
        D[name],T[name]=trajectory(path)
        d=D[name]
        if 'descriptor_valid' in d.columns and 'flow_valid' in d.columns:
            dv=pd.to_numeric(d['descriptor_valid'],errors='coerce').fillna(0).eq(1)
            fv=pd.to_numeric(d['flow_valid'],errors='coerce').fillna(0).eq(1)
            pd.DataFrame([{'descriptor_valid':int(dv.sum()),'flow_valid':int(fv.sum()),
                'both_valid':int((dv&fv).sum()),'neither_valid':int((~dv&~fv).sum()),
                'descriptor_only':int((dv&~fv).sum()),'flow_only':int((~dv&fv).sum())}]).to_csv(
                 out/f'{name}_candidate_diagnostics.csv',index=False)
        d['status'].value_counts(dropna=False).rename_axis('status').reset_index(name='frames').to_csv(
            out/f'{name}_rejection_counts.csv',index=False)
        pd.crosstab(d['match_source'],d['tracking_success']).to_csv(out/f'{name}_source_by_acceptance.csv')
        plot_status(d,out/f'{name}_acceptance_timeline.png')
    common=sorted(set(ref)&set(T['quality'])&set(T['descriptor_first']))
    ntrain=int(0.6*len(common));train=common[:ntrain];test=common[ntrain:]
    summary={'reference_file':str(refpath),'reference_convention_ASSUMED':args.reference_convention,
       'reference_total':len(ref),'jointly_fresh_frames':len(common),'joint_frame_ids':','.join(map(str,common)),
       'train_n':len(train),'test_n':len(test),'train_frames':','.join(map(str,train)),
       'test_frames':','.join(map(str,test))}
    for name in paths:
        df=D[name]
        summary[name+'_transitions']=int(len(df)-1)
        summary[name+'_accepted']=int(df['tracking_success'].eq(1).sum())
        summary[name+'_acceptance_rate']=float(df['tracking_success'].eq(1).sum()/max(1,len(df)-1))
        summary[name+'_fresh_reference_frames']=len(set(ref)&set(T[name]))
        summary[name+'_selected_descriptor']=int(((df['match_source']=='descriptor')&(df['tracking_success']==1)).sum())
        summary[name+'_selected_flow']=int(((df['match_source']=='flow')&(df['tracking_success']==1)).sum())
    if len(train)<args.min_train or len(test)<args.min_test:
        summary['status']='INSUFFICIENT_COMMON_COVERAGE'
        pd.DataFrame([summary]).to_csv(out/'comparison_summary.csv',index=False)
        print('[STOP] Only',len(common),'jointly fresh reference poses: train',len(train),'test',len(test))
        print('Saved coverage audit to',out)
        return
    rows=[];fig,ax=plt.subplots(figsize=(9,4.5))
    for name in paths:
        try:
            A=np.array([T[name][f] for f in train]);B=np.array([ref[f] for f in train])
            model=fit_sim3(A,B)
            est=transform(np.array([T[name][f] for f in test]),model)
            truth=np.array([ref[f] for f in test]);err=np.linalg.norm(est-truth,axis=1)
            summary[name+'_holdout_rmse']=float(np.sqrt(np.mean(err*err)))
            summary[name+'_holdout_median']=float(np.median(err))
            summary[name+'_holdout_p95']=float(np.percentile(err,95))
            summary[name+'_fit_scale']=float(model[0])
            for f,e in zip(test,err):rows.append({'frame':f,'policy':name,'error_colmap_units':float(e)})
            ax.plot(test,err,'o-',label=name)
        except ValueError as ex:
            summary[name+'_error']=str(ex)
    summary['status']='COMPUTED_EXPLORATORY' if len(rows)==2*len(test) else 'PARTIAL_FAILURE'
    pd.DataFrame([summary]).to_csv(out/'comparison_summary.csv',index=False)
    pd.DataFrame(rows).to_csv(out/'heldout_errors.csv',index=False)
    if rows:
        ax.set_xlabel('Frame ID');ax.set_ylabel('Held-out aligned position error (COLMAP units)')
        ax.set_title('Same-frame held-out Sim(3) comparison');ax.grid(alpha=.3);ax.legend()
        fig.tight_layout();fig.savefig(out/'heldout_error.png',dpi=300)
    plt.close(fig)
    print('[RESULT]',summary['status'],'jointly fresh=',len(common),'train=',len(train),'test=',len(test))
    for name in paths:
        print(name,'accepted=',summary[name+'_accepted'],
              'holdout_rmse=',summary.get(name+'_holdout_rmse','unavailable'))
    print('[CAUTION] COLMAP-derived comparison, assumed reference convention, exploratory on Sequence 06.')
    print('[SAVED]',out)

if __name__=='__main__': main()
