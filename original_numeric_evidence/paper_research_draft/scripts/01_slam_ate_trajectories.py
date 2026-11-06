"""True estimated SLAM vs independent reference trajectory evaluator.

Requires actual estimated SLAM poses. DO NOT use two COLMAP exports as
independent SLAM/reference evidence. See README for accepted formats.
"""
import argparse, json
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from common import output, export, style


def quat_rot(q):
    q=np.asarray(q,float); n=np.linalg.norm(q)
    if n < 1e-12: raise ValueError('Zero-length quaternion')
    x,y,z,w=q/n
    return np.array([[1-2*(y*y+z*z),2*(x*y-z*w),2*(x*z+y*w)],
                     [2*(x*y+z*w),1-2*(x*x+z*z),2*(y*z-x*w)],
                     [2*(x*z-y*w),2*(y*z+x*w),1-2*(x*x+y*y)]])


def load(path, fmt, convention, time_unit):
    path=Path(path)
    if fmt == 'csv':
        d=pd.read_csv(path)
        names={c.lower().strip():c for c in d.columns}
        key='frame_id' if 'frame_id' in names else 'timestamp' if 'timestamp' in names else None
        if key is None: raise ValueError(f'{path}: CSV needs frame_id or timestamp column')
        required=['tx','ty','tz','qx','qy','qz','qw']
        miss=[n for n in required if n not in names]
        if miss: raise ValueError(f'{path}: missing CSV fields {miss}')
        ids=pd.to_numeric(d[names[key]]).to_numpy(float)
        pos=d[[names[n] for n in ['tx','ty','tz']]].to_numpy(float)
        q=d[[names[n] for n in ['qx','qy','qz','qw']]].to_numpy(float)
    elif fmt in ('tum','frame_tquat'):
        rows=[]
        for s in path.read_text(encoding='utf-8-sig').splitlines():
            s=s.strip()
            if not s or s.startswith('#'): continue
            parts=s.replace(',',' ').split()
            if len(parts) < 8: raise ValueError(f'{path}: expected 8 columns, got {len(parts)}: {s[:90]}')
            rows.append([float(x) for x in parts[:8]])
        a=np.asarray(rows,float)
        if a.ndim!=2 or a.shape[1]!=8: raise ValueError(f'{path}: empty/invalid trajectory')
        ids,pos,q=a[:,0],a[:,1:4],a[:,4:8]
        key='timestamp' if fmt=='tum' else 'frame_id'
    elif fmt=='kitti':
        a=np.loadtxt(path)
        if a.ndim==1: a=a[None,:]
        if a.shape[1]!=12: raise ValueError('KITTI expects one row of 12 pose matrix entries per frame')
        T=a.reshape(-1,3,4)
        ids=np.arange(len(T),dtype=float)
        # KITTI convention denotes T_wc or T_cw; convert translation below
        pos=T[:,:,3]
        q=None
        key='frame_id'
    else: raise ValueError(fmt)
    if key=='timestamp':
        ids=ids * {'s':1.0, 'ms':1e-3, 'us':1e-6, 'ns':1e-9}[time_unit]
    if convention=='tcw':
        if fmt=='kitti': pos=np.array([-T[i,:,:3].T@pos[i] for i in range(len(pos))])
        else: pos=np.array([-quat_rot(qi).T@ti for qi,ti in zip(q,pos)])
    if not np.isfinite(pos).all() or not np.isfinite(ids).all(): raise ValueError(f'{path}: NaN/Inf poses')
    if len(np.unique(ids)) != len(ids): raise ValueError(f'{path}: duplicate IDs/timestamps')
    return key,ids,pos


def associate(k1,id1,p1,k2,id2,p2,mode,tol,offset):
    if mode=='auto':
        if k1!=k2: raise ValueError(f'Cannot auto associate {k1} and {k2}; supply matching IDs or timestamp exports')
        mode=k1
    if mode=='frame_id':
        if k1!='frame_id' or k2!='frame_id': raise ValueError('Frame association requires frame IDs in both files')
        m={v:i for i,v in enumerate(id2)}
        ii=[i for i,v in enumerate(id1) if v in m]; jj=[m[id1[i]] for i in ii]
    elif mode=='timestamp':
        if k1!='timestamp' or k2!='timestamp': raise ValueError('Timestamp association requires TUM/CSV timestamps in both files')
        # 1-to-1 chronological greedy matching; offset shifts EST timestamps
        e=np.argsort(id1); r=np.argsort(id2)
        xx=id1[e]+offset; yy=id2[r]; i=j=0; ii=[]; jj=[]
        while i<len(e) and j<len(r):
            dif=xx[i]-yy[j]
            if abs(dif)<=tol:
                ii.append(e[i]);jj.append(r[j]);i+=1;j+=1
            elif dif < -tol: i+=1
            else: j+=1
    else: raise ValueError(mode)
    if len(ii)<3: raise ValueError(f'Only {len(ii)} matched poses; need >=3; check frame numbering, units, offsets, tolerance')
    return np.asarray(ii),np.asarray(jj)


def align(src,dst,similarity):
    # Umeyama alignment of centered 3D points
    a=src.mean(axis=0);b=dst.mean(axis=0); X=src-a;Y=dst-b
    U,s,Vt=np.linalg.svd(Y.T@X/len(src)); D=np.diag([1,1,np.linalg.det(U@Vt)])
    R=U@D@Vt
    scale=float(np.sum(s*np.diag(D))/np.mean(np.sum(X*X,axis=1))) if similarity else 1.0
    trans=b-scale*(R@a)
    return scale,R,trans


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',required=True)
    p.add_argument('--estimated',required=True,help='SLAM-generated trajectory; mandatory')
    p.add_argument('--reference',required=True,help='Reference trajectory; mandatory')
    p.add_argument('--est-format',choices=['tum','frame_tquat','csv','kitti'],default='tum')
    p.add_argument('--ref-format',choices=['tum','frame_tquat','csv','kitti'],default='tum')
    p.add_argument('--est-convention',choices=['twc','tcw'],required=True,help='twc=camera center, tcw=world-to-camera translation')
    p.add_argument('--ref-convention',choices=['twc','tcw'],required=True)
    p.add_argument('--est-time-unit',choices=['s','ms','us','ns'],default='s')
    p.add_argument('--ref-time-unit',choices=['s','ms','us','ns'],default='s')
    p.add_argument('--associate',choices=['auto','frame_id','timestamp'],default='auto')
    p.add_argument('--tolerance',type=float,default=.02,help='seconds for timestamp association')
    p.add_argument('--time-offset',type=float,default=0,help='seconds added to estimated timestamps')
    p.add_argument('--alignment',choices=['se3','sim3'],default='se3',help='se3 default for metric SLAM; sim3 for monocular unknown scale')
    args=p.parse_args();style()
    if Path(args.estimated).resolve()==Path(args.reference).resolve(): raise ValueError('Estimated and reference files must be distinct')
    k1,i1,P=load(args.estimated,args.est_format,args.est_convention,args.est_time_unit)
    k2,i2,Q=load(args.reference,args.ref_format,args.ref_convention,args.ref_time_unit)
    ix,jx=associate(k1,i1,P,k2,i2,Q,args.associate,args.tolerance,args.time_offset)
    X=P[ix];Y=Q[jx]
    scale,R,t=align(X,Y,args.alignment=='sim3')
    aligned=scale*(X@R.T)+t
    err=np.linalg.norm(aligned-Y,axis=1)
    metrics={'matched_poses':len(err),'ate_rmse':float(np.sqrt(np.mean(err**2))),
             'ate_mean':float(err.mean()),'ate_median':float(np.median(err)),
             'ate_p95':float(np.percentile(err,95)), 'ate_max':float(err.max()),
             'alignment':args.alignment,'estimated_file':str(Path(args.estimated).resolve()),
             'reference_file':str(Path(args.reference).resolve()),'sim3_scale':float(scale),
             'trajectory_units':'same as reference trajectory (verify independently)'}
    out=output(args.root,'01_true_slam_ate');(out/'ate_metrics.json').write_text(json.dumps(metrics,indent=2),encoding='utf-8')
    pd.DataFrame({'estimated_id':i1[ix],'reference_id':i2[jx], 'ref_x':Y[:,0], 'ref_y':Y[:,1], 'ref_z':Y[:,2],
                  'est_aligned_x':aligned[:,0],'est_aligned_y':aligned[:,1],'est_aligned_z':aligned[:,2],
                  'ate_translation':err}).to_csv(out/'ate_per_pose.csv',index=False)
    fig,axs=plt.subplots(1,3,figsize=(15,4.5))
    for ax,(a,b) in zip(axs,[(0,1),(0,2),(1,2)]):
        ax.plot(Y[:,a],Y[:,b],label='Reference',lw=2)
        ax.plot(aligned[:,a],aligned[:,b],label='Estimated SLAM, aligned',lw=1.3,ls='--')
        ax.set(xlabel='XYZ'[a]+' [reference units]',ylabel='XYZ'[b]+' [reference units]');ax.axis('equal');ax.grid(True);ax.legend(fontsize=8)
    fig.suptitle('SLAM vs reference trajectory (translation)')
    export(fig,out,'01_trajectory_xyz_projections')
    fig,ax=plt.subplots(figsize=(10,4));ax.plot(np.arange(len(err)),err,lw=1)
    ax.set(xlabel='Associated pose index',ylabel='Translation ATE [reference units]',title=f'ATE per pose | RMSE={metrics["ate_rmse"]:.4g}')
    ax.grid(True);export(fig,out,'02_ate_vs_pose')
    fig,ax=plt.subplots(figsize=(7,4));ax.hist(err,bins=min(40,max(5,int(np.sqrt(len(err))))),edgecolor='white')
    ax.set(xlabel='Translation ATE [reference units]',ylabel='Pose count',title='ATE distribution')
    export(fig,out,'03_ate_histogram')
    e=np.sort(err);fig,ax=plt.subplots(figsize=(7,4));ax.plot(e,np.arange(1,len(e)+1)/len(e));ax.grid(True)
    ax.set(xlabel='Translation ATE [reference units]',ylabel='Cumulative fraction',title='ATE empirical CDF')
    export(fig,out,'04_ate_cdf')
    fig=plt.figure(figsize=(8,6));ax=fig.add_subplot(111,projection='3d');ax.plot(*Y.T,label='Reference');ax.plot(*aligned.T,ls='--',label='Estimated')
    ax.set(xlabel='X',ylabel='Y',zlabel='Z',title='3D trajectory overlay');ax.legend();export(fig,out,'05_trajectory_3d')
    print('[ATE]',json.dumps(metrics,indent=2))

if __name__=='__main__':main()
