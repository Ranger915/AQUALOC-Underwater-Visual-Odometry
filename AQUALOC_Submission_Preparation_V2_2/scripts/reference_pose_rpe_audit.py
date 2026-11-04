#!/usr/bin/env python3
"""AQUALOC pose-convention sensitivity and fresh-pose RPE audit.
Neither choice is asserted correct; requires documentary convention resolution.
Writes separate candidate results, never silently selects lower ATE.
"""
import argparse,csv,json,math
from pathlib import Path
import numpy as np

def Rquat(q):
    x,y,z,w=np.asarray(q,dtype=float)/np.linalg.norm(q)
    return np.array([[1-2*(y*y+z*z),2*(x*y-z*w),2*(x*z+y*w)],
                     [2*(x*y+z*w),1-2*(x*x+z*z),2*(y*z-x*w)],
                     [2*(x*z-y*w),2*(y*z+x*w),1-2*(x*x+y*y)]])

def load_ref(p,conv):
    out={}
    for line in p.read_text(encoding='utf-8-sig').splitlines():
        if not line.strip() or line.lstrip().startswith('#'):continue
        a=np.array(list(map(float,line.replace(',',' ').split()[:8])))
        if len(a)!=8:raise ValueError('Malformed reference row')
        k=int(a[0]);rot=Rquat(a[4:8]);t=a[1:4]
        if conv=='tcw':rot,t=rot.T,-rot.T@t
        out[k]=(t,rot)
    return out

def load_vo(p):
    out={}; allrows=[]
    with p.open(encoding='utf-8-sig',newline='') as f:
        for r in csv.DictReader(f):
            allrows.append(r)
            if r['tracking_success']!='1':continue
            k=int(r['frame']); xyz=[float(r[j]) for j in ('x','y','z')]
            q=[float(r[j]) for j in ('qx','qy','qz','qw')]
            out[k]=(np.array(xyz),Rquat(q))
    return out,allrows

def fit(x,y):
    a=x-x.mean(0);b=y-y.mean(0);v=np.mean(np.sum(a*a,axis=1))
    if v<1e-12:raise ValueError('Degenerate trajectory')
    U,d,Vt=np.linalg.svd(b.T@a/len(a));S=np.eye(3)
    if np.linalg.det(U@Vt)<0:S[2,2]=-1
    R=U@S@Vt;s=np.sum(d*np.diag(S))/v
    if s<=0:raise ValueError('Invalid scale')
    return s,R,y.mean(0)-s*R@x.mean(0)

def error(e,r):return np.linalg.norm(e-r,axis=-1)
def angle(R):return math.degrees(math.acos(float(np.clip((np.trace(R)-1)/2,-1,1))))

def evaluate(v,ref,gap,train_frac=.6):
    ids=sorted(set(v)&set(ref))
    n=int(len(ids)*train_frac);tr=ids[:n];te=ids[n:]
    if len(tr)<10 or len(te)<5:return dict(status='insufficient',fresh=len(ids),train=len(tr),test=len(te))
    s,R,t=fit(np.array([v[k][0] for k in tr]),np.array([ref[k][0] for k in tr]))
    errs=error(np.array([s*R@v[k][0]+t for k in te]),np.array([ref[k][0] for k in te]))
    pairrows=[]
    for i in te:
        j=i+gap
        if j not in te:continue
        # Pair displacement in global frame, once aligned using ONLY training frame poses
        ev=s*R@(v[j][0]-v[i][0]);rv=ref[j][0]-ref[i][0]
        # Relative rotation independent of common global frame, assuming both pose conventions correctly decoded
        rev=v[i][1].T@v[j][1]; rref=ref[i][1].T@ref[j][1]
        pairrows.append(dict(frame_i=i,frame_j=j,translation_delta_error=float(np.linalg.norm(ev-rv)),
                             rotation_delta_error_deg=angle(rref.T@rev)))
    return dict(status='computed',fresh=len(ids),train=len(tr),test=len(te),
                heldout_rmse=float(np.sqrt(np.mean(errs**2))),heldout_median=float(np.median(errs)),
                sim3_scale=float(s),gap=gap,rpe_pairs=len(pairrows),
                rpe_translation_rmse=float(np.sqrt(np.mean([p['translation_delta_error']**2 for p in pairrows]))) if pairrows else None,
                rpe_rotation_median_deg=float(np.median([p['rotation_delta_error_deg'] for p in pairrows])) if pairrows else None,
                per_pair=pairrows)

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',type=Path,required=True)
    p.add_argument('--trajectory',type=Path)
    p.add_argument('--reference',type=Path)
    p.add_argument('--out',type=Path)
    p.add_argument('--rpe-gap',type=int,default=20)
    a=p.parse_args();root=a.root.resolve()
    v,_=load_vo(a.trajectory or root/'paper_research_draft/vo_full_sequence_v3/vo_sequence6_orb_clahe_v3.csv')
    source=a.reference or root/'colmap_traj_sequence_6.txt'
    dest=a.out or root/'paper_research_draft/submission_audit/pose_conventions'
    dest.mkdir(parents=True,exist_ok=True)
    result={}
    for conv in ('twc','tcw'):
        r=load_ref(source,conv)
        val=evaluate(v,r,a.rpe_gap)
        rows=val.pop('per_pair',[])
        result[conv]=val
        if rows:
            with (dest/f'rpe_{conv}.csv').open('w',newline='') as f:
                w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    (dest/'convention_sensitivity.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    print(json.dumps(result,indent=2))
    print('WARNING: The lowest error DOES NOT determine which convention is correct; verify source export semantics.')
if __name__=='__main__': main()
