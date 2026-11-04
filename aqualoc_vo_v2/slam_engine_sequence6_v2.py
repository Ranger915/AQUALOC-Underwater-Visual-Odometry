#!/usr/bin/env python3
"""AQUALOC Sequence 06: calibrated monocular visual odometry diagnostic engine.

NOT a complete SLAM system. Translation scale remains arbitrary. No COLMAP
reference data is read during estimation. Outputs are VO estimates, not GT.
This is a separate version: the supplied slam_engine.py is not modified.
"""
from __future__ import annotations
import argparse
import csv
import json
import math
import time
from pathlib import Path
import cv2
import numpy as np

CAMERA = {
    'width': 968, 'height': 608,
    'fx': 543.3327734182214, 'fy': 542.398772982566,
    'cx': 489.02536042247897, 'cy': 305.38727712002805,
    'dist_coeffs': [-0.1255945656257394, 0.053221287232781606,
                    9.94070021080493e-05, 9.550660927242349e-05],
}

FIELDS = ['frame','time_sec','anchor_frame','anchor_gap','x','y','z',
          'qx','qy','qz','qw','features','anchor_features','matches',
          'essential_inliers','essential_ratio','cheirality_inliers',
          'cheirality_ratio_all','cheirality_ratio_essential',
          'median_parallax_deg','tracking_success','rotation_only',
          'status','scale_mode','preprocess_ms','detection_ms',
          'matching_ms','pose_ms','total_ms']


def q_from_R(R):
    # Numerically stable matrix-to-quaternion; output xyzw.
    m = R
    tr = float(np.trace(m))
    if tr > 0:
        s = 2.0 * math.sqrt(tr + 1.0)
        q = np.array([(m[2,1]-m[1,2])/s, (m[0,2]-m[2,0])/s,
                      (m[1,0]-m[0,1])/s, 0.25*s])
    else:
        i = int(np.argmax(np.diag(m)))
        if i == 0:
            s = 2.0*math.sqrt(max(1e-15,1.0+m[0,0]-m[1,1]-m[2,2]))
            q = np.array([0.25*s,(m[0,1]+m[1,0])/s,
                          (m[0,2]+m[2,0])/s,(m[2,1]-m[1,2])/s])
        elif i == 1:
            s = 2.0*math.sqrt(max(1e-15,1.0+m[1,1]-m[0,0]-m[2,2]))
            q = np.array([(m[0,1]+m[1,0])/s,0.25*s,
                          (m[1,2]+m[2,1])/s,(m[0,2]-m[2,0])/s])
        else:
            s = 2.0*math.sqrt(max(1e-15,1.0+m[2,2]-m[0,0]-m[1,1]))
            q = np.array([(m[0,2]+m[2,0])/s,(m[1,2]+m[2,1])/s,
                          0.25*s,(m[1,0]-m[0,1])/s])
    return (q/max(np.linalg.norm(q),1e-15)).tolist()


def make_detector(name, n):
    if name == 'ORB':
        return cv2.ORB_create(nfeatures=n),cv2.NORM_HAMMING
    if name == 'AKAZE':
        ctor = getattr(cv2,'AKAZE_create',None)
        descriptor = getattr(cv2,'AKAZE_DESCRIPTOR_MLDB',None)
        if ctor is None and hasattr(cv2,'xfeatures2d'):
            ctor = getattr(cv2.xfeatures2d,'AKAZE_create',None)
            descriptor = getattr(cv2.xfeatures2d,'AKAZE_DESCRIPTOR_MLDB',None)
        if ctor is None:
            raise RuntimeError('AKAZE unavailable in this OpenCV environment')
        kwargs = {'descriptor_type': descriptor} if descriptor is not None else {}
        return ctor(**kwargs),cv2.NORM_HAMMING
    if name == 'SIFT':
        return cv2.SIFT_create(nfeatures=n),cv2.NORM_L2
    raise ValueError(name)


def features(im,det,max_features):
    kp,des = det.detectAndCompute(im,None)
    if not kp or des is None:
        return [],None
    if len(kp)>max_features:
        keep = np.argsort([-p.response for p in kp])[:max_features]
        kp = [kp[int(k)] for k in keep]
        des = des[keep]
    return kp,des


def match(d1,d2,norm,ratio,mutual):
    if d1 is None or d2 is None or len(d1)<2 or len(d2)<2:
        return []
    matcher = cv2.BFMatcher(norm)
    pairs = matcher.knnMatch(d1,d2,k=2)
    forward = [a for p in pairs if len(p)==2 for a,b in [p] if a.distance < ratio*b.distance]
    if not mutual:
        return forward
    backwards = matcher.knnMatch(d2,d1,k=2)
    valid = {(a.trainIdx,a.queryIdx) for p in backwards if len(p)==2 for a,b in [p]
             if a.distance < ratio*b.distance}
    return [m for m in forward if (m.queryIdx,m.trainIdx) in valid]


def calibrated_points(ka,kb,matches,K,dist):
    pa = np.float64([ka[m.queryIdx].pt for m in matches]).reshape(-1,1,2)
    pb = np.float64([kb[m.trainIdx].pt for m in matches]).reshape(-1,1,2)
    return cv2.undistortPoints(pa,K,dist,P=K).reshape(-1,2),cv2.undistortPoints(pb,K,dist,P=K).reshape(-1,2)


def bearing_vectors(points,K):
    inv = np.linalg.inv(K)
    p = np.column_stack((points,np.ones(len(points))))
    ray=(inv@p.T).T
    return ray/np.linalg.norm(ray,axis=1,keepdims=True)


def estimate_motion(ka,kb,matches,K,dist,min_essential,min_cheirality,
                    min_cheirality_fraction,min_parallax,ransac_px):
    out = {'status':'few_matches','essential_inliers':0,'cheirality_inliers':0,
           'essential_ratio':0.,'cheirality_ratio_all':0.,
           'cheirality_ratio_essential':0.,'median_parallax_deg':float('nan'),
           'T':None,'R':None}
    if len(matches)<max(8,min_essential): return out
    a,b=calibrated_points(ka,kb,matches,K,dist)
    try:
        E, mask=cv2.findEssentialMat(a,b,K,method=cv2.RANSAC,prob=0.999,
                                    threshold=ransac_px,maxIters=2000)
    except cv2.error:
        out['status']='essential_error';return out
    if E is None or mask is None:
        out['status']='essential_failed';return out
    ransac=int(np.count_nonzero(mask))
    out['essential_inliers']=ransac
    out['essential_ratio']=ransac/len(matches)
    if ransac<min_essential:
        out['status']='insufficient_essential_inliers';return out
    best=None
    for start in range(0,E.shape[0],3):
        Ei=E[start:start+3]
        if Ei.shape!=(3,3):continue
        try:
            n,R,t,posemask=cv2.recoverPose(Ei,a,b,K,mask=mask.copy())
        except cv2.error:
            continue
        if best is None or n>best[0]:best=(n,R,t,posemask)
    if best is None:
        out['status']='recover_pose_failed';return out
    n,R,t,posemask=best
    n=int(n)
    out['cheirality_inliers']=n
    out['cheirality_ratio_all']=n/len(matches)
    out['cheirality_ratio_essential']=n/max(1,ransac)
    out['R']=R
    good=np.flatnonzero(posemask.reshape(-1)!=0)
    if len(good):
        rays_a=bearing_vectors(a[good],K)
        rays_b=bearing_vectors(b[good],K)
        # R maps anchor-camera rays into current-camera coordinates.
        # Measure translation-induced angular difference after compensating R.
        rb=(R.T@rays_b.T).T
        cosang=np.clip(np.sum(rays_a*rb,axis=1),-1.,1.)
        out['median_parallax_deg']=float(np.median(np.degrees(np.arccos(cosang))))
    if n<min_cheirality:
        out['status']='insufficient_cheirality_inliers';return out
    if out['cheirality_ratio_essential']<min_cheirality_fraction:
        out['status']='low_cheirality_fraction';return out
    if not np.isfinite(out['median_parallax_deg']) or out['median_parallax_deg']<min_parallax:
        out['status']='low_parallax';return out
    # All thresholds passed, translation direction is observable enough for an
    # experimental step. Its length is set to arbitrary 1 per accepted anchor.
    T=np.eye(4)
    T[:3,:3]=R
    norm_t=float(np.linalg.norm(t))
    if norm_t<1e-12:
        out['status']='zero_translation';return out
    T[:3,3]=t.reshape(3)/norm_t
    out['T']=T;out['status']='accepted'
    return out


def run(args):
    root=Path(args.root).expanduser().resolve()
    img_dir=Path(args.images).resolve() if args.images else root/'raw_data'/'images_sequence_6'
    paths=sorted(img_dir.glob('frame*.png'))
    if args.max_frames>0:paths=paths[:args.max_frames]
    if len(paths)<2:raise RuntimeError(f'Need at least 2 images in {img_dir}')
    out_dir=Path(args.out).resolve() if args.out else root/'paper_research_draft'/'trajectories_v2'
    out_dir.mkdir(parents=True,exist_ok=True)
    cfg=CAMERA.copy()
    if args.camera_json:
        override=json.loads(Path(args.camera_json).read_text(encoding='utf-8'))
        cfg.update(override)
    K=np.array([[cfg['fx'],0,cfg['cx']],[0,cfg['fy'],cfg['cy']],[0,0,1]],dtype=np.float64)
    dist=np.asarray(cfg['dist_coeffs'],dtype=np.float64).reshape(-1,1)
    clahe=cv2.createCLAHE(clipLimit=2.0,tileGridSize=(8,8))
    records=[]
    for algorithm in args.algorithms:
        for prep in args.preprocessings:
            print(f'\n[START] {algorithm} / {prep} | {len(paths)} frames',flush=True)
            det,norm=make_detector(algorithm,args.max_features)
            T_world_anchor=np.eye(4)
            T_world_current=np.eye(4)
            anchor_kp=None;anchor_desc=None;anchor_id=0
            rows=[];status_counts={}; accepted=0
            for seq_idx,path in enumerate(paths):
                start=time.perf_counter()
                # Use numeric frame IDs, not array indices, when available.
                fid=int(''.join(c for c in path.stem if c.isdigit()))
                im=cv2.imread(str(path),cv2.IMREAD_GRAYSCALE)
                if im is None:raise RuntimeError(f'Unreadable image: {path}')
                if im.shape!=(int(cfg['height']),int(cfg['width'])):
                    raise RuntimeError(f'Unexpected image shape {im.shape} for {path}; expected {cfg["height"]}x{cfg["width"]}')
                t=time.perf_counter();processed=clahe.apply(im) if prep=='clahe' else im
                preprocess_ms=(time.perf_counter()-t)*1000
                t=time.perf_counter();kp,desc=features(processed,det,args.max_features)
                detection_ms=(time.perf_counter()-t)*1000
                matches=[];diag={'status':'initial_frame','essential_inliers':0,'cheirality_inliers':0,
                    'essential_ratio':0.,'cheirality_ratio_all':0.,'cheirality_ratio_essential':0.,
                    'median_parallax_deg':float('nan'),'T':None,'R':None}
                matching_ms=0.;pose_ms=0.;tracked=0;rotation_only=0
                if seq_idx==0:
                    anchor_kp,anchor_desc=kp,desc;anchor_id=fid
                else:
                    t=time.perf_counter();matches=match(anchor_desc,desc,norm,args.ratio,args.mutual)
                    matching_ms=(time.perf_counter()-t)*1000
                    t=time.perf_counter();diag=estimate_motion(anchor_kp,kp,matches,K,dist,
                        args.min_essential_inliers,args.min_cheirality_inliers,
                        args.min_cheirality_fraction,args.min_parallax_deg,args.ransac_px)
                    pose_ms=(time.perf_counter()-t)*1000
                    if diag['T'] is not None:
                        T_world_current=T_world_anchor @ np.linalg.inv(diag['T'])
                        # Anchor changes only after accepted motion; avoid repeated
                        # integration of relative motions from the same anchor.
                        T_world_anchor=T_world_current.copy()
                        anchor_kp,anchor_desc=kp,desc;anchor_id=fid
                        tracked=1;accepted+=1
                    elif fid-anchor_id>=args.max_anchor_gap:
                        # Restart feature reference without inventing a camera pose.
                        # This creates an unknown/discontinuous interval, logged.
                        T_world_anchor=T_world_current.copy()
                        anchor_kp,anchor_desc=kp,desc;anchor_id=fid
                        diag['status']='reanchored_after_'+diag['status']
                status_counts[diag['status']]=status_counts.get(diag['status'],0)+1
                pos=T_world_current[:3,3]
                q=q_from_R(T_world_current[:3,:3])
                rows.append(dict(frame=fid,time_sec=seq_idx/20.0,anchor_frame=anchor_id,
                    anchor_gap=fid-anchor_id,x=pos[0],y=pos[1],z=pos[2],
                    qx=q[0],qy=q[1],qz=q[2],qw=q[3],features=len(kp),
                    anchor_features=len(anchor_kp) if anchor_kp is not None else 0,
                    matches=len(matches),essential_inliers=diag['essential_inliers'],
                    essential_ratio=diag['essential_ratio'],
                    cheirality_inliers=diag['cheirality_inliers'],
                    cheirality_ratio_all=diag['cheirality_ratio_all'],
                    cheirality_ratio_essential=diag['cheirality_ratio_essential'],
                    median_parallax_deg=diag['median_parallax_deg'],
                    tracking_success=tracked,rotation_only=rotation_only,
                    status=diag['status'],scale_mode='monocular_arbitrary',
                    preprocess_ms=preprocess_ms,detection_ms=detection_ms,
                    matching_ms=matching_ms,pose_ms=pose_ms,
                    total_ms=(time.perf_counter()-start)*1000))
                if (seq_idx+1)%250==0:
                    print(f'  {seq_idx+1}/{len(paths)} | accepted {accepted}',flush=True)
            name=f'vo_sequence6_{algorithm.lower()}_{prep}_v2.csv'
            path=out_dir/name
            with path.open('w',encoding='utf-8',newline='') as f:
                w=csv.DictWriter(f,fieldnames=FIELDS);w.writeheader();w.writerows(rows)
            keyframes=out_dir/name.replace('.csv','_keyframes.csv')
            with keyframes.open('w',encoding='utf-8',newline='') as f:
                w=csv.DictWriter(f,fieldnames=FIELDS);w.writeheader()
                w.writerows(r for r in rows if int(r['frame'])%20==0)
            result=dict(algorithm=algorithm,preprocessing=prep,frames=len(rows),
                        accepted=accepted,success_rate=accepted/max(1,len(rows)-1),
                        statuses=status_counts,output=str(path),keyframe_output=str(keyframes),
                        calibration=cfg,parameters={k:v for k,v in vars(args).items() if k not in ('root','images','out')})
            records.append(result)
            print(f'[SAVED] {path} | accepted {accepted}/{len(rows)-1} ({100*result["success_rate"]:.1f}%)')
            print('[REJECTIONS]',status_counts)
    (out_dir/'run_summary.json').write_text(json.dumps(records,indent=2,default=str),encoding='utf-8')
    print(f'\n[COMPLETE] Outputs in {out_dir}. No metric-scale or full-SLAM claim.')


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',required=True)
    p.add_argument('--images',default=None,help='Override sequence 6 image directory')
    p.add_argument('--out',default=None,help='Output directory; defaults to paper_research_draft/trajectories_v2')
    p.add_argument('--camera-json',default=None,help='Optional exact Sequence 06 calibration override JSON')
    p.add_argument('--max-frames',type=int,default=100,help='Default 100 (diagnostics); 0=full sequence')
    p.add_argument('--algorithms',nargs='+',choices=['ORB','AKAZE','SIFT'],default=['ORB','AKAZE'])
    p.add_argument('--preprocessings',nargs='+',choices=['raw','clahe'],default=['raw','clahe'])
    p.add_argument('--max-features',type=int,default=1500)
    p.add_argument('--ratio',type=float,default=0.75)
    p.add_argument('--mutual',action='store_true',help='Require symmetric Lowe-ratio matches')
    p.add_argument('--ransac-px',type=float,default=1.5)
    p.add_argument('--min-essential-inliers',type=int,default=20)
    p.add_argument('--min-cheirality-inliers',type=int,default=12)
    p.add_argument('--min-cheirality-fraction',type=float,default=0.10,
                   help='Denominator is essential-RANSAC inliers, not all descriptor matches')
    p.add_argument('--min-parallax-deg',type=float,default=0.1)
    p.add_argument('--max-anchor-gap',type=int,default=20)
    a=p.parse_args()
    if a.max_frames<0 or a.max_anchor_gap<1: p.error('invalid max frames / anchor gap')
    if not 0<a.ratio<1 or not 0<=a.min_cheirality_fraction<=1:
        p.error('invalid ratio or cheirality fraction')
    run(a)

if __name__=='__main__':main()
