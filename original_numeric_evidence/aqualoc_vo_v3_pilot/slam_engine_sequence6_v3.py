#!/usr/bin/env python3
"""AQUALOC Sequence 06 monocular VO (V3 replacement, not metric SLAM).

Evaluates forward/backward KLT flow and descriptor matching separately on each pair,
then uses geometrically accepted candidates only; reports both outcomes.
Reports accepted translation separately from an optional rotation-only orientation
update. Never counts carried-forward or rotation-only poses as translation success.
No reference trajectory is used during estimation.
"""
from __future__ import annotations
import argparse, csv, json, math, time
from pathlib import Path
from collections import Counter
import cv2
import numpy as np

CAMERA = dict(width=968, height=608, fx=543.3327734182214,
    fy=542.398772982566, cx=489.02536042247897,
    cy=305.38727712002805,
    dist_coeffs=[-0.1255945656257394, 0.053221287232781606,
                 9.94070021080493e-05, 9.550660927242349e-05])
FIELDS = ['frame','time_sec','anchor_frame','anchor_gap','motion_step_units',
    'x','y','z','qx','qy','qz','qw','features','anchor_features','matches',
    'essential_inliers','essential_ratio','cheirality_inliers',
    'cheirality_ratio_all','cheirality_ratio_essential','median_parallax_deg',
    'tracking_success','rotation_only','status','scale_mode',
    'preprocess_ms','detection_ms','matching_ms','pose_ms','total_ms',
    'match_source','flow_matches','descriptor_matches','pose_valid',
    'pose_coverage','anchor_reset','consecutive_pose_inliers',
    'descriptor_status','flow_status','descriptor_essential_inliers',
    'flow_essential_inliers','descriptor_cheirality_inliers',
    'flow_cheirality_inliers','descriptor_parallax_deg','flow_parallax_deg',
    'descriptor_valid','flow_valid','both_valid','selection_quality',
    'estimated_anchor_gap','candidate_disagreement_deg']


def quat(R):
    # Convert proper rotation matrix into xyzw quaternion, using OpenCV Rodrigues
    # and elementary rotation-matrix cases.
    tr = float(np.trace(R))
    if tr > 0:
        s = 2 * math.sqrt(tr + 1)
        q = [(R[2,1]-R[1,2])/s,(R[0,2]-R[2,0])/s,
             (R[1,0]-R[0,1])/s, s/4]
    else:
        j = int(np.argmax(np.diag(R)))
        if j == 0:
            s = 2*math.sqrt(max(1e-14,1+R[0,0]-R[1,1]-R[2,2]))
            q = [s/4,(R[0,1]+R[1,0])/s,(R[0,2]+R[2,0])/s,(R[2,1]-R[1,2])/s]
        elif j == 1:
            s = 2*math.sqrt(max(1e-14,1+R[1,1]-R[0,0]-R[2,2]))
            q = [(R[0,1]+R[1,0])/s,s/4,(R[1,2]+R[2,1])/s,(R[0,2]-R[2,0])/s]
        else:
            s = 2*math.sqrt(max(1e-14,1+R[2,2]-R[0,0]-R[1,1]))
            q = [(R[0,2]+R[2,0])/s,(R[1,2]+R[2,1])/s,s/4,(R[1,0]-R[0,1])/s]
    q=np.asarray(q,dtype=float)
    return (q/max(1e-12,np.linalg.norm(q))).tolist()


def detector_for(name, max_features):
    if name == 'ORB':
        return cv2.ORB_create(nfeatures=max_features),cv2.NORM_HAMMING
    if name == 'AKAZE':
        ctor=getattr(cv2,'AKAZE_create',None)
        d=getattr(cv2,'AKAZE_DESCRIPTOR_MLDB',None)
        if ctor is None and hasattr(cv2,'xfeatures2d'):
            ctor=getattr(cv2.xfeatures2d,'AKAZE_create',None)
            d=getattr(cv2.xfeatures2d,'AKAZE_DESCRIPTOR_MLDB',None)
        if ctor is None: raise RuntimeError('AKAZE is unavailable in this OpenCV build')
        return ctor(**({'descriptor_type':d} if d is not None else {})),cv2.NORM_HAMMING
    if name == 'SIFT':return cv2.SIFT_create(nfeatures=max_features),cv2.NORM_L2
    raise ValueError(name)


def detect(img,det,n):
    kp,des=det.detectAndCompute(img,None)
    if not kp or des is None:return [],None
    if len(kp)>n:
        keep=np.argsort([-x.response for x in kp])[:n]
        kp=[kp[int(i)] for i in keep];des=des[keep]
    return kp,des


def descriptor_pairs(k1,d1,k2,d2,norm,ratio,mutual):
    if d1 is None or d2 is None or min(len(d1),len(d2))<2:
        return np.empty((0,2),np.float32),np.empty((0,2),np.float32)
    m=cv2.BFMatcher(norm)
    try:
        forward=[a for pair in m.knnMatch(d1,d2,k=2) if len(pair)==2
                 for a,b in [pair] if a.distance<ratio*b.distance]
        if mutual:
            back={(a.trainIdx,a.queryIdx) for pair in m.knnMatch(d2,d1,k=2)
                  if len(pair)==2 for a,b in [pair] if a.distance<ratio*b.distance}
            forward=[a for a in forward if (a.queryIdx,a.trainIdx) in back]
    except cv2.error:
        forward=[]
    a=np.array([k1[v.queryIdx].pt for v in forward],dtype=np.float32).reshape(-1,2)
    b=np.array([k2[v.trainIdx].pt for v in forward],dtype=np.float32).reshape(-1,2)
    return a,b


def flow_pairs(img1,img2,points,fb_px):
    if len(points)<8:
        z=np.empty((0,2),np.float32);return z,z
    src=np.asarray(points,np.float32).reshape(-1,1,2)
    lk=dict(winSize=(25,25),maxLevel=4,
            criteria=(cv2.TERM_CRITERIA_EPS|cv2.TERM_CRITERIA_COUNT,35,0.01))
    dst,s1,_=cv2.calcOpticalFlowPyrLK(img1,img2,src,None,**lk)
    if dst is None:
        z=np.empty((0,2),np.float32);return z,z
    back,s2,_=cv2.calcOpticalFlowPyrLK(img2,img1,dst,None,**lk)
    if back is None:
        z=np.empty((0,2),np.float32);return z,z
    err=np.linalg.norm(back.reshape(-1,2)-src.reshape(-1,2),axis=1)
    h,w=img2.shape
    xy=dst.reshape(-1,2)
    valid=(s1.ravel()>0)&(s2.ravel()>0)&(err<fb_px)&np.isfinite(err)
    valid&=(xy[:,0]>=2)&(xy[:,0]<w-2)&(xy[:,1]>=2)&(xy[:,1]<h-2)
    return src.reshape(-1,2)[valid],xy[valid]


def flow_points(img,max_points):
    pts=cv2.goodFeaturesToTrack(img,maxCorners=max_points,qualityLevel=0.008,
            minDistance=5,blockSize=7,useHarrisDetector=False)
    return np.empty((0,2),np.float32) if pts is None else pts.reshape(-1,2)


def essential_pose(src,dst,K,dist,args):
    r=dict(status='few_matches',essential_inliers=0,essential_ratio=0.,
        cheirality_inliers=0,cheirality_ratio_all=0.,
        cheirality_ratio_essential=0.,median_parallax_deg=float('nan'),
        R=None,t=None)
    n=len(src)
    if n < max(8,args.min_essential_inliers):return r
    u=cv2.undistortPoints(src.reshape(-1,1,2),K,dist,P=K).reshape(-1,2)
    v=cv2.undistortPoints(dst.reshape(-1,1,2),K,dist,P=K).reshape(-1,2)
    try:
        E,mask=cv2.findEssentialMat(u,v,K,method=cv2.RANSAC,
                       prob=0.999,threshold=args.ransac_px,maxIters=2000)
    except cv2.error:
        r['status']='essential_error';return r
    if E is None or mask is None:
        r['status']='essential_failed';return r
    ei=int(np.count_nonzero(mask))
    r['essential_inliers']=ei;r['essential_ratio']=ei/n
    if ei<args.min_essential_inliers:
        r['status']='insufficient_essential_inliers';return r
    best=None
    for i in range(0,E.shape[0],3):
        chunk=E[i:i+3]
        if chunk.shape!=(3,3):continue
        try:
            x,R,t,pm=cv2.recoverPose(chunk,u,v,K,mask=mask.copy())
        except cv2.error:continue
        if best is None or x>best[0]:best=(int(x),R,t,pm)
    if best is None:r['status']='recover_pose_failed';return r
    ni,R,t,pm=best
    r.update(cheirality_inliers=ni,cheirality_ratio_all=ni/n,
             cheirality_ratio_essential=ni/max(1,ei),R=R,t=t.reshape(3))
    valid=np.flatnonzero(pm.ravel()>0)
    if len(valid):
        # Rotation-compensated angular displacement (parallax indicator).
        a=cv2.undistortPoints(u[valid].reshape(-1,1,2),K,None).reshape(-1,2)
        b=cv2.undistortPoints(v[valid].reshape(-1,1,2),K,None).reshape(-1,2)
        ra=np.column_stack([a,np.ones(len(a))]);rb=np.column_stack([b,np.ones(len(b))])
        ra/=np.linalg.norm(ra,axis=1)[:,None]
        rb/=np.linalg.norm(rb,axis=1)[:,None]
        rb=(R.T@rb.T).T
        r['median_parallax_deg']=float(np.median(np.degrees(np.arccos(
            np.clip(np.sum(ra*rb,axis=1),-1,1)))))
    if ni<args.min_cheirality_inliers:
        r['status']='insufficient_cheirality_inliers'
    elif r['cheirality_ratio_essential']<args.min_cheirality_fraction:
        r['status']='low_cheirality_fraction'
    elif not np.isfinite(r['median_parallax_deg']) or r['median_parallax_deg']<args.min_parallax_deg:
        r['status']='low_parallax'
    else:r['status']='accepted'
    return r


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',required=True)
    p.add_argument('--images')
    p.add_argument('--out',help='Output folder; default: paper_research_draft/trajectories_v3')
    p.add_argument('--camera-json')
    p.add_argument('--max-frames',type=int,default=100,help='0 processes all frames')
    p.add_argument('--algorithms',nargs='+',choices=['ORB','AKAZE','SIFT'],default=['ORB','AKAZE'])
    p.add_argument('--preprocessings',nargs='+',choices=['raw','clahe'],default=['raw','clahe'])
    p.add_argument('--max-features',type=int,default=1500)
    p.add_argument('--flow-points',type=int,default=1800)
    p.add_argument('--flow-fb-px',type=float,default=1.5)
    p.add_argument('--ratio',type=float,default=0.75)
    p.add_argument('--mutual',action='store_true')
    p.add_argument('--ransac-px',type=float,default=1.5)
    p.add_argument('--min-essential-inliers',type=int,default=15)
    p.add_argument('--min-cheirality-inliers',type=int,default=10)
    p.add_argument('--min-cheirality-fraction',type=float,default=0.06)
    p.add_argument('--min-parallax-deg',type=float,default=0.10)
    p.add_argument('--max-anchor-gap',type=int,default=20)
    p.add_argument('--source-policy',choices=['quality','descriptor_first'],default='quality',
        help='quality evaluates both and ranks geometric support and parallax; descriptor_first prefers descriptor if valid')
    p.add_argument('--step-scale',choices=['unit','sqrt_gap','gap'],default='gap')
    args=p.parse_args()
    if args.max_frames<0 or args.max_anchor_gap<1 or not 0<args.ratio<1:
        p.error('invalid parameters')
    root=Path(args.root).resolve()
    image_dir=Path(args.images).resolve() if args.images else root/'raw_data'/'images_sequence_6'
    paths=sorted(image_dir.glob('frame*.png'))
    if args.max_frames:paths=paths[:args.max_frames]
    if len(paths)<2:raise RuntimeError(f'Not enough images in {image_dir}')
    out_dir=Path(args.out).resolve() if args.out else root/'paper_research_draft'/'trajectories_v3'
    out_dir.mkdir(parents=True,exist_ok=True)
    camera=CAMERA.copy()
    if args.camera_json:camera.update(json.loads(Path(args.camera_json).read_text()))
    K=np.array([[camera['fx'],0,camera['cx']],[0,camera['fy'],camera['cy']],[0,0,1]],float)
    dist=np.array(camera['dist_coeffs'],float)
    enhance=cv2.createCLAHE(clipLimit=2.0,tileGridSize=(8,8))
    summaries=[]
    for algorithm in args.algorithms:
        for prep in args.preprocessings:
            print(f'\n[START] {algorithm}/{prep}: {len(paths)} frames',flush=True)
            det,norm=detector_for(algorithm,args.max_features)
            anchor=None;anchor_desc=None;anchor_kp=[];anchor_pts=np.empty((0,2),np.float32)
            anchor_frame=0;Twa=np.eye(4);Tw=np.eye(4)
            accepted=0;rows=[];statuses=Counter();prev_fid=None
            for ii,path in enumerate(paths):
                start=time.perf_counter();fid=int(''.join(x for x in path.stem if x.isdigit()))
                img=cv2.imread(str(path),cv2.IMREAD_GRAYSCALE)
                if img is None:raise RuntimeError(f'Cannot read {path}')
                if img.shape!=(int(camera['height']),int(camera['width'])):
                    raise RuntimeError(f'Image dimensions {img.shape} do not match calibration')
                t=time.perf_counter();cur=enhance.apply(img) if prep=='clahe' else img
                pms=(time.perf_counter()-t)*1000
                t=time.perf_counter();kp,desc=detect(cur,det,args.max_features)
                dms=(time.perf_counter()-t)*1000
                initial=(ii==0);tracked=0;reset=0;step=0.;ms=0.;posems=0.
                nf=0;nd=0;source='none';nmatch=0
                src_results={};selection_quality=float('nan');disagreement=float('nan')
                attempted_gap=0 if initial else fid-anchor_frame
                diag=dict(status='initial_frame' if initial else 'few_matches',essential_inliers=0,
                    essential_ratio=0.,cheirality_inliers=0,cheirality_ratio_all=0.,
                    cheirality_ratio_essential=0.,median_parallax_deg=float('nan'))
                if initial:
                    anchor=cur.copy();anchor_kp=kp;anchor_desc=desc
                    anchor_pts=flow_points(cur,args.flow_points);anchor_frame=fid
                else:
                    t=time.perf_counter()
                    fa,fb=flow_pairs(anchor,cur,anchor_pts,args.flow_fb_px)
                    da,db=descriptor_pairs(anchor_kp,anchor_desc,kp,desc,norm,args.ratio,args.mutual)
                    nf=len(fa);nd=len(da)
                    ms=(time.perf_counter()-t)*1000
                    t=time.perf_counter()
                    # Evaluate BOTH sources independently on the SAME frame pair.
                    # Inliers, parallax and failure reasons are exported for audit.
                    # No COLMAP poses or reference information enter pose selection.
                    options=[]
                    for label,a,b in [('descriptor',da,db),('flow',fa,fb)]:
                        result=essential_pose(a,b,K,dist,args)
                        src_results[label]=result
                        if result['status']=='accepted':
                            # Avoid rewarding a tiny parallax change or one outlier.
                            # Anchor-gap penalty is common to both candidates on
                            # the same frame pair, so it does not introduce bias.
                            para=float(result['median_parallax_deg'])
                            support=int(result['cheirality_inliers'])
                            purity=float(result['cheirality_ratio_essential'])
                            quality=(min(support,200)
                                     *min(purity,1.0)
                                     *min(max(para,0.0),2.0)
                                     /math.sqrt(max(1,fid-anchor_frame)))
                            options.append((quality,label,len(a),result))
                    if len(options)==2:
                        Ra=options[0][3]['R'];Rb=options[1][3]['R']
                        disagreement=float(np.degrees(np.arccos(np.clip(
                            (np.trace(Ra@Rb.T)-1)/2,-1,1))))
                    if options:
                        if args.source_policy=='descriptor_first':
                            options.sort(key=lambda z:(z[1]!='descriptor',-z[0]))
                            selection_quality,source,nmatch,diag=options[0]
                        else:
                            selection_quality,source,nmatch,diag=max(
                                options,key=lambda z:(z[0],z[1]=='descriptor'))
                    else:
                        # Preserve the strongest rejected candidate for meaningful
                        # rejection diagnosis; do not mark it as accepted.
                        candidates=[(label,len(da if label=='descriptor' else fa),r)
                                    for label,r in src_results.items()]
                        source,nmatch,diag=max(candidates,
                            key=lambda z:(z[2]['cheirality_inliers'],
                                          z[2]['essential_inliers'],z[1]))
                    posems=(time.perf_counter()-t)*1000
                    if diag['status']=='accepted':
                        gap=max(1,fid-anchor_frame)
                        step=float(gap) if args.step_scale=='gap' else (
                             math.sqrt(gap) if args.step_scale=='sqrt_gap' else 1.)
                        tvec=diag['t']/max(np.linalg.norm(diag['t']),1e-12)*step
                        Tca=np.eye(4);Tca[:3,:3]=diag['R'];Tca[:3,3]=tvec
                        Tw=Twa@np.linalg.inv(Tca)
                        accepted+=1;tracked=1
                        anchor=cur.copy();anchor_kp=kp;anchor_desc=desc
                        anchor_pts=flow_points(cur,args.flow_points)
                        anchor_frame=fid;Twa=Tw.copy()
                    elif fid-anchor_frame>=args.max_anchor_gap:
                        # Force reanchor but mark unknown displacement, not accepted pose.
                        diag['status']='reanchored_after_'+diag['status'];reset=1
                        anchor=cur.copy();anchor_kp=kp;anchor_desc=desc
                        anchor_pts=flow_points(cur,args.flow_points)
                        anchor_frame=fid;Twa=Tw.copy()
                statuses[diag['status']]+=1
                q=quat(Tw[:3,:3]);xyz=Tw[:3,3]
                rows.append(dict(frame=fid,time_sec=fid/20.,anchor_frame=anchor_frame,
                    anchor_gap=fid-anchor_frame,motion_step_units=step,
                    x=xyz[0],y=xyz[1],z=xyz[2],qx=q[0],qy=q[1],qz=q[2],qw=q[3],
                    features=len(kp),anchor_features=len(anchor_kp),matches=nmatch,
                    essential_inliers=diag['essential_inliers'],essential_ratio=diag['essential_ratio'],
                    cheirality_inliers=diag['cheirality_inliers'],
                    cheirality_ratio_all=diag['cheirality_ratio_all'],
                    cheirality_ratio_essential=diag['cheirality_ratio_essential'],
                    median_parallax_deg=diag['median_parallax_deg'],
                    tracking_success=tracked,rotation_only=0,status=diag['status'],
                    scale_mode='monocular_arbitrary_'+args.step_scale,
                    preprocess_ms=pms,detection_ms=dms,matching_ms=ms,pose_ms=posems,
                    total_ms=(time.perf_counter()-start)*1000,
                    match_source=source,flow_matches=nf,descriptor_matches=nd,
                    pose_valid=tracked,pose_coverage=int(tracked),anchor_reset=reset,
                    consecutive_pose_inliers=0,
                    descriptor_status=src_results.get('descriptor',{}).get('status','not_run'),
                    flow_status=src_results.get('flow',{}).get('status','not_run'),
                    descriptor_essential_inliers=src_results.get('descriptor',{}).get('essential_inliers',0),
                    flow_essential_inliers=src_results.get('flow',{}).get('essential_inliers',0),
                    descriptor_cheirality_inliers=src_results.get('descriptor',{}).get('cheirality_inliers',0),
                    flow_cheirality_inliers=src_results.get('flow',{}).get('cheirality_inliers',0),
                    descriptor_parallax_deg=src_results.get('descriptor',{}).get('median_parallax_deg',float('nan')),
                    flow_parallax_deg=src_results.get('flow',{}).get('median_parallax_deg',float('nan')),
                    descriptor_valid=int(src_results.get('descriptor',{}).get('status')=='accepted'),
                    flow_valid=int(src_results.get('flow',{}).get('status')=='accepted'),
                    both_valid=int(all(src_results.get(k,{}).get('status')=='accepted' for k in ('descriptor','flow'))),
                    selection_quality=selection_quality,estimated_anchor_gap=attempted_gap,
                    candidate_disagreement_deg=disagreement))
                prev_fid=fid
                if (ii+1)%250==0:print(f'  {ii+1}/{len(paths)} accepted {accepted}',flush=True)
            stem=f'vo_sequence6_{algorithm.lower()}_{prep}_v3'
            with (out_dir/f'{stem}.csv').open('w',newline='',encoding='utf-8') as f:
                w=csv.DictWriter(f,fieldnames=FIELDS);w.writeheader();w.writerows(rows)
            with (out_dir/f'{stem}_keyframes.csv').open('w',newline='',encoding='utf-8') as f:
                w=csv.DictWriter(f,fieldnames=FIELDS);w.writeheader()
                w.writerows(r for r in rows if int(r['frame'])%20==0)
            summaries.append(dict(algorithm=algorithm,preprocessing=prep,frames=len(rows),
                descriptor_valid=sum(r['descriptor_valid'] for r in rows),
                flow_valid=sum(r['flow_valid'] for r in rows),
                both_valid=sum(r['both_valid'] for r in rows),
                either_valid=sum(bool(r['descriptor_valid'] or r['flow_valid']) for r in rows),
                accepted=accepted,success_rate=accepted/(len(rows)-1),statuses=dict(statuses),
                parameters=vars(args),calibration=camera))
            print(f'[SAVED] {stem}: accepted {accepted}/{len(rows)-1} ({100*accepted/(len(rows)-1):.1f}%)')
            print(f'[REJECTIONS] {dict(statuses)}',flush=True)
            print('[PAIRED SOURCES] descriptor_valid={}; flow_valid={}; both_valid={}; either_valid={}'.format(
                summaries[-1]['descriptor_valid'], summaries[-1]['flow_valid'],
                summaries[-1]['both_valid'], summaries[-1]['either_valid']),flush=True)
    (out_dir/'run_summary.json').write_text(json.dumps(summaries,indent=2,default=str),encoding='utf-8')
    print(f'[COMPLETE] {out_dir}')

if __name__=='__main__':main()
