"""Illustrative detected keypoints, spatial density, and tentative/inlier matches.
Freshly recomputed visualizations (not original cached benchmark correspondences).
"""
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import cv2
import matplotlib.pyplot as plt
from common import output, export, style, DETECTORS

def detector(name):
    if name=='ORB': return cv2.ORB_create(nfeatures=1500)
    if name=='SIFT': return cv2.SIFT_create(nfeatures=1500)
    if name=='AKAZE_MLDB':
        if hasattr(cv2,'AKAZE_create'): return cv2.AKAZE_create(descriptor_type=cv2.AKAZE_DESCRIPTOR_MLDB)
        if hasattr(cv2,'xfeatures2d') and hasattr(cv2.xfeatures2d,'AKAZE_create'):
            return cv2.xfeatures2d.AKAZE_create(descriptor_type=cv2.xfeatures2d.AKAZE_DESCRIPTOR_MLDB)
        raise RuntimeError('AKAZE_MLDB missing from this OpenCV build')
    raise ValueError(name)

def image(root,frame):
    p=Path(root)/'raw_data'/'images_sequence_6'/f'frame{frame:06d}.png'
    im=cv2.imread(str(p),cv2.IMREAD_GRAYSCALE)
    if im is None:raise FileNotFoundError(f'Cannot read {p}')
    return im

def decisions(root):
    p=Path(root)/'sequence6_adaptive_v2_full'/'adaptive_frame_decisions.csv'
    if not p.exists():raise FileNotFoundError(f'Adaptive decision log missing: {p}; do not guess decisions')
    d=pd.read_csv(p); required=['frame_id','decision','clip_limit']
    for c in required:
        if c not in d:raise ValueError(f'{p} lacks {c}')
    return d.set_index('frame_id')

def prep(im,kind,frame,d):
    if kind=='raw': return im
    if kind=='clahe': return cv2.createCLAHE(clipLimit=2.0,tileGridSize=(8,8)).apply(im)
    row=d.loc[frame]
    if isinstance(row,pd.DataFrame):row=row.iloc[0]
    dec=str(row['decision']).lower()
    if dec=='raw':return im
    if dec not in ('moderate_clahe','strong_clahe'):
        raise ValueError(f'Unknown adaptive preprocessing decision {dec!r} on frame {frame}')
    clip=float(row['clip_limit'])
    return cv2.createCLAHE(clipLimit=clip,tileGridSize=(8,8)).apply(im)

def match(k1,d1,k2,d2,name):
    if d1 is None or d2 is None or len(d1)<2 or len(d2)<2:return [],[],None
    norm=cv2.NORM_L2 if name=='SIFT' else cv2.NORM_HAMMING
    knn=cv2.BFMatcher(norm).knnMatch(d1,d2,k=2)
    ms=[m for pair in knn if len(pair)==2 for m,n in [pair] if m.distance<.75*n.distance]
    if len(ms)<8:return ms,[],None
    a=np.float32([k1[m.queryIdx].pt for m in ms]);b=np.float32([k2[m.trainIdx].pt for m in ms])
    F,mask=cv2.findFundamentalMat(a,b,cv2.FM_RANSAC,1.5,.99)
    if F is None or mask is None or F.shape!=(3,3):return ms,[],None
    inlier=[m for m,good in zip(ms,mask.ravel()) if good]
    return ms,inlier,F

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',required=True);p.add_argument('--frame-a',type=int,default=100)
    p.add_argument('--frame-b',type=int,default=120);p.add_argument('--max-drawn-matches',type=int,default=90)
    args=p.parse_args();style();out=output(args.root,'02_feature_maps_matches')
    d=decisions(args.root); frames={f:image(args.root,f) for f in [args.frame_a,args.frame_b]}
    info=[]
    for detname in DETECTORS:
        det=detector(detname)
        for variant in ['raw','clahe','adaptive_v2']:
            a=prep(frames[args.frame_a],variant,args.frame_a,d);b=prep(frames[args.frame_b],variant,args.frame_b,d)
            k1,d1=det.detectAndCompute(a,None);k2,d2=det.detectAndCompute(b,None)
            draw=cv2.drawKeypoints(a,k1,None,flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
            cv2.imwrite(str(out/f'keypoints_{detname}_{variant}_frame{args.frame_a}.png'),draw)
            # Grid count per frame: visualize keypoint spatial distribution
            gy,gx=8,8;H,W=a.shape
            occupancy=np.zeros((gy,gx),dtype=int)
            for kp in k1:
                xx=min(gx-1,int(kp.pt[0]*gx/W)); yy=min(gy-1,int(kp.pt[1]*gy/H))
                occupancy[yy,xx]+=1
            fig,ax=plt.subplots(figsize=(8,4.8));image_obj=ax.imshow(occupancy,cmap='viridis',origin='upper',aspect='auto')
            fig.colorbar(image_obj,ax=ax,label='Detected keypoints / grid cell')
            ax.set(xlabel='Image column bin',ylabel='Image row bin',title=f'{detname} {variant} | frame {args.frame_a} keypoint density')
            export(fig,out,f'density_{detname}_{variant}_frame{args.frame_a}')
            tentative,inliers,F=match(k1,d1,k2,d2,detname)
            vis=cv2.drawMatches(a,k1,b,k2,inliers[:args.max_drawn_matches],None,flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
            cv2.imwrite(str(out/f'inlier_matches_{detname}_{variant}_{args.frame_a}_{args.frame_b}.png'),vis)
            info.append({'detector':detname,'preprocessing':variant,'frame_a':args.frame_a,'frame_b':args.frame_b,
                         'keypoints_a':len(k1),'keypoints_b':len(k2),'tentative_ratio_matches':len(tentative),
                         'estimated_fundamental_inliers':len(inliers),
                         'epipolar_estimation_success':bool(F is not None)})
    pd.DataFrame(info).to_csv(out/'visualization_pair_metrics.csv',index=False)
    print('[NOTE] These are illustrative reruns; estimated F inliers are not COLMAP-reference verified matches.')

if __name__=='__main__':main()
