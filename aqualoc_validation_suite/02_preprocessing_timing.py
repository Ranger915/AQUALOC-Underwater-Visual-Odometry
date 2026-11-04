#!/usr/bin/env python3
"""Within-run preprocessing timing: raw, fixed CLAHE, adaptive v2.
Images read once each; compare decision+apply separately; no claim of full SLAM FPS.
"""
import argparse,csv,json,statistics,time,sys
from pathlib import Path
import cv2
import numpy as np


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--image-dir',required=True);p.add_argument('--adaptive-package',required=True)
    p.add_argument('--output',required=True);p.add_argument('--repeats',type=int,default=3)
    p.add_argument('--step',type=int,default=1);p.add_argument('--max-frames',type=int,default=0)
    args=p.parse_args()
    sys.path.insert(0,str(Path(args.adaptive_package).resolve()))
    from adaptive_clahe import AdaptiveConfig,decide,apply_adaptive
    config=AdaptiveConfig(version='v2',local_threshold_v2=10.5)
    images=sorted(Path(args.image_dir).glob('frame*.png'))[::args.step]
    if args.max_frames:images=images[:args.max_frames]
    if not images:raise RuntimeError('No image frames found')
    out=Path(args.output);out.mkdir(parents=True,exist_ok=True)
    rows=[]
    # Warm up OpenCV internal allocations and CLAHE.
    warm=cv2.imread(str(images[0]),cv2.IMREAD_GRAYSCALE)
    if warm is None:raise RuntimeError(f'Unreadable {images[0]}')
    for _ in range(3):
        cv2.createCLAHE(clipLimit=2.0,tileGridSize=(8,8)).apply(warm)
        d=decide(warm,config);apply_adaptive(warm,config,d)
    for idx,path in enumerate(images):
        t=time.perf_counter();gray=cv2.imread(str(path),cv2.IMREAD_GRAYSCALE);read_s=time.perf_counter()-t
        if gray is None:raise RuntimeError(f'Unreadable {path}')
        for rep in range(args.repeats):
            # Repeat methods on the *same* input frame, rotating execution order.
            order=['raw','fixed_clahe','adaptive_v2'];order=order[(idx+rep)%3:]+order[:(idx+rep)%3]
            for method in order:
                t0=time.perf_counter();decision=None
                if method=='raw':
                    processed=gray;decision_s=0.;apply_s=0.;mode='raw'
                elif method=='fixed_clahe':
                    decision_s=0.;start=time.perf_counter()
                    processed=cv2.createCLAHE(clipLimit=2.0,tileGridSize=(8,8)).apply(gray)
                    apply_s=time.perf_counter()-start;mode='fixed_clahe'
                else:
                    start=time.perf_counter();decision=decide(gray,config);decision_s=time.perf_counter()-start
                    start=time.perf_counter();processed,_=apply_adaptive(gray,config,decision)
                    apply_s=time.perf_counter()-start;mode=decision['decision']
                elapsed=time.perf_counter()-t0
                rows.append({'frame':path.name,'repeat':rep,'method':method,'mode':mode,
                             'read_s':read_s,'decision_s':decision_s,'apply_s':apply_s,
                             'total_preprocessing_s':elapsed,'image_checksum':int(np.sum(processed,dtype=np.uint64)%1000000007)})
        if (idx+1)%250==0: print(f'[TIMING] {idx+1}/{len(images)}',flush=True)
    dest=out/'preprocessing_timing.csv'
    with dest.open('w',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=rows[0].keys());w.writeheader();w.writerows(rows)
    summary=[]
    for m in ('raw','fixed_clahe','adaptive_v2'):
        selected=[r for r in rows if r['method']==m]
        x=sorted(r['total_preprocessing_s']*1000 for r in selected)
        def pct(p):return x[round((len(x)-1)*p)]
        summary.append({'method':m,'observations':len(x),'mean_ms':statistics.mean(x),'median_ms':statistics.median(x),'p95_ms':pct(.95),'mean_decision_ms':statistics.mean(r['decision_s']*1000 for r in selected),'mean_apply_ms':statistics.mean(r['apply_s']*1000 for r in selected)})
    path=out/'preprocessing_timing_summary.csv'
    with path.open('w',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=summary[0].keys());w.writeheader();w.writerows(summary)
    (out/'timing_metadata.json').write_text(json.dumps({'opencv':cv2.__version__,'numpy':np.__version__,'image_dir':str(Path(args.image_dir).resolve()),'repeats':args.repeats,'step':args.step,'frames':len(images),'interpretation':'Within-run wall time; reads excluded from total preprocessing; excludes detector/matcher geometry time. For full end-to-end compare identical pipelines with timing instrumentation.'},indent=2),encoding='utf-8')
    for r in summary:print(r)
    print('[SAVED]',dest,path)

if __name__=='__main__':main()
