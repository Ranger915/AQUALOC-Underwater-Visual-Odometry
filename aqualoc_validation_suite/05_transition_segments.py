#!/usr/bin/env python3
"""Describe temporal RAW/CLAHE regimes and exact mixed pairs (no p-values)."""
import argparse,csv,json
from collections import Counter,defaultdict
from pathlib import Path

def load(path):
    with open(path,newline='',encoding='utf-8-sig') as f:return list(csv.DictReader(f))
def norm(x):return x.upper().replace('AKAZE_MLDB','AKAZE')
def main():
    p=argparse.ArgumentParser();p.add_argument('--decisions',required=True);p.add_argument('--fixed',required=True);p.add_argument('--adaptive',required=True);p.add_argument('--output',required=True)
    a=p.parse_args();out=Path(a.output);out.mkdir(parents=True,exist_ok=True)
    dec=load(a.decisions); d={int(r['frame_id']):r['decision'] for r in dec}
    ids=sorted(d);seg=[]
    if not ids:raise ValueError('Empty decisions CSV')
    first=last=ids[0];mode=d[first]
    for f in ids[1:]:
        if f!=last+1 or d[f]!=mode:
            seg.append({'mode':mode,'first_frame':first,'last_frame':last,'frames':last-first+1});first=f;mode=d[f]
        last=f
    seg.append({'mode':mode,'first_frame':first,'last_frame':last,'frames':last-first+1})
    fixed={(norm(r['detector']),int(r['frame_id_1']),int(r['frame_id_2']),int(r['frame_stride'])):r for r in load(a.fixed)}
    adapt={(norm(r['detector']),int(r['frame_id_1']),int(r['frame_id_2']),int(r['frame_stride'])):r for r in load(a.adaptive)}
    if set(fixed)!=set(adapt):raise ValueError('Pair keys differ')
    results=[]
    for key in sorted(fixed):
        detector,i,j,stride=key
        if d[i]==d[j]:continue
        c=fixed[key];v=adapt[key]
        cs=c['status']=='ok' and int(float(c['ransac_inliers']))>=8
        vs=v['status']=='ok' and int(float(v['ransac_inliers']))>=8
        results.append({'detector':detector,'frame_a':i,'frame_b':j,'stride':stride,
                        'mode_a':d[i],'mode_b':d[j],
                        'fixed_ok':int(cs),'adaptive_ok':int(vs),
                        'delta_success':int(vs)-int(cs),
                        'delta_inliers':float(v['ransac_inliers'])-float(c['ransac_inliers'])})
    for filename,rows in [('preprocessing_segments.csv',seg),('mixed_pair_details.csv',results)]:
        dest=out/filename
        with dest.open('w',newline='',encoding='utf-8') as f:
            w=csv.DictWriter(f,fieldnames=rows[0].keys());w.writeheader();w.writerows(rows)
        print('[SAVED]',dest)
    print('Contiguous segments:',len(seg),'RAW segments:',sum(r['mode']=='raw' for r in seg))
    print('Mixed pairs by detector/stride:')
    for (det,stride),n in sorted(Counter((r['detector'],r['stride']) for r in results).items()):
        bad=sum(r['delta_success']<0 for r in results if r['detector']==det and r['stride']==stride)
        print(f'{det:6} stride={stride:2} mixed_pairs={n:3} adaptive_lost_success_pairs={bad}')

if __name__=='__main__':main()
