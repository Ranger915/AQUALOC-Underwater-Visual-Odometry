from common import *
from fast_match import fast_correspond
import argparse
PAIRS=readcsv(DATA/'raw_data/sequence6_calibrated_reference/sequence6_calibrated_pair_manifest.csv')
IDS=list(range(0,3381,20));REF=refposes();LABEL=np.array([int(p['colmap_overlap_label']) for p in PAIRS]);FUND=[fundamental(REF[int(p['frame_id_i'])],REF[int(p['frame_id_j'])]) for p in PAIRS]
def getfeatures(alg,mode):
 folder=CACHE/'reference_features'/alg/mode;folder.mkdir(parents=True,exist_ok=True);out={};times=[];settings={}
 low=mode in ['raw_matched','raw_budget'];det,norm=detector(alg);fixed_budget=None
 if low:
  tune=OUT/f'{alg}_threshold_match.json'
  if tune.exists():cfg=json.loads(tune.read_text())
  else:
   fixed,_=getfeatures(alg,'c2t8');target=float(np.mean([len(x[0]) for x in fixed.values()]));grid={'ORB':[1,5,10,20],'AKAZE':[.000001,.00001,.00005,.0001,.0003,.001],'SIFT':[.0001,.001,.005,.01,.02,.04]}[alg];trials=[]
   # Feature counts alone select the global threshold; no pair outcomes enter.
   for threshold in grid:
    d,_=detector(alg)
    if alg=='ORB':d.setFastThreshold(int(threshold))
    elif alg=='AKAZE':d.setThreshold(float(threshold))
    else:d.setContrastThreshold(float(threshold))
    counts=[len(feature(image(fid),d)[0]) for fid in IDS]
    trials.append(dict(threshold=threshold,mean_count=float(np.mean(counts)),target_mean=target))
   cfg=dict(min(trials,key=lambda t:(abs(t['mean_count']-target),-t['threshold'])));cfg['grid']=trials;dump(tune,cfg)
  th=cfg['threshold'] if mode=='raw_matched' else min(t['threshold'] for t in cfg['grid']);settings=cfg
  if mode=='raw_budget':fixed_budget=getfeatures(alg,'c2t8')[0]
  if alg=='ORB':det.setFastThreshold(int(th))
  elif alg=='AKAZE':det.setThreshold(float(th))
  else:det.setContrastThreshold(float(th))
 for fid in IDS:
  p=folder/f'{fid}.npz'
  if p.exists():
   with np.load(p) as z:out[fid]=(z['p'],z['d']);times.append(float(z['seconds']))
  else:
   im=image(fid)
   if mode.startswith('c'):
    clip,tile=mode[1:].split('t');im=cv2.createCLAHE(float(clip),(int(tile),int(tile))).apply(im)
   start=time.perf_counter();a,b=feature(im,det)
   if fixed_budget is not None:
    keep=len(fixed_budget[fid][0]);a=a[:keep];b=b[:keep]
   elapsed=time.perf_counter()-start;np.savez_compressed(p,p=a,d=b,seconds=elapsed);out[fid]=(a,b);times.append(elapsed)
 writecsv(OUT/'feature_counts'/f'{alg}_{mode}.csv',[dict(detector=alg,mode=mode,frame=fid,features=len(out[fid][0]),extraction_seconds=tm) for fid,tm in zip(IDS,times)])
 return out,norm

def evaluate(alg,left,right):
 name=f'{alg}_{left}_{right}';path=OUT/'frontend'/f'{name}.csv'
 if path.exists() and len(readcsv(path))==len(PAIRS):return
 a,norm=getfeatures(alg,left);b=a if right==left else getfeatures(alg,right)[0];partial=path.with_suffix('.partial.csv');rr=readcsv(partial) if partial.exists() else [];start=time.perf_counter()
 for idx in range(len(rr),len(PAIRS)):
  p,F=PAIRS[idx],FUND[idx]
  i,j=int(p['frame_id_i']),int(p['frame_id_j']);x,y=fast_correspond(a[i],b[j],norm,True)
  d=sampson(undistort(x),undistort(y),F) if len(x) else np.array([])
  rr.append(dict(i=i,j=j,label=int(p['colmap_overlap_label']),matches=len(d),inliers=int(np.sum(d<=2)),median_sampson=float(np.median(d)) if len(d) else float('nan')))
  if idx%2000==0:
   dump(OUT/f'frontend_{alg}_progress.json',dict(task=name,pair=idx,seconds=time.perf_counter()-start));writecsv(partial,rr)
 writecsv(path,rr);print('DONE',name,f'{time.perf_counter()-start:.1f}s',flush=True)

def run(alg):
 (OUT/'frontend').mkdir(exist_ok=True);(OUT/'feature_counts').mkdir(exist_ok=True)
 # Score all four endpoint processing combinations, permitting exact switching
 # controls and threshold sweeps without approximating mixed matches.
 for left,right in [('c2t8','c2t8'),('raw','raw'),('raw','c2t8'),('c2t8','raw'),('raw_matched','raw_matched')]:evaluate(alg,left,right)
 for clip in [1,2,3,4]:
  for tile in [4,8,16]:evaluate(alg,f'c{clip}t{tile}',f'c{clip}t{tile}')
 print('FRONTEND COMPLETE',alg,flush=True)
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('detector');p.add_argument('--budget-only',action='store_true');a=p.parse_args()
 if a.budget_only:evaluate(a.detector,'raw_budget','raw_budget')
 else:run(a.detector)
