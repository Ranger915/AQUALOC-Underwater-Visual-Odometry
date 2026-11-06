"""Keep causal single-epoch windows separate from final-map, all-fit ATE."""
import os,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];SEQ=int(sys.argv[1]);assert SEQ in [5,6]
os.environ['AQUALOC_REFERENCE']=str(ROOT/f'sequence5_results/reference/new_archaeo_colmap_traj_sequence_{SEQ:02d}.txt')
from dense_evaluate import *
import pandas as pd

def epochs(map_ids,state):
 return np.cumsum(np.r_[False,(map_ids[1:]!=map_ids[:-1])|((state[1:]<=1)&(state[:-1]>1))])

def main():
 dest=ROOT/'baseline_results';target=dest/f'evaluation_sequence{SEQ}';target.mkdir(exist_ok=True)
 manifests=np.array([int(s.split()[0]) for s in (dest/f'sequence{SEQ}.manifest').read_text().splitlines()],np.int64);expected=len(manifests)
 segments=[];errors=[];coverage=[];native_metrics=[];native_errors=[];checks=[]
 def check(label,ok):assert bool(ok),label;checks.append(dict(check=label,passed=True))
 for prep in ['raw','fixed','adaptive']:
  own=model((ROOT/'sequence5_results/vo' if SEQ==5 else OUT/'vo')/f'ORB_{prep}_rho0.06_quality_depth1e+09.csv')
  for trial in range(3):
   name=f'sequence{SEQ}_{prep}_trial{trial}';folder=dest/name;meta=json.loads((folder/'done.json').read_text());d=pd.read_csv(folder/'online.csv');maps=pd.read_csv(folder/'maps.csv');state=d.state.to_numpy(int);ok=state==2;n=len(FRAMES)
   check('complete '+name,meta['completed'] and meta['frames']==len(d)==expected and np.array_equal(d.frame,np.arange(expected)) and np.array_equal(d.timestamp_ns,manifests) and meta['config_sha256']==sha(dest/'aqualoc.yaml'))
   xyz=d[['px','py','pz']].to_numpy();q=d[['qx','qy','qz','qw']].to_numpy();check('valid tracked poses '+name,np.isfinite(xyz[ok]).all() and np.isfinite(q[ok]).all() and np.allclose(np.linalg.norm(q[ok],axis=1),1,atol=1e-5))
   rotations=np.tile(np.eye(3),(len(d),1,1));rotations[ok]=np.array([quatmat(v) for v in q[ok]]).reshape(-1,3,3);epoch=epochs(d.map_id.to_numpy(),state);runseg=[];runerr=[]
   for ep in np.unique(epoch[:n]):
    fresh=ok[:n]&(epoch[:n]==ep)
    if fresh.sum()<30:continue
    baseline=dict(name='ORB_SLAM3',p=xyz[:n],R=rotations[:n],fresh=fresh)
    for clean in [False,True]:
     ss,ee,_=eval_group([baseline,own],name,clean=clean,fit_lengths=(800,),safe_reference=True)
     eligible={r['test_boundary'] for r in ss if np.all(epoch[r['test_boundary']-800:min(r['test_boundary']+200,n)]==ep)}
     for r in ss:
      if r['test_boundary'] in eligible:r['segment']=r['test_boundary']//200;runseg.append(r)
     for r in ee:
      if r['fit_boundary'] in eligible:r['segment']=r['fit_boundary']//200;runerr.append(r)
   segments+=runseg;errors+=runerr
   positive=maps[maps.keyframes>0];chosen=int(positive.sort_values(['keyframes','map_id'],ascending=[False,True]).iloc[0].map_id) if len(positive) else -1
   cov=dict(sequence=SEQ,preprocessing=prep,trial=trial,frames=expected,tracking_OK=int(ok.sum()),tracking_percent=100*ok.mean(),epochs=int(epoch.max()+1),nonempty_final_maps=len(positive),largest_map_id=chosen,largest_map_keyframes=int(positive.keyframes.max()) if len(positive) else 0,causal_windows=len({r['test_boundary'] for r in runseg if r['mask']=='all_reference'}),final_native_poses=0,paired_native_poses=0)
   if chosen>=0:
    file=folder/f'map_{chosen}.txt';a=np.loadtxt(file,ndmin=2) if file.stat().st_size else np.empty((0,8))
    if len(a):
     check('eight-column final poses '+name,a.shape[1]==8)
     idx=np.searchsorted(manifests,np.rint(a[:,0]).astype(np.int64));idx=np.clip(idx,0,expected-1);check('final trajectory timestamp association '+name,np.max(abs(manifests[idx]-a[:,0]))<1)
     check('unique optimized tracked poses '+name,len(np.unique(idx))==len(idx) and np.isfinite(a).all() and np.allclose(np.linalg.norm(a[:,4:8],axis=1),1,atol=1e-5))
     valid=np.isin(idx,IDS)&ok[idx];a=a[valid];idx=idx[valid];cov['final_native_poses']=len(idx)
     for paired in [False,True]:
      keep=own['fresh'][idx] if paired else np.ones(len(idx),bool);ids=idx[keep];aa=a[keep];label='paired_native_all_fit' if paired else 'baseline_native_all_fit'
      if paired:cov['paired_native_poses']=len(ids)
      if len(ids)<6:continue
      pp=aa[:,1:4];rr=np.array([quatmat(v) for v in aa[:,4:8]])
      for method,p,R in [('ORB_SLAM3',pp,rr)]+([('pairwise_ORB',own['p'][ids],own['R'][ids])] if paired else []):
       if np.linalg.svd(p-p.mean(0),compute_uv=False)[1]<1e-10:continue
       scale,align,t=fit(p,C[ids]);check('positive final alignment '+name+'/'+label+'/'+method,np.isfinite(scale) and scale>0)
       residual=scale*(p@align.T)+t-C[ids];err=np.linalg.norm(residual,axis=1);lookup={int(v):i for i,v in enumerate(ids)};rpe=[];rot=[];direction=[]
       for i,frame in enumerate(ids):
        j=lookup.get(int(frame)+20)
        if j is None:continue
        dr=ROTS[frame].T@(C[ids[j]]-C[frame]);de=R[i].T@(p[j]-p[i]);rpe.append(float(np.linalg.norm(scale*de-dr)));rot.append(degrees((ROTS[frame].T@ROTS[ids[j]]).T@(R[i].T@R[j])));direction.append(angle(dr,de))
       native_metrics.append(dict(sequence=SEQ,preprocessing=prep,trial=trial,scope=label,method=method,map_id=chosen,poses=len(ids),rmse_m=float(np.sqrt(np.mean(err**2))),scale=float(scale),path_fraction=float(np.sqrt(np.mean(err**2))/PATH),rpe_1s_rmse_m=float(np.sqrt(np.mean(np.square(rpe)))) if rpe else np.nan,rotation_1s_median_deg=float(np.median(rot)) if rot else np.nan,direction_1s_median_deg=float(np.nanmedian(direction)) if np.isfinite(direction).any() else np.nan,relative_pairs=len(rpe),fit_ids=';'.join(map(str,ids))))
       native_errors.extend(dict(sequence=SEQ,preprocessing=prep,trial=trial,scope=label,method=method,frame=int(frame),error_m=float(e)) for frame,e in zip(ids,err))
   coverage.append(cov)
 writecsv(target/'causal_segments.csv',segments);writecsv(target/'causal_per_pose.csv',errors);writecsv(target/'coverage.csv',coverage);writecsv(target/'native_all_fit_metrics.csv',native_metrics);writecsv(target/'native_all_fit_per_pose.csv',native_errors)
 summary=[]
 if segments:
  seg=pd.DataFrame(segments);ee=pd.DataFrame(errors);keys=['group','mask','configuration','segment'];d=seg.set_index(keys).join(ee.groupby(keys).error.agg(n='size',ss=lambda x:float(x@x)))
  check('causal residual counts and SSE',np.array_equal(d.test_n,d.n) and np.allclose(d.sse,d.ss,atol=1e-9,rtol=1e-9))
  for (group,mask,bound),d in seg.groupby(['group','mask','test_boundary']):check('same safe single-epoch window '+group+'/'+mask+'/'+str(bound),len(d)==2 and d.train_ids.nunique()==d.test_ids.nunique()==1 and max(map(int,d.train_ids.iloc[0].split(';')))<=IDS[IDS<bound][-1])
  for (group,mask,configuration),d in seg.groupby(['group','mask','configuration']):summary.append(dict(run=group,mask=mask,method=configuration,windows=len(d),tests=int(d.test_n.sum()),rmse_m=float(np.sqrt(d.sse.sum()/d.test_n.sum()))))
  writecsv(target/'causal_paired_ranges.csv',boot(segments,replicates=2000))
 writecsv(target/'causal_summary.csv',summary);writecsv(target/'validation.csv',checks);dump(target/'validation_summary.json',dict(passed=True,checks=len(checks),runs=9,sequence=SEQ,causal_windows='safe reference brackets; no window crosses a reset/map epoch',final_native='all-fit Sim(3), largest final map by keyframes; not held-out and may use future map optimization',reference=str(REFERENCE)))
 print('PASS',len(checks),'checks; runs',len(coverage),'causal segments',len(segments),'native metric rows',len(native_metrics),flush=True)

if __name__=='__main__':
 assert np.array_equal(epochs(np.array([0,0,0,0,0,2]),np.array([1,2,3,1,2,1])),[0,0,0,1,1,2])
 main()
