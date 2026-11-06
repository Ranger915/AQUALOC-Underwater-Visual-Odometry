from common import *
from functools import lru_cache
import argparse,importlib.util,math,pickle
spec=importlib.util.spec_from_file_location('legacy',DATA/'aqualoc_vo_v3_pilot/slam_engine_sequence6_v3.py')
legacy=importlib.util.module_from_spec(spec);spec.loader.exec_module(legacy)

@lru_cache(maxsize=28)
def prepared(fid,prep):
 im=image(fid)
 if prep=='fixed':im=cv2.createCLAHE(2.,(8,8)).apply(im)
 elif prep=='adaptive':
  if DECISIONS[fid]!='raw':im=cv2.createCLAHE(2.,(8,8)).apply(im)
 return im
DECISIONS={int(r['frame_id']):r['decision'] for r in readcsv(DATA/'sequence6_adaptive_v2_full/adaptive_frame_decisions.csv')}

def pose(a,b,cutoff,seed,permute=False):
 n=len(a);r=dict(ne=0,nc=0,rho=0.,parallax=float('nan'),R=None,t=None,status='few_matches')
 if n<15:return r
 u=undistort(a);v=undistort(b);cv2.setRNGSeed(seed)
 if permute:
  order=np.random.default_rng(seed).permutation(n);u=u[order];v=v[order]
 try:E,mask=cv2.findEssentialMat(u,v,K,method=cv2.RANSAC,prob=.999,threshold=1.5,maxIters=2000)
 except cv2.error:r['status']='essential_error';return r
 if E is None or mask is None:r['status']='essential_failed';return r
 ne=int(mask.sum());r['ne']=ne
 if ne<15:r['status']='insufficient_essential_inliers';return r
 best=None
 for j in range(0,len(E),3):
  e=E[j:j+3]
  if e.shape!=(3,3):continue
  try:out=cv2.recoverPose(e,u,v,K,distanceThresh=float(cutoff),mask=mask.copy())
  except cv2.error:continue
  if len(out)!=5:raise RuntimeError('Unexpected recoverPose overload; distanceThresh was not applied')
  nc,R,t,pm=out[:4]
  if best is None or nc>best[0]:best=(nc,R,t.reshape(3),pm)
 if best is None:r['status']='recover_pose_failed';return r
 nc,R,t,pm=best;valid=pm.ravel()>0
 r.update(nc=int(nc),rho=float(nc/max(1,ne)),R=R,t=t,status='geometry')
 if np.any(valid):
  ra=cv2.undistortPoints(u[valid].reshape(-1,1,2),K,None).reshape(-1,2);rb=cv2.undistortPoints(v[valid].reshape(-1,1,2),K,None).reshape(-1,2)
  ra=np.column_stack([ra,np.ones(len(ra))]);rb=np.column_stack([rb,np.ones(len(rb))]);ra/=np.linalg.norm(ra,axis=1)[:,None];rb/=np.linalg.norm(rb,axis=1)[:,None];rb=rb@R
  r['parallax']=float(np.median(np.degrees(np.arccos(np.clip(np.sum(ra*rb,axis=1),-1,1)))))
 return r

def gate(r,rho):
 if r['status']!='geometry':return r['status']
 if r['nc']<10:return 'insufficient_cheirality_inliers'
 if r['rho']<rho:return 'low_cheirality_fraction'
 if not np.isfinite(r['parallax']) or r['parallax']<.1:return 'low_parallax'
 return 'accepted'

def run(algorithm,prep,seed=20261105,result_dir=None,replicate=False,frames=3385,comparisons=None):
 global OUT
 assert isinstance(frames,int) and frames>0
 if result_dir is not None:OUT=Path(result_dir);OUT.mkdir(exist_ok=True,parents=True)
 tstart=time.perf_counter();det,norm=detector(algorithm)
 states=[]
 combinations=[(.06,'quality',50.),(.06,'quality',1e9)]
 if algorithm=='ORB' and prep=='fixed':combinations=[(rho,pol,cut) for cut in [50.,1e9] for rho in [.03,.06,.10,.15] for pol in ['quality','descriptor_first']]
 if replicate:combinations=[(.06,pol,1e9) for pol in (['quality','descriptor_first'] if prep=='fixed' else ['quality'])]
 if comparisons is not None:combinations=comparisons
 seeds=seed if isinstance(seed,list) else [seed]
 for active_seed,rho,policy,cut in [(sd,*comb) for sd in seeds for comb in combinations]:
  name=f'{algorithm}_{prep}_rho{rho:.2f}_{policy}_depth{cut:g}'
  if replicate:name+=f'_seed{active_seed}'
  done=OUT/'vo'/f'{name}.done.json'
  if done.exists():
   meta=json.loads(done.read_text())
   if meta.get('frames')!=frames:raise ValueError(f'Existing run has a different frame count: {done}')
   if cut==50 or meta.get('recovery_overload')=='explicit_distanceThresh_keyword':continue
   obsolete=OUT/'obsolete_depth_binding_runs';obsolete.mkdir(exist_ok=True)
   for f in (OUT/'vo').glob(name+'*'):
    if f.is_file():f.replace(obsolete/f.name)
  states.append(dict(name=name,rho=rho,policy=policy,cutoff=cut,seed=active_seed,anchor=0,Twa=np.eye(4),Tw=np.eye(4),rows=[],moves=[],resets=0,unknown=0))
 if not states:print('Already complete',algorithm,prep,flush=True);return
 (OUT/'vo').mkdir(exist_ok=True,parents=True)
 checkpoint=OUT/'vo'/f'{algorithm}_{prep}{"_permutation_trials" if isinstance(seed,list) else ("_seed"+str(seed) if replicate else "")}.checkpoint.pkl'
 resume=0
 if checkpoint.exists():
  saved=pickle.loads(checkpoint.read_bytes())
  if saved.get('recovery_overload')=='explicit_distanceThresh_keyword' and {s['name'] for s in saved['states']}=={s['name'] for s in states}:
   states=saved['states'];resume=saved['next_frame'];print('RESUMING',algorithm,prep,resume,flush=True)
 @lru_cache(maxsize=128)
 def features(fid):return feature(prepared(fid,prep),det)
 @lru_cache(maxsize=28)
 def corners(fid):return legacy.flow_points(prepared(fid,prep),1800)
 dest=OUT/'vo';dest.mkdir(exist_ok=True)
 for fid in range(resume,frames):
  cur=prepared(fid,prep);t0=time.perf_counter();fc=features(fid);det_ms=(time.perf_counter()-t0)*1e3
  paircache={};geomcache={}
  for st in states:
   anchor=st['anchor'];gap=fid-anchor;accepted=0;reset=0;src='none';chosen=dict(ne=0,nc=0,rho=0.,parallax=float('nan'));status='initial_frame';dvalid=fvalid=False;pose_ms=0
   Tbefore=st['Tw'].copy()
   if fid:
    if anchor not in paircache:
     a,b=correspond(features(anchor),fc,norm,False)
     c,d=legacy.flow_pairs(prepared(anchor,prep),cur,corners(anchor),1.5)
     paircache[anchor]={'descriptor':(a,b),'flow':(c,d)}
    results={};start=time.perf_counter()
    for si,(label,(a,b)) in enumerate(paircache[anchor].items()):
     key=(anchor,label,st['cutoff'],st.get('seed',20261105))
     if key not in geomcache:geomcache[key]=pose(a,b,st['cutoff'],(st.get('seed',20261105)+fid*31+anchor*7+si)%2147483647,permute=replicate)
     results[label]=geomcache[key]
    pose_ms=(time.perf_counter()-start)*1000
    valid={label:r for label,r in results.items() if gate(r,st['rho'])=='accepted'}
    dvalid='descriptor' in valid;fvalid='flow' in valid
    if valid:
     def quality(item):
      label,r=item;return (min(r['nc'],200)*min(r['rho'],1)*min(max(r['parallax'],0),2)/math.sqrt(max(1,gap)),label=='descriptor')
     if st['policy']=='descriptor_first' and dvalid:src='descriptor';chosen=valid[src]
     else:src,chosen=max(valid.items(),key=quality)
     R,t=chosen['R'],chosen['t'];t=t/max(1e-12,np.linalg.norm(t));Tc=np.eye(4);Tc[:3,:3]=R;Tc[:3,3]=gap*t
     st['Tw']=st['Twa']@np.linalg.inv(Tc);accepted=1;status='accepted'
     st['moves'].append(dict(frame=fid,anchor=anchor,gap=gap,R=R.tolist(),direction=t.tolist(),anchor_world_R=st['Twa'][:3,:3].tolist()))
     st['anchor']=fid;st['Twa']=st['Tw'].copy()
    else:
     src,chosen=max(results.items(),key=lambda item:(item[1]['nc'],item[1]['ne'],len(paircache[anchor][item[0]][0])))
     status=gate(chosen,st['rho'])
     if gap>=20:reset=1;st['resets']+=1;st['unknown']+=gap;st['anchor']=fid;st['Twa']=st['Tw'].copy()
   Tw=st['Tw'];rr=dict(frame=fid,attempted_anchor=anchor,attempted_gap=gap,fresh=accepted,source=src,status=status,reset=reset,cumulative_resets=st['resets'],cumulative_unknown_frames=st['unknown'],features=len(fc[0]),ne=chosen['ne'],nc=chosen['nc'],rho=chosen['rho'],parallax_deg=chosen['parallax'],descriptor_valid=int(dvalid),flow_valid=int(fvalid),detection_ms=det_ms,pose_ms=pose_ms)
   rr.update({f'p{k}':float(Tw[k,3]) for k in range(3)});rr.update({f'R{i}{j}':float(Tw[i,j]) for i in range(3) for j in range(3)});st['rows'].append(rr)
  if fid%100==0:
   progress=dict(algorithm=algorithm,preprocessing=prep,frame=fid,seconds=time.perf_counter()-tstart,states=len(states),accepted={s['name']:sum(r['fresh'] for r in s['rows']) for s in states})
   dump(dest/f'{algorithm}_{prep}_progress.json',progress);print(algorithm,prep,fid,f'{progress["seconds"]:.1f}s',flush=True)
   tmp=checkpoint.with_suffix('.tmp');tmp.write_bytes(pickle.dumps({'next_frame':fid+1,'states':states,'recovery_overload':'explicit_distanceThresh_keyword'},protocol=5));tmp.replace(checkpoint)
 for st in states:
  writecsv(dest/(st['name']+'.csv'),st['rows']);dump(dest/(st['name']+'_moves.json'),st['moves']);dump(dest/(st['name']+'.done.json'),dict(name=st['name'],frames=frames,accepted=sum(r['fresh'] for r in st['rows']),seconds=time.perf_counter()-tstart,settings={k:st[k] for k in ['rho','policy','cutoff']},seed=st.get('seed',20261105),recovery_overload='explicit_distanceThresh_keyword',sampling='seeded_correspondence_permutation' if replicate else 'legacy_fixed_internal_RNG'))
 print('DONE',algorithm,prep,f'{time.perf_counter()-tstart:.1f}s',flush=True)

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('detector');p.add_argument('preprocessing');a=p.parse_args();choice=json.loads((OUT/'reference_choice.json').read_text())['best_geometric_hypothesis'];assert choice['order']=='xyzw' and choice['direction']=='twc';run(a.detector,a.preprocessing)
