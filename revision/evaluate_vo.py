from common import *
import itertools
REF=refposes();RID=sorted(REF)
REFPATH=float(sum(np.linalg.norm(REF[b][1]-REF[a][1]) for a,b in zip(RID[:-1],RID[1:])))
def position(r):return np.array([float(r[f'p{k}']) for k in range(3)])
def rotation(r):return np.array([[float(r[f'R{i}{j}']) for j in range(3)] for i in range(3)])
def angle(a,b):
 na,nb=np.linalg.norm(a),np.linalg.norm(b)
 return float(np.degrees(np.arccos(np.clip(a@b/(na*nb),-1,1)))) if min(na,nb)>1e-10 else float('nan')
def refcenter(frame):
 return np.array([np.interp(frame,RID,[REF[i][1][k] for i in RID]) for k in range(3)])
def replay(path,mode):
 src=readcsv(path);moves=json.loads(path.with_name(path.stem+'_moves.json').read_text());mm={m['frame']:m for m in moves};by={};out=[]
 for r in src:
  fid=int(r['frame']);r=dict(r)
  if fid==0:R=np.eye(3);C=np.zeros(3)
  elif fid in mm:
   m=mm[fid];anc=m['anchor'];Ra,Ca=by[anc];rel=np.array(m['R']);direction=np.array(m['direction']);g=m['gap']
   scale=1 if mode=='unit' else (np.sqrt(g) if mode=='sqrt_gap' else float(np.linalg.norm(refcenter(fid)-refcenter(anc))))
   R=Ra@rel.T;C=Ca-Ra@rel.T@(scale*direction)
  by[fid]=(R.copy(),C.copy());r.update({f'p{k}':C[k] for k in range(3)});r.update({f'R{i}{j}':R[i,j] for i in range(3) for j in range(3)});out.append(r)
 target=path.with_name(path.stem+'_scale_'+mode+'.csv');writecsv(target,out)
 return target
def model(path):
 rr=readcsv(path);by={int(r['frame']):r for r in rr};fresh=sorted(i for i in RID if i in by and int(by[i]['fresh'])==1)
 return dict(name=path.stem,by=by,fresh=fresh,rows=rr)
def evaluate_window(mod,train,test,group,k,segment):
 by=mod['by'];p=np.array([position(by[i]) for i in train]);q=np.array([REF[i][1] for i in train])
 if np.linalg.matrix_rank(p-p.mean(0),tol=1e-10)<2:return [],None
 s,R,t=fit(p,q)
 if not np.isfinite(s) or s<=0:return [],None
 errs=[]
 for i in test:
  residual=s*R@position(by[i])+t-REF[i][1];r=by[i]
  e=dict(group=group,configuration=mod['name'],k=k,segment=segment,frame=i,fit_last=train[-1],error=float(np.linalg.norm(residual)),residual_x=residual[0],residual_y=residual[1],residual_z=residual[2],fitted_scale=float(s),resets_since_fit=int(r.get('cumulative_resets',0))-int(by[train[-1]].get('cumulative_resets',0)),unknown_frames_since_fit=int(r.get('cumulative_unknown_frames',0))-int(by[train[-1]].get('cumulative_unknown_frames',0)),elapsed_image_frames=i-train[-1]);errs.append(e)
 e=np.array([r['error'] for r in errs]);sp=np.linalg.svd(p-p.mean(0),compute_uv=False);sq=np.linalg.svd(q-q.mean(0),compute_uv=False);seg=dict(group=group,configuration=mod['name'],k=k,segment=segment,train_n=len(train),test_n=len(test),train_first=train[0],train_last=train[-1],test_first=test[0],test_last=test[-1],train_ids=';'.join(map(str,train)),test_ids=';'.join(map(str,test)),source_scatter_smallest_largest=float(sp[-1]/max(sp[0],1e-20)),reference_scatter_smallest_largest=float(sq[-1]/max(sq[0],1e-20)),rmse=float(np.sqrt(np.mean(e*e))),rmse_fraction_full_reference_path=float(np.sqrt(np.mean(e*e))/REFPATH))
 # Relative motions between successive fresh held-out reference poses.
 rel=[]
 for a,b in zip(test[:-1],test[1:]):
  Ra,Ca=REF[a];Rb,Cb=REF[b];Ea=rotation(by[a]);Eb=rotation(by[b]);refdir=Ra.T@(Cb-Ca);estdir=Ea.T@(position(by[b])-position(by[a]));rot=degrees((Ra.T@Rb).T@(Ea.T@Eb))
  rel.append((rot,angle(refdir,estdir),float(np.linalg.norm(s*estdir-refdir))))
 if rel:
  rel=np.array(rel);seg.update(rotation_error_median_deg=float(np.nanmedian(rel[:,0])),integrated_direction_error_median_deg=float(np.nanmedian(rel[:,1])),relative_translation_rmse=float(np.sqrt(np.mean(rel[:,2]**2))))
 return errs,seg
def rolling(models,ids,group):
 errors=[];segments=[];coverage=[]
 for k in [12,24,36]:
  count=0
  for start in range(k,len(ids)-5,6):
   train=ids[start-k:start];test=ids[start:start+6]
   for m in models:
    ee,ss=evaluate_window(m,train,test,group,k,count);errors+=ee
    if ss:segments.append(ss)
   count+=1
  coverage.append(dict(group=group,k=k,common_fresh=len(ids),requested_segments=count,requested_test_poses=count*6,configurations=len(models),status='computed' if count else 'insufficient_common_fresh_poses'))
 return errors,segments,coverage
def paired_ci(segments):
 rr=[];rng=np.random.default_rng(20261105)
 groups={}
 for r in segments:groups.setdefault((r['group'],r['k']),{}).setdefault(r['configuration'],{})[r['segment']]=r['rmse']**2
 for (group,k),mods in groups.items():
  if group=='per_configuration':continue
  names=list(mods)
  for a,b in itertools.combinations(names,2):
   ids=sorted(set(mods[a])&set(mods[b]));n=len(ids)
   if not n:continue
   x=np.array([mods[a][i] for i in ids]);y=np.array([mods[b][i] for i in ids]);effect=float(np.sqrt(y.mean())-np.sqrt(x.mean()))
   for block in [1,2,3]:
    vals=[]
    if n>=4:
     for rep in range(1000):
      starts=rng.integers(0,n,size=math.ceil(n/block));ii=np.concatenate([(j+np.arange(block))%n for j in starts])[:n];vals.append(np.sqrt(y[ii].mean())-np.sqrt(x[ii].mean()))
    lo,hi=(np.quantile(vals,[.025,.975]) if vals else [float('nan')]*2)
    rr.append(dict(group=group,k=k,a=a,b=b,segments=n,block_segments=block,rmse_b_minus_a=effect,ci_low=float(lo),ci_high=float(hi),replicates=len(vals),interpretation='within-sequence resampling sensitivity; overlapping alignment windows; not independent-sequence confidence'))
 return rr
def main():
 paths=[]
 for meta in (OUT/'vo').glob('*.done.json'):
  info=json.loads(meta.read_text());cut=info.get('settings',{}).get('cutoff',50)
  if cut!=50 and info.get('recovery_overload')!='explicit_distanceThresh_keyword':continue
  p=meta.with_name(meta.name.removesuffix('.done.json')+'.csv')
  if p.exists():paths.append(p)
 expanded=list(paths)
 for p in list(paths):
  if p.stem.startswith('ORB_fixed') and p.with_name(p.stem+'_moves.json').exists():
   for mode in ['unit','sqrt_gap','reference_step_oracle']:expanded.append(replay(p,mode))
 paths=expanded;mods=[model(p) for p in paths];summary=[]
 for m in mods:
  fresh=m['fresh'];n=len(m['rows']);accepted=sum(int(r['fresh']) for r in m['rows']);row=dict(configuration=m['name'],output_rows=n,accepted=accepted,attempted=3384,acceptance=accepted/3384,fresh_reference=len(fresh),reference_path_length=REFPATH)
  if len(fresh)>=18:
   cut=int(.6*len(fresh));e,seg=evaluate_window(m,fresh[:cut],fresh[cut:],'legacy_60_40',cut,0)
   if seg:row.update(legacy_test_n=len(fresh)-cut,legacy_heldout_rmse=seg['rmse'],legacy_rmse_fraction_path=seg['rmse']/REFPATH)
  summary.append(row)
 writecsv(OUT/'vo_configuration_summary.csv',[{k:r.get(k,'') for k in sorted(set().union(*(r.keys() for r in summary)))} for r in summary])
 mainmods=[m for m in mods if '_rho0.06_quality_depth50' in m['name'] and '_scale_' not in m['name']]
 errors=[];segments=[];coverage=[]
 def add(mm,ids,g):
  e,s,c=rolling(mm,ids,g);errors.extend(e);segments.extend(s);coverage.extend(c)
 if mainmods:
  common=sorted(set.intersection(*(set(m['fresh']) for m in mainmods)));add(mainmods,common,'all_matrix_common')
 highmods=[m for m in mods if '_rho0.06_quality_depth1e+09' in m['name'] and '_scale_' not in m['name']]
 if highmods:
  common=sorted(set.intersection(*(set(m['fresh']) for m in highmods)));add(highmods,common,'high_depth_matrix_common')
 for depth in ['50','1e+09']:
  sweep=[m for m in mods if m['name'].startswith('ORB_fixed_rho') and m['name'].endswith('_quality_depth'+depth)]
  if len(sweep)==4:
   common=sorted(set.intersection(*(set(m['fresh']) for m in sweep)));add(sweep,common,'rho_sweep_common_depth'+depth)
 for m in mods:add([m],m['fresh'],'per_configuration__'+m['name'])
 baseline=next((m for m in mods if m['name']=='ORB_fixed_rho0.06_quality_depth50'),None)
 if baseline:
  for m in mods:
   if m is baseline:continue
   common=sorted(set(baseline['fresh'])&set(m['fresh']));add([baseline,m],common,'paired_with_ORB_fixed__'+m['name'])
  # Original-figure diagnostic: fit earlier fresh poses but export residuals for
  # every reference ID, explicitly flagging stale output rows as diagnostics.
  ids=baseline['fresh'];cut=int(.6*len(ids));train=ids[:cut];allids=[i for i in RID if i in baseline['by']];e,_=evaluate_window(baseline,train,allids,'boundary_diagnostic',len(train),0)
  for r in e:r['fresh']=int(baseline['by'][r['frame']]['fresh']);r['split']='fit' if r['frame'] in train else 'later'
  writecsv(OUT/'boundary_error_and_missing_motion.csv',e)
  gap={}
  for r in baseline['rows'][1:]:
   g=int(r['attempted_gap']);gap.setdefault(g,[]).append(int(r['fresh']))
  writecsv(OUT/'acceptance_by_attempted_gap.csv',[dict(gap=g,attempts=len(a),accepted=sum(a),acceptance=np.mean(a)) for g,a in sorted(gap.items())])
 writecsv(OUT/'rolling_per_pose_errors.csv',errors)
 # Normalize heterogeneous optional fields in segment rows.
 keys=sorted(set().union(*(r.keys() for r in segments))) if segments else []
 writecsv(OUT/'rolling_segment_metrics.csv',[{k:r.get(k,'') for k in keys} for r in segments]);writecsv(OUT/'rolling_evaluation_coverage.csv',coverage);writecsv(OUT/'paired_segment_bootstrap.csv',paired_ci(segments))
 dump(OUT/'trajectory_metric_definitions.json',dict(full_reference_path_length=REFPATH,reference_positions=170,reference_units='local reconstruction units',normalization='RMSE divided by full 170-position reference path length',oracle='privileged linear interpolation of the 170 reference centers at arbitrary image endpoints; not true measured per-frame scale',direction='integrated inter-pose direction can still depend on per-step magnitudes; direct two-view direction is evaluated separately',rolling='12/24/36 earlier common fresh poses; next 6 non-overlapping test poses; identical window IDs within each paired comparison',common_set_limit='empty or small common sets are reported, never filled using stale poses'))
 print('Evaluated',len(mods),'configurations;',len(segments),'segments;',len(errors),'pose errors; path',REFPATH,flush=True)
if __name__=='__main__':
 import math
 main()
