from dense_reference import *
import itertools,argparse,math
def eval_group(mods,group,clean=False,fit_lengths=(400,800,1200),collect=True,safe_reference=False,min_fit=20,min_test=10):
 ids=FRAMES[(FRAMES>=IDS[0])&np.logical_and.reduce([m['fresh'] for m in mods])&(CLEAN if clean else True)];segments=[];errors=[];coverage=[]
 for fitframes in fit_lengths:
  count=0;skipped=0
  for boundary in range(fitframes,len(FRAMES),200):
   if boundary-fitframes<IDS[0]:skipped+=1;continue
   train=ids[(ids>=boundary-fitframes)&(ids<boundary)];test=ids[(ids>=boundary)&(ids<min(boundary+200,len(FRAMES)))];temp=[]
   if safe_reference:train=train[train<=IDS[IDS<boundary][-1]]
   ok=len(train)>=min_fit and len(test)>=min_test
   if not ok:skipped+=1;continue
   for m in mods:
    p=m['p'][train];q=C[train];sp=np.linalg.svd(p-p.mean(0),compute_uv=False);sq=np.linalg.svd(q-q.mean(0),compute_uv=False)
    if sp[1]<1e-10 or sq[1]<1e-10:ok=False;break
    scale,align,t=fit(p,q)
    if not np.isfinite(scale) or scale<=0:ok=False;break
    residual=scale*(m['p'][test]@align.T)+t-C[test];e=np.linalg.norm(residual,axis=1);rot=[];direction=[];relative=[]
    # Fixed one-second endpoints from fresh held-out frames, rather than an
    # estimator-dependent succession of tiny intervals.
    testset=set(test.tolist())
    for a in test:
     b=int(a)+20
     if b not in testset:continue
     Rref=ROTS[a].T@ROTS[b];Rest=m['R'][a].T@m['R'][b];rot.append(degrees(Rref.T@Rest));refdelta=ROTS[a].T@(C[b]-C[a]);estdelta=m['R'][a].T@(m['p'][b]-m['p'][a]);direction.append(angle(refdelta,estdelta));relative.append(np.linalg.norm(scale*estdelta-refdelta))
    row=dict(group=group,mask='robust_reference_edges_excluded' if clean else 'all_reference',configuration=m['name'],fit_frames=fitframes,test_boundary=boundary,test_end=min(boundary+200,len(FRAMES))-1,segment=count,fit_n=len(train),test_n=len(test),sse=float(e@e),rmse=float(np.sqrt(np.mean(e*e))),rmse_fraction_path=float(np.sqrt(np.mean(e*e))/PATH),rmse_fraction_diagonal=float(np.sqrt(np.mean(e*e))/DIAG),fitted_scale=float(scale),source_scatter_ratio=float(sp[-1]/max(sp[0],1e-30)),reference_scatter_ratio=float(sq[-1]/max(sq[0],1e-30)),rotation_error_median_deg=float(np.median(rot)) if rot else float('nan'),direction_error_median_deg=float(np.nanmedian(direction)) if direction else float('nan'),relative_translation_rmse_m=float(np.sqrt(np.mean(np.array(relative)**2))) if relative else float('nan'),relative_pairs=len(rot),train_ids=';'.join(map(str,train)),test_ids=';'.join(map(str,test)))
    ee=[dict(group=group,mask=row['mask'],configuration=m['name'],fit_frames=fitframes,segment=count,frame=int(frame),fit_boundary=boundary,error=float(ez),residual_x=float(r[0]),residual_y=float(r[1]),residual_z=float(r[2]),reference_interpolated=int(frame%20!=0)) for frame,ez,r in zip(test,e,residual)] if collect else []
    temp.append((row,ee))
   if not ok:skipped+=1;continue
   for row,ee in temp:segments.append(row);errors.extend(ee)
   count+=1
  coverage.append(dict(group=group,mask='robust_reference_edges_excluded' if clean else 'all_reference',fit_frames=fitframes,common_fresh=len(ids),segments=count,skipped_windows=skipped,configurations=len(mods)))
 return segments,errors,coverage
def boot(segments,replicates=1000):
 rr=[];groups={}
 for r in segments:groups.setdefault((r['group'],r['mask'],r['fit_frames']),{}).setdefault(r['configuration'],{})[r['segment']]=r
 for (group,mask,f),mods in groups.items():
  for a,b in itertools.combinations(sorted(mods),2):
   ids=sorted(set(mods[a])&set(mods[b]));n=len(ids)
   if not n:continue
   xa=np.array([mods[a][i]['sse'] for i in ids]);xb=np.array([mods[b][i]['sse'] for i in ids]);nn=np.array([mods[a][i]['test_n'] for i in ids]);assert np.array_equal(nn,[mods[b][i]['test_n'] for i in ids]);effect=float(np.sqrt(xb.sum()/nn.sum())-np.sqrt(xa.sum()/nn.sum()))
   for block in [1,2,3]:
    vals=[];rng=np.random.default_rng(20261105)
    if n>=4:
     for rep in range(replicates):
      starts=rng.integers(0,n,size=math.ceil(n/block));ix=np.concatenate([(j+np.arange(block))%n for j in starts])[:n];vals.append(np.sqrt(xb[ix].sum()/nn[ix].sum())-np.sqrt(xa[ix].sum()/nn[ix].sum()))
    lo,hi=np.quantile(vals,[.025,.975]) if vals else [np.nan,np.nan];rr.append(dict(group=group,mask=mask,fit_frames=f,a=a,b=b,segments=n,test_positions=int(nn.sum()),block_segments=block,rmse_b_minus_a=effect,low=float(lo),high=float(hi),replicates=len(vals)))
 return rr
def main(seeds=False):
 save_reference_audit();mods={p.stem:model(p) for p in (OUT/'vo').glob('*.csv') if '_scale_' not in p.stem and p.with_suffix('.done.json').exists()};groups=[]
 high=[mods[f'{alg}_{prep}_rho0.06_quality_depth1e+09'] for alg in ['ORB','AKAZE','SIFT'] for prep in ['raw','fixed','adaptive']]
 base=mods['ORB_fixed_rho0.06_quality_depth1e+09'];desc=mods['ORB_fixed_rho0.06_descriptor_first_depth1e+09'];low=mods['ORB_fixed_rho0.06_quality_depth50'];lowdesc=mods['ORB_fixed_rho0.06_descriptor_first_depth50']
 if seeds:
  for p in (R2/'seed_order_runs/vo').glob('*.done.json'):
   q=p.with_name(p.name.replace('.done.json','.csv'));mods[q.stem]=model(q)
  sets=[]
  for seed in [20261105,20261106,20261107,20261108,20261109]:
   suffix='' if seed==20261105 else f'_seed{seed}'
   names=[f'ORB_{p}_rho0.06_quality_depth1e+09'+suffix for p in ['raw','fixed','adaptive']]+['ORB_fixed_rho0.06_descriptor_first_depth1e+09'+suffix]
   if not all(n in mods for n in names):continue
   sets.extend([mods[n] for n in names]);groups.append((f'seed_{seed}_paired',[mods[names[1]],mods[names[3]]]))
  if len(sets)==20:groups.append(('five_trial_common',sets))
  stem='seed_dense'
 else:
  groups=[('high_matrix_common',high),('source_policies_depth1e9',[base,desc]),('source_policies_depth50',[low,lowdesc])]
  for m in high:
   if m['name']!=base['name']:groups.append(('pair_with_ORB_fixed__'+m['name'],[base,m]))
  stem='dense'
 seg=[];err=[];cov=[]
 for name,mm in groups:
  for clean in ([False] if seeds else [False,True]):
   a,b,c=eval_group(mm,name,clean=clean,fit_lengths=(800,) if seeds else (400,800,1200));seg+=a;err+=b;cov+=c
 writecsv(R2/(stem+'_segments.csv'),seg);writecsv(R2/(stem+'_per_pose.csv'),err);writecsv(R2/(stem+'_coverage.csv'),cov);writecsv(R2/(stem+'_paired_ranges.csv'),boot(seg));print(stem,'segments',len(seg),'residual rows',len(err),flush=True)
if __name__=='__main__':
 import math
 p=argparse.ArgumentParser();p.add_argument('--seeds',action='store_true');a=p.parse_args();main(a.seeds)
