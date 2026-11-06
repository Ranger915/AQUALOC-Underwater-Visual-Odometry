"""Reference-conditioned controls for an imposed-step pairwise reconstruction.

Uses saved trajectories, acceptance schedules and reference poses. Oracles are
privileged diagnostics, never deployable VO or guaranteed lower bounds.
"""
from dense_reference import *
import pandas as pd,itertools,math

R3=ROOT/'revision3_results';R3.mkdir(exist_ok=True)

def oracle_replays(m):
 n=len(FRAMES);p={k:np.zeros((n,3)) for k in ['replay','reference_heading_gap_step','estimated_heading_reference_step','reference_heading_and_step']};undefined=0
 for i in range(1,n):
  for x in p.values():x[i]=x[i-1]
  row=m['rows'][i]
  if not int(row['fresh']):continue
  a=int(row['attempted_anchor']);g=int(row['attempted_gap']);assert a<i and g==i-a
  d=m['p'][i]-m['p'][a];length=np.linalg.norm(d);assert abs(length-g)<1e-7
  u=d/length;rd=C[i]-C[a];rl=np.linalg.norm(rd)
  p['replay'][i]=p['replay'][a]+g*u
  p['estimated_heading_reference_step'][i]=p['estimated_heading_reference_step'][a]+rl*u
  p['reference_heading_and_step'][i]=p['reference_heading_and_step'][a]+rd
  if rl>1e-10:p['reference_heading_gap_step'][i]=p['reference_heading_gap_step'][a]+g*rd/rl
  else:undefined+=1;p['reference_heading_gap_step'][i]=p['reference_heading_gap_step'][a]
 assert np.max(np.abs(p.pop('replay')-m['p']))<1e-7
 return p,undefined

def fit_predict_reference(train,test,kind):
 if kind=='reference_hold':return np.repeat(C[train[-1]][None,:],len(test),axis=0)
 use=train[train>=train[-1]-100] if kind=='reference_cv_last5s' else train
 if len(use)<2:use=train[-2:]  # Sparse native-only sets may need a longer span.
 assert len(use)>=2
 x=np.column_stack([np.ones(len(use)),(use-train[-1])/20]);coef=np.linalg.lstsq(x,C[use],rcond=None)[0]
 return np.column_stack([np.ones(len(test)),(test-train[-1])/20])@coef

def paired_ranges(segments):
 rows=[];df=pd.DataFrame(segments)
 for (group,mask),dd in df.groupby(['group','mask']):
  for a,b in itertools.combinations(sorted(dd.configuration.unique()),2):
   aa=dd[dd.configuration==a].set_index('test_boundary');bb=dd[dd.configuration==b].set_index('test_boundary');ids=sorted(set(aa.index)&set(bb.index))
   aa=aa.loc[ids];bb=bb.loc[ids];assert aa.test_n.tolist()==bb.test_n.tolist()
   xa=aa.sse.to_numpy();xb=bb.sse.to_numpy();nn=aa.test_n.to_numpy();n=len(ids);effect=np.sqrt(xb.sum()/nn.sum())-np.sqrt(xa.sum()/nn.sum())
   for block in [1,2,3]:
    rng=np.random.default_rng(20261106);draws=[]
    for _ in range(2000):
     start=rng.integers(0,n,math.ceil(n/block));ix=np.concatenate([(s+np.arange(block))%n for s in start])[:n];draws.append(np.sqrt(xb[ix].sum()/nn[ix].sum())-np.sqrt(xa[ix].sum()/nn[ix].sum()))
    lo,hi=np.quantile(draws,[.025,.975]);rows.append(dict(group=group,mask=mask,a=a,b=b,segments=n,test_positions=int(nn.sum()),block_segments=block,rmse_b_minus_a=float(effect),low=float(lo),high=float(hi),replicates=2000))
 return rows

def main():
 src=pd.read_csv(R2/'dense_segments.csv');src=src[(src.fit_frames==800)&src.group.isin(['high_matrix_common','source_policies_depth1e9','source_policies_depth50'])];mods={name:model(OUT/'vo'/f'{name}.csv') for name in src.configuration.unique()};oracles={};checks=[]
 # A reference interpolated immediately before a test boundary can use the
 # supplied pose at that boundary. Restrict every new fit/control to centers
 # whose interpolation bracket is wholly before the boundary.
 src=src.copy();src['original_train_ids']=src.train_ids
 for index,row in src.iterrows():
  ids=np.array(list(map(int,row.train_ids.split(';'))));last_native=IDS[IDS<int(row.test_boundary)][-1];safe=ids[ids<=last_native]
  assert len(safe)>=20
  src.at[index,'train_ids']=';'.join(map(str,safe));src.at[index,'fit_n']=len(safe)
 for name,m in mods.items():
  oracles[name],undefined=oracle_replays(m);checks.append(dict(check='exact_saved_step_replay',configuration=name,passed=True,undefined_reference_headings=undefined))
 seg=[];err=[]
 def add(base,name,pred,scale=np.nan):
  train=np.array(list(map(int,base.train_ids.split(';'))));test=np.array(list(map(int,base.test_ids.split(';'))));e=np.linalg.norm(pred-C[test],axis=1)
  row=dict(group=base.group,mask=base.mask,configuration=name,test_boundary=int(base.test_boundary),fit_frames=800,fit_n=len(train),test_n=len(test),sse=float(e@e),rmse=float(np.sqrt(np.mean(e*e))),fitted_scale=float(scale),train_ids=base.train_ids,test_ids=base.test_ids)
  seg.append(row);err.extend(dict(group=base.group,mask=base.mask,configuration=name,test_boundary=int(base.test_boundary),frame=int(i),error=float(z)) for i,z in zip(test,e))
 for (group,mask,bound),dd in src.groupby(['group','mask','test_boundary']):
  assert dd.train_ids.nunique()==1 and dd.test_ids.nunique()==1
  base=next(dd.itertuples());train=np.array(list(map(int,base.train_ids.split(';'))));test=np.array(list(map(int,base.test_ids.split(';'))))
  for kind in ['reference_hold','reference_cv_all40s','reference_cv_last5s']:add(base,kind,fit_predict_reference(train,test,kind))
  for row in dd.itertuples():
   m=mods[row.configuration];s,R,t=fit(m['p'][train],C[train]);pred=s*(m['p'][test]@R.T)+t;add(row,m['name'],pred,s)
   oldtrain=np.array(list(map(int,row.original_train_ids.split(';'))));old_s,old_R,old_t=fit(m['p'][oldtrain],C[oldtrain]);old_e=np.linalg.norm(old_s*(m['p'][test]@old_R.T)+old_t-C[test],axis=1)
   assert abs(old_e@old_e-row.sse)<1e-8
   for kind,p in oracles[m['name']].items():
    s,R,t=fit(p[train],C[train]);assert np.isfinite(s) and s>0;add(row,m['name']+'__'+kind,s*(p[test]@R.T)+t,s)
 writecsv(R3/'motion_control_segments.csv',seg);writecsv(R3/'motion_control_per_pose.csv',err)
 summ=[]
 for (group,mask,name),dd in pd.DataFrame(seg).groupby(['group','mask','configuration']):
  summ.append(dict(group=group,mask=mask,configuration=name,segments=len(dd),test_positions=int(dd.test_n.sum()),rmse_m=float(np.sqrt(dd.sse.sum()/dd.test_n.sum()))))
 writecsv(R3/'motion_control_summary.csv',summ)
 # Cross-detector comparisons already share the exact nine-way sets. Recompute
 # all paired fixed-detector contrasts with matching resamples and all blocks.
 fixed=[r for r in seg if r['group']=='high_matrix_common' and r['configuration'] in [f'{alg}_fixed_rho0.06_quality_depth1e+09' for alg in ['ORB','AKAZE','SIFT']]]
 writecsv(R3/'cross_detector_paired_ranges.csv',paired_ranges(fixed))
 writecsv(R3/'matrix_all_paired_ranges.csv',paired_ranges([r for r in seg if r['group']=='high_matrix_common' and '__' not in r['configuration'] and not r['configuration'].startswith('reference_')]))
 writecsv(R3/'source_policy_paired_ranges.csv',paired_ranges([r for r in seg if r['group'].startswith('source_policies_') and '__' not in r['configuration'] and not r['configuration'].startswith('reference_')]))
 # Pair the fixed ORB reconstruction and its controls; these ranges are descriptive.
 key='ORB_fixed_rho0.06_quality_depth1e+09'
 controls=[r for r in seg if r['group']=='high_matrix_common' and (r['configuration'].startswith(key) or r['configuration'].startswith('reference_'))]
 writecsv(R3/'baseline_paired_ranges.csv',paired_ranges(controls))
 # Native supplied centers only, fitting and testing without interpolation.
 native=[];coverage=[]
 names=[f'{alg}_{prep}_rho0.06_quality_depth1e+09' for alg in ['ORB','AKAZE','SIFT'] for prep in ['raw','fixed','adaptive']]
 for clean in [False,True]:
  mask='robust_reference_edges_excluded' if clean else 'all_reference';ids=IDS[np.logical_and.reduce([mods[name]['fresh'][IDS] for name in names]) & (CLEAN[IDS] if clean else True)];skipped=0
  for bound in range(800,len(FRAMES),200):
   train=ids[(ids>=bound-800)&(ids<bound)];test=ids[(ids>=bound)&(ids<min(bound+200,len(FRAMES)))];rows=[]
   if len(train)<6 or len(test)<1:skipped+=1;continue
   for name in names:
    p=mods[name]['p'];sp=np.linalg.svd(p[train]-p[train].mean(0),compute_uv=False);sq=np.linalg.svd(C[train]-C[train].mean(0),compute_uv=False)
    if min(sp[1],sq[1])<1e-10:break
    s,R,t=fit(p[train],C[train]);e=np.linalg.norm(s*(p[test]@R.T)+t-C[test],axis=1)
    rows.append(dict(group='native_only_common',mask=mask,configuration=name,test_boundary=bound,fit_n=len(train),test_n=len(test),sse=float(e@e),rmse=float(np.sqrt(np.mean(e*e))),train_ids=';'.join(map(str,train)),test_ids=';'.join(map(str,test))))
   if len(rows)!=9:skipped+=1;continue
   for kind in ['reference_hold','reference_cv_all40s','reference_cv_last5s']:
    e=np.linalg.norm(fit_predict_reference(train,test,kind)-C[test],axis=1);rows.append(dict(group='native_only_common',mask=mask,configuration=kind,test_boundary=bound,fit_n=len(train),test_n=len(test),sse=float(e@e),rmse=float(np.sqrt(np.mean(e*e))),train_ids=';'.join(map(str,train)),test_ids=';'.join(map(str,test))))
   native.extend(rows)
  coverage.append(dict(mask=mask,common_native_positions=len(ids),minimum_fit_positions=6,minimum_test_positions=1,skipped_windows=skipped))
 writecsv(R3/'native_only_segments.csv',native);writecsv(R3/'native_only_coverage.csv',coverage)
 ns=[]
 for (mask,name),dd in pd.DataFrame(native).groupby(['mask','configuration']):ns.append(dict(mask=mask,configuration=name,segments=len(dd),test_positions=int(dd.test_n.sum()),rmse_m=float(np.sqrt(dd.sse.sum()/dd.test_n.sum()))))
 writecsv(R3/'native_only_summary.csv',ns)
 writecsv(R3/'native_cross_detector_ranges.csv',paired_ranges([r for r in native if r['configuration'] in [f'{a}_fixed_rho0.06_quality_depth1e+09' for a in ['ORB','AKAZE','SIFT']]]))
 for r in native:
  assert all(int(i)%20==0 for i in (r['train_ids']+';'+r['test_ids']).split(';'))
 checks.extend([dict(check='matched_original_dense_SSE_before_causal_refit',passed=True),dict(check='all_new_fit_reference_brackets_precede_test_boundary',passed=True),dict(check='native_fit_and_test_exclude_all_interpolation',passed=True)])
 writecsv(R3/'motion_controls_validation.csv',checks)
 dump(R3/'motion_controls_protocol.json',dict(scope='as-built exploratory controls; reference-relative reconstruction consistency, not VO accuracy',reference_baselines='hold last training center; least-squares CV on all training centers or final 5 seconds (last two available training positions if fewer than two fall in that span); no test reference input',oracles='future reference directions and/or step lengths, same saved accepted anchors and missing-motion schedule; Sim(3) fit only on training positions; not deployable and not guaranteed numerical lower bounds',dense_sets='prior nine-way and source-policy held-out sets and calendar windows; all new training sets exclude frames whose reference interpolation bracket reaches the test boundary; 40 s fit, 10 s test',native_only='fit and test only supplied centers; same nine-way intersection; fit >=6 native poses, test >=1, rank >=2; sparse sensitivity, not equal-information replication',uncertainty='paired circular segment-block resampling, blocks 1/2/3, 2000 replicates; descriptive single-sequence sensitivity, no population confidence claim',validation='saved-step replay exactly reproduces each input trajectory; original dense SSE matched before causal refit; native-only IDs verified'))
 print('Motion controls complete:',len(seg),'segments,',len(err),'residual rows;',len(native),'native segments',flush=True)

if __name__=='__main__':main()
