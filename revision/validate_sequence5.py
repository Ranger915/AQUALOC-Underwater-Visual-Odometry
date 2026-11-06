"""Validate frozen transfer, reference domains, and paired exported errors."""
from common import *
import pandas as pd
DEST=ROOT/'sequence5_results'

def main():
 checks=[]
 def check(label,ok):
  assert bool(ok),label;checks.append(dict(check=label,passed=True))
 paths=list((DEST/'vo').glob('*.done.json'));check('twenty completed configurations',len(paths)==20);runs={}
 for p in paths:
  m=json.loads(p.read_text());d=pd.read_csv(p.with_name(p.name.replace('.done.json','.csv')));runs[m['name']]=d
  check('complete ordered frames '+m['name'],m['frames']==3983 and len(d)==3983 and np.array_equal(d.frame,np.arange(3983)))
  check('acceptance and cap '+m['name'],int(d.fresh.sum())==m['accepted'] and d.features.le(1500).all() and m['recovery_overload']=='explicit_distanceThresh_keyword')
  xyz=d[['p0','p1','p2']].to_numpy();R=d[[f'R{i}{j}' for i in range(3) for j in range(3)]].to_numpy().reshape(-1,3,3);check('finite rigid poses '+m['name'],np.isfinite(xyz).all() and np.isfinite(R).all() and np.max(abs(R.transpose(0,2,1)@R-np.eye(3)))<1e-7 and np.allclose(np.linalg.det(R),1,atol=1e-7))
  accepted=d[d.fresh==1];check('frozen gates '+m['name'],accepted['ne'].ge(15).all() and accepted.nc.ge(10).all() and accepted.rho.ge(.06).all() and accepted.parallax_deg.ge(.1).all() and (accepted.attempted_gap==accepted.frame-accepted.attempted_anchor).all())
 for cut in ['50','1e+09']:
  q=runs[f'ORB_fixed_rho0.06_quality_depth{cut}'];d=runs[f'ORB_fixed_rho0.06_descriptor_first_depth{cut}'];check('same accepted anchors for source policies '+cut,np.array_equal(q[['fresh','attempted_anchor','reset']],d[['fresh','attempted_anchor','reset']]))
 decisions=pd.read_csv(DEST/'adaptive_frame_decisions.csv');check('frozen 10.5 switch',len(decisions)==3983 and np.array_equal(decisions.decision=='raw',decisions.local_std>=10.5) and sum(decisions.decision=='raw')==107)
 check('original prefix regression',json.loads((DEST/'sequence6_regression/validation.json').read_text())['passed'])
 source=Path(r'C:\Users\Raaghav\Downloads\archaeo_sequence_5_raw_data/raw_data');timestamps=readcsv(source/'img_sequence_5.csv');check('image timestamp association',[list(r.values())[1].strip() for r in timestamps]==[f'frame{i:06d}.png' for i in range(3983)])
 t5=[int(list(r.values())[0]) for r in timestamps];t6=[int(list(r.values())[0]) for r in readcsv(DATA/'raw_data/img_sequence_6.csv')];check('monotone and disjoint sequence timestamps',np.all(np.diff(t5)>0) and (t5[-1]<t6[0] or t6[-1]<t5[0]))
 for seq in [5,6]:
  newfile=DEST/'reference'/f'new_archaeo_colmap_traj_sequence_{seq:02d}.txt';new=refposes(file=newfile);old=refposes(file=DEST/'reference/colmap_traj_sequence_5.txt' if seq==5 else DATA/'colmap_traj_sequence_6.txt');ids=sorted(new);s,R,t=fit(np.array([old[i][1] for i in ids]),np.array([new[i][1] for i in ids]));pos=max(np.linalg.norm(s*R@old[i][1]+t-new[i][1]) for i in ids);rot=max(np.linalg.norm(R@old[i][0]-new[i][0]) for i in ids)
  check('global reference Sim(3) and rotations '+str(seq),pos<1e-8 and rot<1e-8)
  dump(DEST/'reference'/f'reference_{seq}_shape_comparison.json',dict(common_poses=len(ids),old_to_new_scale=float(s),max_aligned_position_difference_m=float(pos),max_world_rotation_matrix_difference=float(rot),interpretation='reference-to-reference alignment on all retained poses; establishes score invariance, not estimator accuracy or physical truth'))
 for version in ['evaluation_updated','evaluation_historical','sequence6_updated']:
  folder=DEST/version;meta=json.loads((folder/'reference.json').read_text());native=set(refposes(file=meta['source']));seg=pd.read_csv(folder/'segments.csv');err=pd.read_csv(folder/'per_pose.csv');keys=['group','mask','configuration','segment'];joined=seg.set_index(keys).join(err.groupby(keys).error.agg(n='size',ss=lambda z:float(z@z)))
  check('per-pose counts '+version,np.array_equal(joined.test_n,joined.n));check('per-pose SSE '+version,np.allclose(joined.sse,joined.ss,rtol=1e-10,atol=1e-10))
  for (group,mask,bound),d in seg.groupby(['group','mask','test_boundary']):
   train=list(map(int,d.train_ids.iloc[0].split(';')));test=list(map(int,d.test_ids.iloc[0].split(';')));check('paired support '+version+'/'+group+'/'+mask+'/'+str(bound),d.train_ids.nunique()==d.test_ids.nunique()==1 and min(train)>=meta['first'] and bound-800>=meta['first'] and max(train)<=max(i for i in native if i<bound)<min(test))
   if group=='native_only_common':check('native-only '+version+'/'+mask+'/'+str(bound),set(train+test)<=native)
  controls=pd.read_csv(folder/'control_segments.csv');allseg=pd.concat([seg,controls],ignore_index=True);summary=pd.read_csv(folder/'summary.csv')
  for r in summary.itertuples():
   d=allseg[(allseg.group==r.group)&(allseg['mask']==r.mask)&(allseg.configuration==r.configuration)];check('pooled summary '+version+'/'+r.group+'/'+r.mask+'/'+r.configuration,int(d.test_n.sum())==r.test_positions and abs(np.sqrt(d.sse.sum()/d.test_n.sum())-r.rmse_m)<1e-10)
  for r in pd.read_csv(folder/'paired_ranges.csv').itertuples():
   d=seg[(seg.group==r.group)&(seg['mask']==r.mask)];a=d[d.configuration==r.a];b=d[d.configuration==r.b];effect=np.sqrt(b.sse.sum()/b.test_n.sum())-np.sqrt(a.sse.sum()/a.test_n.sum());check('paired effect '+version+'/'+r.group+'/'+r.mask+'/'+r.a+'/'+r.b+'/'+str(r.block_segments),abs(effect-r.rmse_b_minus_a)<1e-10 and r.replicates==2000 and r.low<=r.high)
  for r in controls.itertuples():
   train=list(map(int,r.train_ids.split(';')));test=list(map(int,r.test_ids.split(';')));check('control domain '+version+'/'+r.group+'/'+r.mask+'/'+r.configuration+'/'+str(r.test_boundary),min(train)>=meta['first'] and max(train)<=max(i for i in native if i<r.test_boundary)<min(test))
 # Matched windows in Seq05 differ only by the publisher's global gauge/scale.
 a=pd.read_csv(DEST/'evaluation_historical/segments.csv');b=pd.read_csv(DEST/'evaluation_updated/segments.csv');keys=['group','mask','configuration','test_boundary'];d=a.merge(b,on=keys,suffixes=('_old','_new'));scale=json.loads((DEST/'reference/reference_5_shape_comparison.json').read_text())['old_to_new_scale'];check('matched-reference identity of sets',d.train_ids_old.eq(d.train_ids_new).all() and d.test_ids_old.eq(d.test_ids_new).all());check('matched-reference SSE scales as squared metric scale',np.allclose(d.sse_new,scale**2*d.sse_old,rtol=1e-9,atol=1e-9))
 front=pd.read_csv(DEST/'frontend_transfer_summary.csv');check('all eighteen matching summaries',len(front)==18 and front[front.sequence==5].n.eq(16653).all() and front[front.sequence==6].n.eq(13695).all());check('endpoint identity',pd.read_csv(DEST/'frontend_transfer_identity.csv').enhanced_identity.all());check('epipolar invariance',pd.read_csv(DEST/'sequence6_reference_epipolar_invariance.csv').inliers_identical.all())
 for alg in ['ORB','AKAZE','SIFT']:check('new-data exact matcher '+alg,all(r['identical'] for r in json.loads((DEST/f'frontend_{alg}_validation.json').read_text())['matcher_checks']))
 check('corrected cutoff engine reproduces saved choices',json.loads((DEST/'cutoff_corrected_sequence6/same_anchor_cutoff_probe_validation.json').read_text())['baseline_acceptance_or_source_mismatches']==0)
 writecsv(DEST/'validation.csv',checks);dump(DEST/'validation_summary.json',dict(passed=True,checks=len(checks),completed_new_estimator_runs=20,new_images=3983,reference='publisher-corrected primary; historical replication separate',scope='within-site sequence transfer; numerical validation, not physical ground truth',established_system_baseline='not completed',git_push=False));print('PASS',len(checks),'transfer checks',flush=True)

if __name__=='__main__':main()
