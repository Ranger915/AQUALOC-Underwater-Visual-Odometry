"""Reuse the existing evaluator and controls on either supplied reference export."""
import os,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];DEST=ROOT/'sequence5_results';seq=int(sys.argv[2]) if len(sys.argv)>2 else 5;assert seq in [5,6]
kind=sys.argv[1];assert kind in ['updated','historical']
reference=DEST/'reference'/f'new_archaeo_colmap_traj_sequence_{seq:02d}.txt' if kind=='updated' else (DEST/'reference/colmap_traj_sequence_5.txt' if seq==5 else Path(r'C:\Users\Raaghav\Downloads\archaeo_sequence_6_raw_data/colmap_traj_sequence_6.txt'))
os.environ['AQUALOC_REFERENCE']=str(reference)
from dense_reference import *
from dense_evaluate import eval_group,boot
from motion_controls_v3 import oracle_replays,fit_predict_reference
import pandas as pd,copy

def main():
 target=DEST/(f'evaluation_{kind}' if seq==5 else f'sequence6_{kind}');target.mkdir(exist_ok=True);folder=DEST/'vo' if seq==5 else OUT/'vo';mods={p.stem:model(p) for p in folder.glob('*.csv') if p.with_suffix('.done.json').exists()};assert len(mods)>=20
 high=[mods[f'{a}_{p}_rho0.06_quality_depth1e+09'] for a in ['ORB','AKAZE','SIFT'] for p in ['raw','fixed','adaptive']];low=[mods[f'{a}_{p}_rho0.06_quality_depth50'] for a in ['ORB','AKAZE','SIFT'] for p in ['raw','fixed','adaptive']]
 groups=[('high_matrix_common',high),('low_matrix_common',low)]+[(f'source_depth{c}',[mods[f'ORB_fixed_rho0.06_{policy}_depth{c}'] for policy in ['quality','descriptor_first']]) for c in ['50','1e+09']]
 seg=[];err=[];coverage=[]
 for group,mm in groups:
  for clean in [False,True]:
   s,e,c=eval_group(mm,group,clean=clean,fit_lengths=(800,),safe_reference=True);seg+=s;err+=e;coverage+=c
 # Native-only sensitivity has the same deliberately sparse thresholds as Seq 06.
 native=[dict(m,fresh=m['fresh']&np.isin(FRAMES,IDS)) for m in high]
 for clean in [False,True]:
  s,e,c=eval_group(native,'native_only_common',clean=clean,fit_lengths=(800,),safe_reference=True,min_fit=6,min_test=1);seg+=s;err+=e;coverage+=c
 writecsv(target/'segments.csv',seg);writecsv(target/'per_pose.csv',err);writecsv(target/'coverage.csv',coverage);writecsv(target/'paired_ranges.csv',boot(seg,replicates=2000))
 controls=[];key='ORB_fixed_rho0.06_quality_depth1e+09';oracles,undefined=oracle_replays(mods[key])
 for r in [r for r in seg if r['group'] in ['high_matrix_common','native_only_common'] and r['configuration']==key]:
  train=np.array(list(map(int,r['train_ids'].split(';'))));test=np.array(list(map(int,r['test_ids'].split(';'))))
  for name in ['reference_hold','reference_cv_all40s','reference_cv_last5s',*oracles]:
   if name in oracles:s,R,t=fit(oracles[name][train],C[train]);prediction=s*(oracles[name][test]@R.T)+t
   else:prediction=fit_predict_reference(train,test,name)
   e=np.linalg.norm(prediction-C[test],axis=1);controls.append(dict(group=r['group'],mask=r['mask'],configuration=name,test_boundary=r['test_boundary'],fit_n=len(train),test_n=len(test),sse=float(e@e),rmse=float(np.sqrt(np.mean(e*e))),train_ids=r['train_ids'],test_ids=r['test_ids']))
 writecsv(target/'control_segments.csv',controls);summary=[]
 for (group,mask,name),d in pd.DataFrame(seg+controls).groupby(['group','mask','configuration']):summary.append(dict(group=group,mask=mask,configuration=name,segments=len(d),test_positions=int(d.test_n.sum()),rmse_m=float(np.sqrt(d.sse.sum()/d.test_n.sum()))))
 writecsv(target/'summary.csv',summary)
 directions=[]
 used={m['name']:m for _,group_models in groups for m in group_models}
 for name,m in used.items():
  for frame in np.flatnonzero(m['fresh']):
   a=int(m['rows'][frame]['attempted_anchor'])
   if a<IDS[0]:continue
   refR=ROTS[frame].T@ROTS[a];estR=m['R'][frame].T@m['R'][a];tref=ROTS[frame].T@(C[a]-C[frame]);test=m['R'][frame].T@(m['p'][a]-m['p'][frame])
   directions.append(dict(configuration=name,frame=int(frame),anchor=a,gap=int(frame-a),source=m['rows'][frame]['source'],rotation_error_deg=degrees(refR.T@estR),direction_error_deg=angle(tref,test),reference_step_m=float(np.linalg.norm(tref)),clean_endpoints=int(CLEAN[a] and CLEAN[frame])))
 writecsv(target/'actual_transition_directions.csv',directions)
 angular=[]
 for (name,clean),d in pd.DataFrame(directions).groupby(['configuration','clean_endpoints']):
  angular.append(dict(configuration=name,clean_endpoints=int(clean),transitions=len(d),median_direction_deg=float(d.direction_error_deg.median()),median_rotation_deg=float(d.rotation_error_deg.median()),fraction_below_90=float(np.mean(d.direction_error_deg<90)),fraction_flow=float(np.mean(d.source=='flow'))))
 writecsv(target/'actual_transition_direction_summary.csv',angular)
 checks=[]
 for r in seg+controls:
  train=list(map(int,r['train_ids'].split(';')));test=list(map(int,r['test_ids'].split(';')));assert min(train)>=IDS[0] and max(train)<=IDS[IDS<r['test_boundary']][-1] and max(train)<min(test)
  checks.append(dict(group=r['group'],mask=r['mask'],configuration=r['configuration'],test_boundary=r['test_boundary'],passed=True))
 for (group,mask,bound),d in pd.DataFrame(seg).groupby(['group','mask','test_boundary']):assert d.train_ids.nunique()==d.test_ids.nunique()==1
 writecsv(target/'validation.csv',checks);dump(target/'reference.json',dict(sequence=seq,source=str(REFERENCE),sha256=sha(REFERENCE),native_poses=len(IDS),first=int(IDS[0]),last=int(IDS[-1]),path_m=PATH,diagonal_m=DIAG,robust_step_threshold_m=THRESH,flagged_edges=int(sum(BAD)),undefined_oracle_prefix_headings=undefined,scope=('held-out Sequence 05' if seq==5 else 'development Sequence 06')+'; same archaeological site; reconstruction consistency against supplied reference, not complete odometry accuracy',validation_checks=len(checks)))
 print('Evaluated Sequence',seq,kind,':',len(seg),'segments;',len(err),'residual rows',flush=True)

if __name__=='__main__':main()
