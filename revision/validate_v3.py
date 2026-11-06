from dense_reference import *
import pandas as pd,re
R3=ROOT/'revision3_results';PAPER=ROOT/'IEEE_AQUALOC_Revision3'

def main():
 checks=[]
 def check(name,condition):
  assert bool(condition),name;checks.append(dict(check=name,passed=True))
 seg=pd.read_csv(R3/'motion_control_segments.csv');errors=pd.read_csv(R3/'motion_control_per_pose.csv');keys=['group','mask','configuration','test_boundary'];observed=errors.groupby(keys).error.agg(n='size',sse=lambda e:np.sum(e.to_numpy()**2));joined=seg.set_index(keys).join(observed,rsuffix='_residual')
 check('every current residual group exists and is unique',len(observed)==len(seg) and len(seg)==len(seg.drop_duplicates(keys)))
 check('all segment SSE match exported residuals',np.allclose(joined.sse,joined.sse_residual,rtol=1e-10,atol=1e-12))
 check('all segment counts match exported residuals',joined.test_n.eq(joined.n).all())
 for (group,mask,bound),d in seg.groupby(['group','mask','test_boundary']):
  check(f'paired identities {group}/{mask}/{bound}',d.train_ids.nunique()==1 and d.test_ids.nunique()==1)
  train=np.array(list(map(int,d.train_ids.iloc[0].split(';'))));test=np.array(list(map(int,d.test_ids.iloc[0].split(';'))));check(f'strict reference brackets {group}/{mask}/{bound}',len(train)>=20 and max(train)<=IDS[IDS<bound][-1] and max(train)<min(test))
  if mask!='all_reference':check(f'exclusion {group}/{bound}',CLEAN[train].all() and CLEAN[test].all())
 summary=pd.read_csv(R3/'motion_control_summary.csv')
 for r in summary.itertuples():
  d=seg[(seg.group==r.group)&(seg['mask']==r.mask)&(seg.configuration==r.configuration)];check('pooled summary '+r.group+'/'+r.mask+'/'+r.configuration,len(d)==r.segments and int(d.test_n.sum())==r.test_positions and abs(np.sqrt(d.sse.sum()/d.test_n.sum())-r.rmse_m)<1e-12)
 native=pd.read_csv(R3/'native_only_segments.csv')
 for r in native.itertuples():
  train=list(map(int,r.train_ids.split(';')));test=list(map(int,r.test_ids.split(';')));check('native-only '+r.mask+'/'+r.configuration+'/'+str(r.test_boundary),all(i%20==0 for i in train+test) and len(train)>=6 and max(train)<r.test_boundary<=min(test))
 for path in ['cross_detector_paired_ranges.csv','matrix_all_paired_ranges.csv','source_policy_paired_ranges.csv','baseline_paired_ranges.csv']:
  for r in pd.read_csv(R3/path).itertuples():
   d=seg[(seg.group==r.group)&(seg['mask']==r.mask)];a=d[d.configuration==r.a];b=d[d.configuration==r.b];effect=np.sqrt(b.sse.sum()/b.test_n.sum())-np.sqrt(a.sse.sum()/a.test_n.sum());check('paired effect '+path+'/'+r.mask+'/'+r.a+'/'+r.b+'/'+str(r.block_segments),abs(effect-r.rmse_b_minus_a)<1e-12 and r.low<=r.high and r.replicates==2000)
 key='ORB_fixed_rho0.06_quality_depth1e+09';allr=summary[(summary.group=='high_matrix_common')&(summary['mask']=='all_reference')].set_index('configuration');names=[f'{a}_{p}_rho0.06_quality_depth1e+09' for a in ['ORB','AKAZE','SIFT'] for p in ['raw','fixed','adaptive']]
 check('recent reference velocity beats all nine reconstructions on stated metric',allr.loc['reference_cv_last5s'].rmse_m<allr.loc[names].rmse_m.min())
 check('reference heading with gap steps retains material error',allr.loc[key+'__reference_heading_gap_step'].rmse_m>.4)
 check('combined high-cutoff reference oracle identity',allr.loc[[n+'__reference_heading_and_step' for n in names]].rmse_m.max()<1e-10)
 seeds=pd.read_csv(R3/'seed_causal_segments.csv')
 for bound,d in seeds.groupby('test_boundary'):
  check('all twenty sampling trials share fit/test identities '+str(bound),len(d)==20 and d.train_ids.nunique()==1 and d.test_ids.nunique()==1 and max(map(int,d.train_ids.iloc[0].split(';')))<=IDS[IDS<bound][-1])
 check('sampling common tests retained',pd.read_csv(R3/'seed_causal_summary.csv').common_test_positions.eq(675).all())
 check('sampling source effect all five ordered/permuted trials',pd.read_csv(R3/'seed_causal_differences.csv').difference_D_minus_Q_m.gt(0).all())
 d=pd.read_csv(R3/'cross_detector_paired_ranges.csv');a=d[(d['mask']=='all_reference')&(d.block_segments==2)&d.a.str.startswith('AKAZE')&d.b.str.startswith('SIFT')].iloc[0];check('AKAZE SIFT ordering unresolved',a.low<0<a.high)
 # Training-only controls must be insensitive to arbitrary changes after the
 # test boundary. Exercise the actual predictor rather than only inspecting IDs.
 import motion_controls_v3 as controls
 r=next(seg[seg.configuration=='reference_cv_last5s'].itertuples());train=np.array(list(map(int,r.train_ids.split(';'))));test=np.array(list(map(int,r.test_ids.split(';'))));saved=controls.C
 try:
  for kind in ['reference_hold','reference_cv_all40s','reference_cv_last5s']:
   controls.C=saved.copy();before=controls.fit_predict_reference(train,test,kind);controls.C[int(r.test_boundary):]+=np.array([1000.,-2000.,3000.]);after=controls.fit_predict_reference(train,test,kind);check('no future reference input '+kind,np.array_equal(before,after))
 finally:controls.C=saved
 body=(PAPER/'manuscript.md').read_text(encoding='utf-8').split('## REFERENCES')[0];check('Q definitions precede score',body.index('ne is the number')<body.index('$$ Q =') and 'measured in degrees' in body and 'g the current image ID' in body)
 seen=[]
 for num in re.findall(r'\[(\d+)\]',body):
  if int(num) not in seen:seen.append(int(num))
 check('citation order retained',seen==list(range(1,18)))
 check('primary control and scope present',f"{allr.loc['reference_cv_last5s'].rmse_m:.3f} m" in body and 'not a complete odometry' in body and 'have not been completed' in body)
 check('no stale dense headline',all(x not in body for x in ['0.331 m','0.398–0.671 m','0.430 m Q-versus-D']))
 writecsv(R3/'validation_v3.csv',checks);dump(R3/'validation_v3_summary.json',dict(passed=True,checks=len(checks),current_control_segments=len(seg),current_control_residuals=len(errors),native_segments=len(native),sampling_segments=len(seeds),remaining=['independent sequence: dataset download pending','established ORB-SLAM3/UVS-class baseline not completed'],scope='numerical and implementation validation, not independent physical ground truth'))
 print('PASS',len(checks),'current validation checks',flush=True)

if __name__=='__main__':main()
