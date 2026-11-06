from dense_reference import *
import pandas as pd,re
checks=[]
def check(name,condition,detail=''):
 checks.append(dict(check=name,passed=bool(condition),detail=detail))
 if not condition:raise AssertionError((name,detail))
check('interpolation reproduces supplied positions',np.allclose(C[IDS],CENTERS,atol=1e-12))
check('interpolation reproduces supplied rotations',all(np.allclose(ROTS[i],REF[int(i)][0],atol=1e-12) for i in IDS))
check('SLERP returns proper rotations',np.allclose(np.linalg.det(ROTS),1,atol=1e-10) and np.allclose(ROTS@ROTS.transpose(0,2,1),np.eye(3),atol=1e-10))
check('no extrapolated frames',FRAMES[-1]==3380)
for stem in ['dense','seed_dense']:
 seg=pd.read_csv(R2/(stem+'_segments.csv'));e=pd.read_csv(R2/(stem+'_per_pose.csv'))
 key=['group','mask','fit_frames','configuration','segment'];by=e.groupby(key).agg(n=('error','size'),sse=('error',lambda x:float(x@x))).reset_index();m=seg.merge(by,on=key)
 check(stem+' every segment has residuals',len(m)==len(seg))
 check(stem+' residual count and SSE',np.array_equal(m.test_n,m.n) and np.allclose(m.sse_x,m.sse_y,atol=1e-8))
 check(stem+' exported residual norm',np.allclose(e.error,np.linalg.norm(e[['residual_x','residual_y','residual_z']].to_numpy(),axis=1),atol=1e-10))
 for (group,mask,fitframes,segment),a in seg.groupby(['group','mask','fit_frames','segment']):
  check(stem+' paired fit/test IDs '+group+str(fitframes)+str(segment),a.train_ids.nunique()==1 and a.test_ids.nunique()==1)
  train=np.array(list(map(int,a.train_ids.iloc[0].split(';'))));test=np.array(list(map(int,a.test_ids.iloc[0].split(';'))));boundary=int(a.test_boundary.iloc[0]);check(stem+' chronological bounds '+group+str(fitframes)+str(segment),np.all((train>=boundary-fitframes)&(train<boundary)) and np.all((test>=boundary)&(test<=a.test_end.iloc[0])) and np.all(test<=3380))
  if mask!='all_reference':check(stem+' robust exclusion '+group+str(fitframes)+str(segment),np.all(CLEAN[train]) and np.all(CLEAN[test]))
 # Error vector rows are unique within each comparison/window.
 check(stem+' no duplicated frame residuals',not e.duplicated(key+['frame']).any())
for alg in ['ORB','AKAZE','SIFT']:
 d=pd.read_csv(R2/f'switch_null_draws_{alg}.csv');summary=pd.read_csv(R2/'switch_null_summary.csv');summary=summary[summary.detector==alg]
 check(alg+' null counts',sum(d['null']=='uniform_frame_keyframe_budgets')==10000 and sum(d['null']=='uniform_frame_budget')==10000 and sum(d['null']=='keyframe_circular_shift')==170 and sum(d['null']=='full_frame_circular_shift')==3385)
 check(alg+' exact keyframe budgets',d[d['null'].isin(['uniform_frame_keyframe_budgets','keyframe_circular_shift'])].raw_keyframes.eq(41).all())
 check(alg+' preserved circular runs',d[d['null']=='keyframe_circular_shift'].keyframe_circular_raw_runs.eq(1).all())
 obs=float(d[d['null']=='observed_contrast'].ap.iloc[0]);check(alg+' identity shift',abs(float(d[(d['null']=='keyframe_circular_shift')&(d.draw==0)].ap.iloc[0])-obs)<1e-12)
 for kind in ['uniform_frame_keyframe_budgets','uniform_frame_budget','keyframe_circular_shift','full_frame_circular_shift']:
  a=d[d['null']==kind];b=summary[summary['null']==kind].iloc[0];tail=int(sum(a.ap>=obs-1e-12));expected=tail/len(a) if 'shift' in kind else (tail+1)/(len(a)+1);check(alg+' correct tail '+kind,tail==b.tail_ge_observed and abs(expected-b.tail_fraction)<1e-12)
metas=list((R2/'seed_order_runs/vo').glob('*.done.json'));check('16 completed sampling-order full runs',len(metas)==16)
for p in metas:
 meta=json.loads(p.read_text());q=p.with_name(p.name.replace('.done.json','.csv'));a=pd.read_csv(q);check(p.stem+' full frame coverage',len(a)==3385 and np.array_equal(a.frame,np.arange(3385)));check(p.stem+' valid intervention',meta['sampling']=='seeded_correspondence_permutation' and meta['recovery_overload']=='explicit_distanceThresh_keyword');check(p.stem+' finite integrated poses',np.isfinite(a[['p0','p1','p2','R00','R11','R22']]).all().all())
for seed in [20261106,20261107,20261108,20261109]:
 folder=R2/'seed_order_runs/vo';q=pd.read_csv(folder/f'ORB_fixed_rho0.06_quality_depth1e+09_seed{seed}.csv');d=pd.read_csv(folder/f'ORB_fixed_rho0.06_descriptor_first_depth1e+09_seed{seed}.csv');check('fixed policy same acceptance '+str(seed),np.array_equal(q.fresh,d.fresh))
d=pd.read_csv(R2/'short_gap_cutoff_probe.csv');check('cutoff scope',len(d)==2535 and d.frame.nunique()==169);check('one selected row per pair/cutoff',len(d[d.source=='selected'])==169*5);check('cutoff anchors held fixed',d.groupby('frame').anchor.nunique().eq(1).all());meta=json.loads((R2/'same_anchor_cutoff_probe_validation.json').read_text());check('probe reproduces high-depth accepted/source choices',meta['baseline_acceptance_or_source_mismatches']==0)
meta=json.loads((R2/'boundary_transition_summary.json').read_text());check('transition telescoping identity',meta['boundary_accepted_steps']==72 and meta['sum_increment_check_max_abs']<1e-10)
paper=ROOT/'IEEE_AQUALOC_Revision2';body=(paper/'manuscript.md').read_text(encoding='utf-8').split('## REFERENCES')[0];seen=[]
for x in re.findall(r'\[(\d+)\]',body):
 if int(x) not in seen:seen.append(int(x))
check('references ordered by first citation',seen==list(range(1,18)),str(seen));check('unsupported pooled/preregistration language absent','72/72' not in body and 'before its new outcome' not in body);check('clean matrix test count correct','exclusion evaluates 683 positions' in body);check('intermediate cutoff count matches table','144 at 10⁴, and 143 at 10⁹' in body)
writecsv(R2/'validation_v2.csv',checks);dump(R2/'validation_v2_summary.json',dict(passed=True,checks=len(checks),new_full_runs=16,null_draws_per_iid_family_per_detector=10000,dense_residual_rows=233728,sampling_common_test_positions=675,cutoff_probe_pairs=169,scope='numerical consistency and implementation checks; not independent ground-truth validation'))
print('PASS',len(checks),'checks',flush=True)
