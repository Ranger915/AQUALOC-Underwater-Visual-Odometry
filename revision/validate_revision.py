from common import *
import pandas as pd
checks=[]
def check(name,condition,detail=''):
 checks.append(dict(check=name,passed=bool(condition),detail=detail))
 if not condition:raise AssertionError((name,detail))
metas=list((OUT/'vo').glob('*.done.json'));check('32 directly executed complete runs',len(metas)==32,str(len(metas)))
for p in metas:
 m=json.loads(p.read_text());d=pd.read_csv(p.with_name(p.name.replace('.done.json','.csv')))
 check(p.stem+' frame sequence',len(d)==3385 and np.array_equal(d.frame,np.arange(3385)))
 check(p.stem+' finite integrated poses',np.isfinite(d[['p0','p1','p2','R00','R11','R22']]).all().all())
 if 'depth1e+09' in p.name:check(p.stem+' explicit overload',m.get('recovery_overload')=='explicit_distanceThresh_keyword')
seg=pd.read_csv(OUT/'rolling_segment_metrics.csv');err=pd.read_csv(OUT/'rolling_per_pose_errors.csv')
check('six test positions per segment',seg.test_n.eq(6).all())
check('no fit/test chronology leakage',(seg.train_last<seg.test_first).all())
check('per-pose rows correspond to segment counts',len(err)==int(seg.test_n.sum()))
for (group,k,segment),a in seg.groupby(['group','k','segment']):
 if group.startswith('per_configuration'):continue
 check('paired IDs '+group+str(k)+str(segment),a.test_ids.nunique()==1 and a.train_ids.nunique()==1)
for alg in ['ORB','AKAZE','SIFT']:
 a=pd.read_csv(OUT/'feature_counts'/f'{alg}_c2t8.csv');b=pd.read_csv(OUT/'feature_counts'/f'{alg}_raw_budget.csv');check(alg+' exact per-image raw feature budget',np.array_equal(a.features,b.features))
 sw=pd.read_csv(OUT/'switch_threshold_random_controls.csv');sw=sw[sw.detector==alg];x=sw[sw.type=='random_matched_keyframes'];check(alg+' random budgets',len(x)==24 and np.allclose(x.fraction_raw_all,808/3385) and np.allclose(x.fraction_raw_keyframes,41/170))
 check(alg+' all 12 fixed settings',sum((OUT/'frontend'/f'{alg}_c{c}t{t}_c{c}t{t}.csv').exists() for c in [1,2,3,4] for t in [4,8,16])==12)
z=json.loads((OUT/'depth_overload_validation.json').read_text());check('synthetic far-depth intervention',z['passed'])
z=json.loads((OUT/'decision_agreement_summary.json').read_text());check('all-frame vectorized decision agreement',z['n']==3385 and z['reshape_disagreements']==z['integral_disagreements']==0)
R=np.array([[0.,-1,0],[1,0,0],[0,0,1]]);rng=np.random.default_rng(3);p=rng.normal(size=(40,3));q=2.7*(p@R.T)+np.array([1,2,3]);s,r,t=fit(p,q);check('independent known Sim3 recovery',np.max(np.abs(s*(p@r.T)+t-q))<1e-10)
writecsv(OUT/'revision_validation.csv',checks);dump(OUT/'revision_validation_summary.json',dict(passed=all(r['passed'] for r in checks),checks=len(checks),executed_full_runs=len(metas),rolling_segments=len(seg),per_pose_rows=len(err),saved_adaptive_reproduction='ORB and AKAZE exact; SIFT one count on one pair differs'))
print('PASS',len(checks),'checks',flush=True)
