"""Apply the existing causal evaluator to all five frozen ORB order trials."""
import os
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
os.environ['AQUALOC_REFERENCE']=str(ROOT/'sequence5_results/reference/new_archaeo_colmap_traj_sequence_05.txt')
from dense_evaluate import *
import pandas as pd

def main():
 dest=ROOT/'sequence5_results/order_runs';markers=list((dest/'vo').glob('*.done.json'));assert len(markers)==16
 mods=[];checks=[];metadata=[]
 def check(label,ok):assert bool(ok),label;checks.append(dict(check=label,passed=True))
 protocol=json.loads((dest/'protocol.json').read_text());original=json.loads((ROOT/'sequence5_results/protocol.json').read_text())
 check('unchanged numerical estimator source',all(protocol['source_sha256'][p]==sha(ROOT/'revision'/p)==original['source_sha256'][p] for p in ['run_vo.py','common.py']))
 for seed in [20261105,20261106,20261107,20261108,20261109]:
  suffix='' if seed==20261105 else f'_seed{seed}';folder=ROOT/'sequence5_results' if seed==20261105 else dest
  for prep,policy in [('raw','quality'),('fixed','quality'),('adaptive','quality'),('fixed','descriptor_first')]:
   name=f'ORB_{prep}_rho0.06_{policy}_depth1e+09'+suffix;p=folder/'vo'/(name+'.csv');meta=json.loads(p.with_suffix('.done.json').read_text());d=pd.read_csv(p);m=model(p);mods.append(m)
   check('complete frozen run '+name,meta['frames']==len(d)==3983 and np.array_equal(d.frame,np.arange(3983)) and meta['recovery_overload']=='explicit_distanceThresh_keyword' and int(d.fresh.sum())==meta['accepted'])
   check('rigid poses '+name,np.isfinite(m['p']).all() and np.isfinite(m['R']).all() and np.max(abs(m['R'].transpose(0,2,1)@m['R']-np.eye(3)))<1e-7 and np.allclose(np.linalg.det(m['R']),1,atol=1e-7))
   accepted=d[d.fresh==1];check('gates '+name,accepted['ne'].ge(15).all() and accepted.nc.ge(10).all() and accepted.rho.ge(.06).all() and accepted.parallax_deg.ge(.1).all() and d.features.le(1500).all())
   metadata.append(dict(configuration=name,seed=seed,preprocessing=prep,policy=policy,accepted=meta['accepted'],acceptance_percent=100*meta['accepted']/3982,order='ordered' if seed==20261105 else 'permuted'))
  q,z=mods[-3],mods[-1];check('same source-policy anchors '+str(seed),all(a[k]==b[k] for a,b in zip(q['rows'],z['rows']) for k in ['fresh','attempted_anchor','reset']))
 groups=[('five_trial_common',mods)]+[(f'paired_trial_{seed}',mods[i*4+1:i*4+2]+mods[i*4+3:i*4+4]) for i,seed in enumerate([20261105,20261106,20261107,20261108,20261109])]
 segments=[];errors=[];coverage=[]
 for group,mm in groups:
  for clean in [False,True]:
   s,e,c=eval_group(mm,group,clean=clean,fit_lengths=(800,),safe_reference=True);segments+=s;errors+=e;coverage+=c
 data=pd.DataFrame(segments);assert len(data)
 summary=[];info={r['configuration']:r for r in metadata}
 for (group,mask,name),d in data.groupby(['group','mask','configuration']):summary.append(dict(group=group,mask=mask,**info[name],windows=len(d),test_positions=int(d.test_n.sum()),rmse_m=float(np.sqrt(d.sse.sum()/d.test_n.sum()))))
 for (group,mask,bound),d in data.groupby(['group','mask','test_boundary']):check('paired safe IDs '+group+'/'+mask+'/'+str(bound),d.train_ids.nunique()==d.test_ids.nunique()==1 and max(map(int,d.train_ids.iloc[0].split(';')))<=IDS[IDS<bound][-1] and bound-800>=IDS[0])
 sums=pd.DataFrame(errors).groupby(['group','mask','configuration','segment']).error.agg(n='size',sse=lambda e:float(e@e));joined=data.set_index(['group','mask','configuration','segment']).join(sums,rsuffix='_error');check('per-pose counts and SSE',np.array_equal(joined.test_n,joined.n) and np.allclose(joined.sse,joined.sse_error,rtol=1e-10,atol=1e-10))
 diffs=[];summary=pd.DataFrame(summary)
 for group in summary.group.unique():
  for seed in [20261105,20261106,20261107,20261108,20261109]:
   d=summary[(summary.group==group)&(summary.seed==seed)&(summary['mask']=='all_reference')&(summary.preprocessing=='fixed')]
   if len(d)!=2:continue
   q=d[d.policy=='quality'].iloc[0];z=d[d.policy=='descriptor_first'].iloc[0];diffs.append(dict(group=group,seed=seed,test_positions=int(q.test_positions),windows=int(q.windows),Q_rmse_m=q.rmse_m,D_rmse_m=z.rmse_m,D_minus_Q_m=z.rmse_m-q.rmse_m))
 writecsv(dest/'segments.csv',segments);writecsv(dest/'per_pose.csv',errors);writecsv(dest/'coverage.csv',coverage);writecsv(dest/'summary.csv',summary.to_dict('records'));writecsv(dest/'source_differences.csv',diffs);writecsv(dest/'paired_ranges.csv',boot([r for r in segments if r['group']!='five_trial_common'],replicates=2000));writecsv(dest/'validation.csv',checks)
 dump(dest/'validation_summary.json',dict(passed=True,checks=len(checks),new_complete_runs=16,trials=5,frames=3983,reference=str(REFERENCE),scope='within-sequence input-order sensitivity; not five independent sequence replications'))
 print('PASS',len(checks),'checks;',len(segments),'segments;',len(errors),'per-pose errors',flush=True)

if __name__=='__main__':main()
