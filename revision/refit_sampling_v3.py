"""Refit saved order trials without reference interpolation across test boundaries."""
from dense_reference import *
import pandas as pd,re
R3=ROOT/'revision3_results';R3.mkdir(exist_ok=True)

def main():
 d=pd.read_csv(R2/'seed_dense_segments.csv');d=d[(d.group=='five_trial_common')&(d.fit_frames==800)];mods={};out=[]
 for name,dd in d.groupby('configuration'):
  file=OUT/'vo'/f'{name}.csv' if '_seed' not in name else R2/'seed_order_runs/vo'/f'{name}.csv';m=model(file);mods[name]=m
  for r in dd.itertuples():
   train=np.array(list(map(int,r.train_ids.split(';'))));train=train[train<=IDS[IDS<r.test_boundary][-1]];test=np.array(list(map(int,r.test_ids.split(';'))));assert len(train)>=20
   s,R,t=fit(m['p'][train],C[train]);e=np.linalg.norm(s*(m['p'][test]@R.T)+t-C[test],axis=1);out.append(dict(configuration=name,test_boundary=r.test_boundary,fit_n=len(train),test_n=len(test),sse=float(e@e),rmse=float(np.sqrt(np.mean(e*e))),train_ids=';'.join(map(str,train)),test_ids=r.test_ids))
 writecsv(R3/'seed_causal_segments.csv',out)
 rows=[]
 old=pd.read_csv(R2/'sampling_trial_summary.csv').set_index('configuration')
 for name,dd in pd.DataFrame(out).groupby('configuration'):
  r=old.loc[name];rows.append(dict(seed=int(r.seed),order=r.order,configuration=name,preprocessing=r.preprocessing,policy=r.policy,accepted=int(r.accepted),acceptance_percent=float(r.acceptance_percent),common_segments=len(dd),common_test_positions=int(dd.test_n.sum()),common_rmse_m=float(np.sqrt(dd.sse.sum()/dd.test_n.sum()))))
 writecsv(R3/'seed_causal_summary.csv',rows)
 diff=[];d=pd.DataFrame(rows)
 for seed,dd in d.groupby('seed'):
  q=dd[(dd.preprocessing=='fixed')&(dd.policy=='quality')].iloc[0];desc=dd[(dd.preprocessing=='fixed')&(dd.policy=='descriptor_first')].iloc[0];diff.append(dict(seed=int(seed),quality_rmse_m=float(q.common_rmse_m),descriptor_rmse_m=float(desc.common_rmse_m),difference_D_minus_Q_m=float(desc.common_rmse_m-q.common_rmse_m),common_test_positions=int(q.common_test_positions)))
 writecsv(R3/'seed_causal_differences.csv',diff)
 print('Causal refits complete:',len(rows),'trials;',len(out),'segments',flush=True)

if __name__=='__main__':main()
