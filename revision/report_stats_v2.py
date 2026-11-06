from dense_reference import *
import pandas as pd
def main():
 seg=pd.read_csv(R2/'seed_dense_segments.csv');rr=[];trial=[]
 for seed in [20261105,20261106,20261107,20261108,20261109]:
  suf='' if seed==20261105 else f'_seed{seed}'
  for prep,policy in [('raw','quality'),('fixed','quality'),('adaptive','quality'),('fixed','descriptor_first')]:
   name=f'ORB_{prep}_rho0.06_{policy}_depth1e+09'+suf;path=(OUT/'vo' if seed==20261105 else R2/'seed_order_runs/vo')/(name+'.csv');a=pd.read_csv(path);ss=seg[(seg.group=='five_trial_common')&(seg.configuration==name)];rmse=float(np.sqrt(ss.sse.sum()/ss.test_n.sum())) if len(ss) else np.nan
   rr.append(dict(seed=seed,order='ordered baseline' if seed==20261105 else 'seeded joint correspondence permutation',configuration=name,preprocessing=prep,policy=policy,accepted=int(a.fresh.sum()),acceptance_percent=100*float(a.fresh.sum())/3384,common_segments=len(ss),common_test_positions=int(ss.test_n.sum()),common_rmse_m=rmse,percent_path=100*rmse/PATH,percent_diagonal=100*rmse/DIAG))
  q=next(r for r in rr if r['seed']==seed and r['preprocessing']=='fixed' and r['policy']=='quality');d=next(r for r in rr if r['seed']==seed and r['policy']=='descriptor_first');trial.append(dict(seed=seed,quality_rmse_m=q['common_rmse_m'],descriptor_rmse_m=d['common_rmse_m'],difference_D_minus_Q_m=d['common_rmse_m']-q['common_rmse_m'],common_test_positions=q['common_test_positions'],segments=q['common_segments']))
 writecsv(R2/'sampling_trial_summary.csv',rr);writecsv(R2/'sampling_policy_differences.csv',trial)
 aggregate=[];df=pd.DataFrame(rr)
 for (prep,policy),a in df.groupby(['preprocessing','policy']):aggregate.append(dict(preprocessing=prep,policy=policy,trials=len(a),acceptance_min=float(a.acceptance_percent.min()),acceptance_max=float(a.acceptance_percent.max()),rmse_min=float(a.common_rmse_m.min()),rmse_median=float(a.common_rmse_m.median()),rmse_max=float(a.common_rmse_m.max()),common_test_positions=int(a.common_test_positions.iloc[0]),common_segments=int(a.common_segments.iloc[0])))
 writecsv(R2/'sampling_trial_ranges.csv',aggregate)
 summary=pd.read_csv(R2/'dense_summary.csv');ranges=pd.read_csv(R2/'dense_paired_ranges.csv')
 def get(group,name,mask='all_reference',fit=800):return summary[(summary.group==group)&(summary.configuration==name)&(summary['mask']==mask)&(summary.fit_frames==fit)].iloc[0].to_dict()
 def delta(group,a,b,mask='all_reference',fit=800):
  z=ranges[(ranges.group==group)&(ranges['mask']==mask)&(ranges.fit_frames==fit)&(ranges.block_segments==2)];x=z[(z.a==a)&(z.b==b)];sign=1
  if not len(x):x=z[(z.a==b)&(z.b==a)];sign=-1
  if not len(x):return dict(effect=np.nan,low=np.nan,high=np.nan)
  r=x.iloc[0];return dict(effect=sign*float(r.rmse_b_minus_a),low=float(r.low) if sign==1 else -float(r.high),high=float(r.high) if sign==1 else -float(r.low))
 matrix=[]
 for alg in ['ORB','AKAZE','SIFT']:
  base=f'{alg}_fixed_rho0.06_quality_depth1e+09'
  for prep in ['raw','fixed','adaptive']:
   name=f'{alg}_{prep}_rho0.06_quality_depth1e+09';a=get('high_matrix_common',name);clean=get('high_matrix_common',name,'robust_reference_edges_excluded');a.update(detector=alg,preprocessing=prep,clean_rmse_m=clean['rmse_m'],clean_test_positions=clean['test_positions']);a.update(delta('high_matrix_common',base,name) if prep!='fixed' else dict(effect=0.,low=0.,high=0.));matrix.append(a)
 policy=[]
 for dep in ['50','1e+09']:
  group='source_policies_depth'+('50' if dep=='50' else '1e9');q=f'ORB_fixed_rho0.06_quality_depth{dep}';d=f'ORB_fixed_rho0.06_descriptor_first_depth{dep}'
  for ft in [400,800,1200]:
   for mask in ['all_reference','robust_reference_edges_excluded']:
    a=get(group,q,mask,ft);b=get(group,d,mask,ft);r=dict(depth=dep,fit_seconds=ft/20,mask=mask,test_positions=a['test_positions'],segments=a['segments'],quality_rmse=a['rmse_m'],descriptor_rmse=b['rmse_m'],quality_percent_diagonal=a['percent_diagonal'],quality_percent_path=a['percent_path']);r.update(delta(group,q,d,mask,ft));policy.append(r)
 writecsv(R2/'paper_matrix_v2.csv',matrix);writecsv(R2/'paper_policy_v2.csv',policy);print(json.dumps(dict(trial_ranges=aggregate,trial_policy_differences=trial,primary_policy=[r for r in policy if r['fit_seconds']==40]),indent=2),flush=True)
if __name__=='__main__':main()
