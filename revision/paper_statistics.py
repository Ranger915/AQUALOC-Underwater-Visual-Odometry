from common import *
import pandas as pd
def aggregate():
 vo=pd.read_csv(OUT/'vo_configuration_summary.csv');seg=pd.read_csv(OUT/'rolling_segment_metrics.csv');cov=pd.read_csv(OUT/'rolling_evaluation_coverage.csv');front=pd.read_csv(OUT/'frontend_primary_summary.csv');switch=pd.read_csv(OUT/'switch_threshold_random_controls.csv');direct=pd.read_csv(OUT/'direct_two_view_motion.csv')
 matrix=[]
 for alg in ['ORB','AKAZE','SIFT']:
  for prep,mode in [('raw','raw_raw'),('fixed','c2t8_c2t8'),('adaptive','adaptive_observed')]:
   ap=float(front[(front.detector==alg)&(front['mode']==mode)].iloc[0].ap_count)
   r=dict(detector=alg,preprocessing=prep,ap=ap)
   for dep in ['50','1e+09']:
    name=f'{alg}_{prep}_rho0.06_quality_depth{dep}';a=vo[vo.configuration==name]
    if len(a):r.update({f'accept_{dep}':float(a.iloc[0].acceptance),f'fresh_{dep}':int(a.iloc[0].fresh_reference)})
   highname=f'{alg}_{prep}_rho0.06_quality_depth1e+09';ss=seg[(seg.group=='high_depth_matrix_common')&(seg.k==24)&(seg.configuration==highname)]
   r['common_high_rmse']=float(np.sqrt(np.mean(ss.rmse**2))) if len(ss) else float('nan');r['common_high_tests']=int(ss.test_n.sum()) if len(ss) else 0;matrix.append(r)
 comparisons=[]
 for k in [12,24,36]:
  group='paired_with_ORB_fixed__ORB_fixed_rho0.06_descriptor_first_depth50';rr=[]
  for name in ['ORB_fixed_rho0.06_quality_depth50','ORB_fixed_rho0.06_descriptor_first_depth50']:
   a=seg[(seg.group==group)&(seg.k==k)&(seg.configuration==name)];rr.append(float(np.sqrt(np.mean(a.rmse**2))) if len(a) else float('nan'))
  bootstrap=pd.read_csv(OUT/'paired_segment_bootstrap.csv');bs=bootstrap[(bootstrap.group==group)&(bootstrap.k==k)&(bootstrap.block_segments==2)]
  a=seg[(seg.group==group)&(seg.k==k)];comparisons.append(dict(k=k,quality_rmse=rr[0],descriptor_rmse=rr[1],test_poses=int(a.test_n.sum()/2),segments=int(len(a)/2),difference=rr[1]-rr[0],ci_low=float(bs.iloc[0].ci_low) if len(bs) else float('nan'),ci_high=float(bs.iloc[0].ci_high) if len(bs) else float('nan')))
 random=[]
 for alg in ['ORB','AKAZE','SIFT']:
  a=switch[(switch.detector==alg)&(switch.type=='contrast_rule')].iloc[0]
  for kind in ['random','random_matched_keyframes']:
   x=switch[(switch.detector==alg)&(switch.type==kind)]
   random.append(dict(detector=alg,control=kind,n=len(x),adaptive_ap=float(a.ap_count),random_min=float(x.ap_count.min()),random_median=float(x.ap_count.median()),random_max=float(x.ap_count.max()),random_strictly_below=int(np.sum(x.ap_count<a.ap_count))))
 common_summaries=[]
 for dep in [50.,1e9]:
  a=direct[(direct.source=='selected')&(direct.cutoff==dep)].copy();sets=[]
  for alg in ['ORB','AKAZE','SIFT']:
   for prep in ['raw','fixed','adaptive']:
    b=a[(a.detector==alg)&(a.preprocessing==prep)];sets.append(set(zip(b.i,b.j)))
  common=set.intersection(*sets)
  for alg in ['ORB','AKAZE','SIFT']:
   for prep in ['raw','fixed','adaptive']:
    b=a[(a.detector==alg)&(a.preprocessing==prep)];b=b[[x in common for x in zip(b.i,b.j)]]
    common_summaries.append(dict(detector=alg,preprocessing=prep,cutoff=dep,common_pairs=len(common),median_rotation_error_deg=float(b.rotation_error_deg.median()),median_direction_error_deg=float(b.translation_direction_error_deg.median())))
 writecsv(OUT/'paper_matrix_summary.csv',matrix);writecsv(OUT/'paper_full_policy_comparison.csv',comparisons);writecsv(OUT/'paper_random_comparison.csv',random);writecsv(OUT/'paper_direct_common_pairs.csv',common_summaries)
 dump(OUT/'paper_coverage_summary.json',cov[cov.group.isin(['all_matrix_common','high_depth_matrix_common','rho_sweep_common_depth50','rho_sweep_common_depth1e+09'])].to_dict('records'))
 print('Matrix configs',len(matrix),'with default',sum('accept_50' in r for r in matrix),'high',sum('accept_1e+09' in r for r in matrix),flush=True)
 return dict(matrix=matrix,policy=comparisons,random=random,direct=common_summaries)
if __name__=='__main__':aggregate()
