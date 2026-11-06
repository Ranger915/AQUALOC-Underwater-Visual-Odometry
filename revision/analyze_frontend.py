from common import *
import pandas as pd,itertools
def main():
 decisions=pd.read_csv(DATA/'sequence6_adaptive_v2_full/adaptive_frame_decisions.csv').set_index('frame_id');ids=np.arange(0,3381,20);gate=decisions.loc[ids,'local_std'].to_numpy();allgate=decisions.loc[np.arange(3385),'local_std'].to_numpy()
 summary=[];switch=[];vertex=[];labels=[];checks=[];features=[];reproduction=[]
 ref=refposes()
 for alg in ['ORB','AKAZE','SIFT']:
  base=OUT/'frontend';tables={}
  for code,filename in [(0,'c2t8_c2t8'),(1,'c2t8_raw'),(2,'raw_c2t8'),(3,'raw_raw')]:
   p=base/f'{alg}_{filename}.csv'
   if not p.exists():continue
   tables[code]=pd.read_csv(p)
  if len(tables)!=4:continue
  d=tables[0];y=d.label.to_numpy();ii=(d.i.to_numpy()//20).astype(int);jj=(d.j.to_numpy()//20).astype(int);counts=np.array([tables[c].inliers.to_numpy() for c in range(4)])
  def counts_for(raw):return counts[raw[ii].astype(int)*2+raw[jj].astype(int),np.arange(len(ii))]
  observed=counts_for(gate>=10.5);fixed=counts[0]
  old=pd.read_csv(DATA/'sequence6_adaptive_v2_mutual_matching_results/sequence6_mutual_feature_metrics.csv')
  old=old[old.detector==('AKAZE_MLDB' if alg=='AKAZE' else alg)].rename(columns={'frame_id_i':'i','frame_id_j':'j'})
  aligned=d[['i','j']].merge(old[['i','j','mutual_inliers_2px']],on=['i','j'],how='left')
  reproduction.append(dict(detector=alg,n=len(observed),changed_vs_saved_adaptive=int(np.sum(observed!=aligned.mutual_inliers_2px.to_numpy())),max_abs_difference=float(np.max(np.abs(observed-aligned.mutual_inliers_2px.to_numpy())))))
  for path in sorted(base.glob(f'{alg}_*.csv')):
   mode=path.stem[len(alg)+1:];a=pd.read_csv(path)
   if mode.split('_')[0]!=mode.split('_')[-1] and mode not in ['raw_matched_raw_matched','raw_budget_raw_budget']:continue
   r=score(a.label,a.inliers);r.update(detector=alg,mode=mode);summary.append(r)
  r=score(y,observed);r.update(detector=alg,mode='adaptive_observed');summary.append(r)
  for th in range(21):
   c=counts_for(gate>=th);r=score(y,c);r.update(detector=alg,type='threshold',threshold=th,seed='',fraction_raw_all=float(np.mean(allgate>=th)),fraction_raw_keyframes=float(np.mean(gate>=th)));switch.append(r)
  for seed in range(24):
   rng=np.random.default_rng(20261105+seed);raw=np.zeros(3385,bool);raw[rng.choice(3385,808,replace=False)]=True;c=counts_for(raw[ids]);r=score(y,c);r.update(detector=alg,type='random',threshold='',seed=seed,fraction_raw_all=float(raw.mean()),fraction_raw_keyframes=float(raw[ids].mean()));switch.append(r)
  for seed in range(24):
   rng=np.random.default_rng(20262105+seed);raw=np.zeros(3385,bool);raw[rng.choice(ids,int(np.sum(gate>=10.5)),replace=False)]=True;others=np.setdiff1d(np.arange(3385),ids);raw[rng.choice(others,808-int(raw.sum()),replace=False)]=True;c=counts_for(raw[ids]);r=score(y,c);r.update(detector=alg,type='random_matched_keyframes',threshold='',seed=seed,fraction_raw_all=float(raw.mean()),fraction_raw_keyframes=float(raw[ids].mean()));switch.append(r)
  r=score(y,observed);r.update(detector=alg,type='contrast_rule',threshold=10.5,seed='',fraction_raw_all=float(np.mean(allgate>=10.5)),fraction_raw_keyframes=float(np.mean(gate>=10.5)));switch.append(r)
  raw=gate>=10.5;stratum=np.where(raw[ii]&raw[jj],'raw_raw',np.where(~raw[ii]&~raw[jj],'enhanced_enhanced','mixed'))
  for s in ['raw_raw','enhanced_enhanced','mixed']:
   m=stratum==s;checks.append(dict(detector=alg,stratum=s,n=int(m.sum()),changed=int(np.sum(observed[m]!=fixed[m])),sum_inlier_delta=int(np.sum(observed[m]-fixed[m]))))
  # Induced-vertex resampling: inclusion of unique sampled blocks, without
  # product multiplicity. This sensitivity is not claimed to have nominal coverage.
  for B in [1,5,10,20]:
   rng=np.random.default_rng(20261105);nblocks=math.ceil(170/B);effects=[]
   for rep in range(1000):
    sampled=np.unique(rng.integers(0,nblocks,size=nblocks));inc=np.isin(np.arange(170)//B,sampled);mask=inc[ii]&inc[jj]
    if not np.any(y[mask]):continue
    a=score(y[mask],observed[mask]);f=score(y[mask],fixed[mask]);effects.append((a['ap_count']-f['ap_count'],a['f1']-f['f1'],int(mask.sum())))
   e=np.array(effects)
   for j,metric in enumerate(['ap_count','f1']):vertex.append(dict(detector=alg,block_keyframes=B,metric=metric,observed_delta=score(y,observed)[metric]-score(y,fixed)[metric],ci_low=float(np.quantile(e[:,j],.025)),ci_high=float(np.quantile(e[:,j],.975)),replicates=len(e),median_pairs=float(np.median(e[:,2])),resampling='induced unique sampled vertices; percentile sensitivity, not calibrated confidence'))
  for distance in [.5,1.,2.]:
   for view in [30.,60.,90.]:
    yy=[]
    for i,j in zip(d.i,d.j):
     Ri,Ci=ref[int(i)];Rj,Cj=ref[int(j)];viewangle=angle(Ri[:,2],Rj[:,2]);yy.append(int(np.linalg.norm(Cj-Ci)<=distance and viewangle<=view))
    yy=np.array(yy)
    for code,name in [(0,'fixed'),(3,'raw'),(-1,'adaptive')]:
     r=score(yy,observed if code==-1 else counts[code]);r.update(detector=alg,mode=name,max_distance_units=distance,max_view_angle_deg=view,label_definition='camera-center distance plus viewing-axis angle; heuristic visibility proxy');labels.append(r)
  for p in (OUT/'feature_counts').glob(f'{alg}_*.csv'):
   a=pd.read_csv(p);features.append(dict(detector=alg,mode=p.stem[len(alg)+1:],n=len(a),mean_features=float(a.features.mean()),median_features=float(a.features.median()),p10=float(a.features.quantile(.1)),p90=float(a.features.quantile(.9))))
 writecsv(OUT/'frontend_primary_summary.csv',summary);writecsv(OUT/'switch_threshold_random_controls.csv',switch);writecsv(OUT/'new_stratified_decomposition.csv',checks);writecsv(OUT/'vertex_bootstrap_sensitivity.csv',vertex);writecsv(OUT/'geometric_label_sensitivity.csv',labels);writecsv(OUT/'feature_count_summary.csv',features)
 writecsv(OUT/'adaptive_reproduction_check.csv',reproduction)
 print('Summaries',len(summary),'switch runs',len(switch),'vertex',len(vertex),flush=True)
def angle(a,b):return float(np.degrees(np.arccos(np.clip(a@b/(np.linalg.norm(a)*np.linalg.norm(b)),-1,1))))
if __name__=='__main__':
 import math
 main()
