from common import *
import pandas as pd
R2=ROOT/'revision2_results';R2.mkdir(exist_ok=True)
def fastscore(y,c):
 c=np.asarray(c,np.int32);h=np.bincount(c);p=np.bincount(c,weights=y,minlength=len(h));nn=np.cumsum(h[::-1]);pp=np.cumsum(p[::-1]);ap=float(np.sum(np.divide(p[::-1]*pp,nn,out=np.zeros_like(pp),where=nn>0))/sum(y));tp=int(np.sum(y[c>=8]));fp=int(np.sum((1-y)[c>=8]));fn=int(sum(y))-tp
 return ap,2*tp/max(1,2*tp+fp+fn)
def runs(v):return int(np.sum(v&~np.roll(v,1)))
def main(reference_ids=None,result_dir=None):
 global R2
 if result_dir is not None:R2=Path(result_dir);R2.mkdir(exist_ok=True)
 dec=pd.read_csv(DATA/'sequence6_adaptive_v2_full/adaptive_frame_decisions.csv').set_index('frame_id');ids=np.arange(0,3381,20) if reference_ids is None else np.asarray(reference_ids);g=dec.loc[np.arange(3385),'local_std'].to_numpy();actual=g>=10.5;key=actual[ids];others=np.setdiff1d(np.arange(3385),ids);summaries=[];decomp=[];selected=[];nkeys=len(ids);rawkeys=int(key.sum());rawframes=int(actual.sum());index={v:k for k,v in enumerate(ids)}
 for alg in ['ORB','AKAZE','SIFT']:
  files=['c2t8_c2t8','c2t8_raw','raw_c2t8','raw_raw'];a=[pd.read_csv(OUT/'frontend'/f'{alg}_{x}.csv') for x in files];a=[d[d.i.isin(ids)&d.j.isin(ids)].reset_index(drop=True) for d in a];d=a[0];y=d.label.to_numpy(np.int32);i=d.i.map(index).to_numpy(int);j=d.j.map(index).to_numpy(int);c=np.array([x.inliers.to_numpy(np.int32) for x in a]);cols=np.arange(len(i));rows=[]
  def ev(raw):return c[2*raw[i].astype(int)+raw[j].astype(int),cols]
  obs=ev(key);ap,f1=fastscore(y,obs);assert abs(ap-score(y,obs)['ap_count'])<1e-12
  def add(kind,draw,raw,allraw=np.nan):
   cc=ev(raw);aa,ff=fastscore(y,cc);mix=raw[i]!=raw[j];rows.append(dict(detector=alg,null=kind,draw=draw,ap=aa,f1=ff,raw_keyframes=int(raw.sum()),raw_frame_count=allraw,keyframe_circular_raw_runs=runs(raw),positive_mixed_pairs=int(np.sum(mix&(y==1))),mixed_pairs=int(mix.sum())))
  for kind in ['uniform_frame_budget','uniform_frame_keyframe_budgets']:
   rng=np.random.default_rng(20261105)
   for rep in range(10000):
    if kind=='uniform_frame_budget':raw=np.zeros(3385,bool);raw[rng.choice(3385,rawframes,replace=False)]=True;raw=raw[ids]
    else:raw=np.zeros(nkeys,bool);raw[rng.choice(nkeys,rawkeys,replace=False)]=True
    add(kind,rep,raw,rawframes)
  for shift in range(3385):add('full_frame_circular_shift',shift,np.roll(actual,shift)[ids],rawframes)
  for shift in range(nkeys):add('keyframe_circular_shift',shift,np.roll(key,shift),rawframes)
  rev=np.zeros(3385,bool);rev[np.argsort(g,kind='stable')[:rawframes]]=True;add('reversed_contrast_frame_budget',0,rev[ids],rawframes)
  rev2=np.zeros(3385,bool);rev2[ids[np.argsort(g[ids],kind='stable')[:rawkeys]]]=True;rev2[others[np.argsort(g[others],kind='stable')[:rawframes-rawkeys]]]=True;add('reversed_contrast_both_budgets',0,rev2[ids],rawframes);add('observed_contrast',0,key,rawframes)
  for kind in sorted(set(r['null'] for r in rows)):
   x=[r for r in rows if r['null']==kind];v=np.array([r['ap'] for r in x]);tail=int(np.sum(v>=ap-1e-12));enumerated='circular_shift' in kind
   summaries.append(dict(detector=alg,null=kind,n=len(x),observed_ap=ap,minimum_ap=float(v.min()),median_ap=float(np.median(v)),maximum_ap=float(v.max()),tail_ge_observed=tail,tail_fraction=(tail/len(v) if enumerated else ((tail+1)/(len(v)+1) if len(v)>1 else float('nan'))),test_type='exact circular-shift reference distribution including identity' if enumerated else ('plus-one Monte Carlo reference tail; same masks across detectors' if len(v)>1 else 'deterministic control'),median_positive_mixed=float(np.median([r['positive_mixed_pairs'] for r in x])),median_key_runs=float(np.median([r['keyframe_circular_raw_runs'] for r in x]))))
  # Full-frame shifts conditional on precisely the observed keyframe budget.
  x=[r for r in rows if r['null']=='full_frame_circular_shift' and r['raw_keyframes']==rawkeys];v=np.array([r['ap'] for r in x]);summaries.append(dict(detector=alg,null=f'full_frame_shift_conditioned_{rawkeys}_keyframes',n=len(x),observed_ap=ap,minimum_ap=float(v.min()),median_ap=float(np.median(v)),maximum_ap=float(v.max()),tail_ge_observed=int(np.sum(v>=ap-1e-12)),tail_fraction=float(np.mean(v>=ap-1e-12)),test_type='conditional enumerated shifts; not independent draws',median_positive_mixed=float(np.median([r['positive_mixed_pairs'] for r in x])),median_key_runs=float(np.median([r['keyframe_circular_raw_runs'] for r in x]))))
  masks={'raw_raw':key[i]&key[j],'mixed':key[i]!=key[j],'enhanced_enhanced':~key[i]&~key[j]}
  for s,m in masks.items():
   for label in [0,1]:
    mm=m&(y==label);delta=obs[mm]-c[0,mm];decomp.append(dict(detector=alg,stratum=s,label=label,n=int(mm.sum()),fixed_mean_inliers=float(c[0,mm].mean()),adaptive_mean_inliers=float(obs[mm].mean()),mean_loss=float(-delta.mean()),median_loss=float(-np.median(delta)),total_loss=int(-delta.sum()),fraction_pairs_losing=float(np.mean(delta<0)),fraction_pairs_gaining=float(np.mean(delta>0))))
  writecsv(R2/f'switch_null_draws_{alg}.csv',rows);print('Completed nulls',alg,len(rows),flush=True)
 writecsv(R2/'switch_null_summary.csv',summaries);writecsv(R2/'stratum_conditional_losses.csv',decomp)
 dump(R2/'switch_null_interpretation.json',dict(draw_masks_shared_across_detectors=True,pooled_72_pvalue=False,circular_shift_stationarity='conditional reference distribution; cyclic temporal exchangeability is an assumption, not established causality',reversed_rule='exact budget-controlled raw selection on lowest contrasts',validation='histogram tied-score AP equals original sorted AP to 1e-12 on observed policies'))
if __name__=='__main__':main()
