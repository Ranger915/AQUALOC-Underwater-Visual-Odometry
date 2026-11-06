"""Descriptive isotropic-direction reference and native-pair comparisons."""
from dense_reference import *
import pandas as pd,itertools,math
R3=ROOT/'revision3_results';R3.mkdir(exist_ok=True)

def block_interval(df,column,stat=np.median,block_frames=400):
 blocks=[x[column].to_numpy(float) for _,x in df.groupby(df.frame.astype(int)//block_frames)];rng=np.random.default_rng(20261106);vals=[]
 for _ in range(2000):
  ix=rng.integers(0,len(blocks),len(blocks));vals.append(float(stat(np.concatenate([blocks[i] for i in ix]))))
 return len(blocks),*map(float,np.quantile(vals,[.025,.975]))

def main():
 short=pd.read_csv(R2/'short_gap_cutoff_probe.csv');short=short[(short.source=='selected')&(short.accepted==1)].copy();short['direction']=short.direction_error_deg;short['rotation']=short.rotation_error_deg
 direct=pd.read_csv(OUT/'direct_two_view_motion.csv');direct=direct[(direct.source=='selected')&(direct.accepted==1)].copy();direct['frame']=direct.j;direct['direction']=direct.translation_direction_error_deg;direct['rotation']=direct.rotation_error_deg
 direct['reference_clean']=[int(CLEAN[int(i)] and CLEAN[int(j)]) for i,j in zip(direct.i,direct.j)]
 sets=[]
 for cut in [50,1e9]:
  for clean in [False,True]:
   d=short[short.cutoff==cut];d=d[d.reference_clean==1] if clean else d;sets.append(('short_gap_ORB_fixed',cut,'clean_endpoints' if clean else 'all_reference','ORB_fixed',d))
   for alg in ['ORB','AKAZE','SIFT']:
    d=direct[(direct.cutoff==cut)&(direct.detector==alg)&(direct.preprocessing=='fixed')];d=d[d.reference_clean==1] if clean else d;sets.append(('native_one_second',cut,'clean_endpoints' if clean else 'all_reference',alg+'_fixed',d))
 out=[]
 for domain,cut,mask,name,d in sets:
  d=d[np.isfinite(d.direction)].copy();assert len(d)
  n,lo,hi=block_interval(d,'direction');out.append(dict(domain=domain,cutoff=cut,mask=mask,configuration=name,n=len(d),calendar_blocks=n,median_direction_deg=float(d.direction.median()),median_direction_block_low=lo,median_direction_block_high=hi,median_rotation_deg=float(d.rotation.median()),mean_cosine=float(np.cos(np.radians(d.direction)).mean()),fraction_below_30=float(np.mean(d.direction<30)),fraction_below_60=float(np.mean(d.direction<60)),fraction_below_90=float(np.mean(d.direction<90)),isotropic_median_deg=90.,isotropic_mean_cosine=0.,isotropic_fraction_below_30=float((1-np.cos(np.pi/6))/2),isotropic_fraction_below_60=.25,isotropic_fraction_below_90=.5))
 writecsv(R3/'direction_chance_diagnostics.csv',out)
 # Paired native-pair contrasts on a common acceptance set, not availability-
 # conditioned medians with different pair identities.
 pairs=[];common=[]
 for clean in [False,True]:
  for cut in [50,1e9]:
   df=direct[(direct.cutoff==cut)&(direct.preprocessing=='fixed')];df=df[df.reference_clean==1] if clean else df
   ids=set.intersection(*[set(df[df.detector==a].frame.astype(int)) for a in ['ORB','AKAZE','SIFT']]);d=df[df.frame.isin(ids)]
   common.extend(dict(mask='clean_endpoints' if clean else 'all_reference',cutoff=cut,detector=r.detector,frame=int(r.frame),direction=float(r.direction),rotation=float(r.rotation)) for r in d.itertuples())
   for a,b in itertools.combinations(['ORB','AKAZE','SIFT'],2):
    aa=d[d.detector==a].set_index('frame').sort_index();bb=d[d.detector==b].set_index('frame').sort_index();assert aa.index.tolist()==bb.index.tolist()
    for metric in ['direction','rotation']:
     delta=pd.DataFrame(dict(frame=aa.index.to_numpy(),difference=bb[metric].to_numpy()-aa[metric].to_numpy()));n,lo,hi=block_interval(delta,'difference');pairs.append(dict(mask='clean_endpoints' if clean else 'all_reference',cutoff=cut,a=a,b=b,metric=metric,common_pairs=len(aa),calendar_blocks=n,median_paired_b_minus_a_deg=float(delta.difference.median()),low=lo,high=hi,replicates=2000,block_frames=400))
 writecsv(R3/'native_direct_common_pairs.csv',common);writecsv(R3/'native_direct_paired_ranges.csv',pairs)
 # Closed-form checks for a uniformly random oriented direction on S^2.
 assert abs((1-np.cos(np.pi/2))/2-.5)<1e-14
 rng=np.random.default_rng(20261106);v=rng.normal(size=(100000,3));v/=np.linalg.norm(v,axis=1)[:,None];theta=np.degrees(np.arccos(np.clip(v[:,2],-1,1)));assert abs(np.median(theta)-90)<.7
 dump(R3/'direction_diagnostics_protocol.json',dict(random_reference='oriented isotropic 3D unit direction: F(theta)=(1-cos(theta))/2 for 0<=theta<=180 degrees; median 90 degrees, mean cosine zero',interpretation='descriptive geometric comparator, not a population significance test or useful-VO threshold',uncertainty='resample nonoverlapping 400-frame calendar blocks, 2000 draws; sparse within-sequence sensitivity only',short_gap='uses interpolated reference and is sensitive to unresolved within-second motion',native='uses only supplied one-second endpoint poses, no interpolation or imposed step lengths; same accepted pair intersection for cross-detector contrasts',validation=dict(isotropic_simulation_n=100000,simulated_median_deg=float(np.median(theta)),cdf_90=.5)))
 print('Direction diagnostics complete:',len(out),'summaries and',len(pairs),'paired native contrasts',flush=True)

if __name__=='__main__':main()
