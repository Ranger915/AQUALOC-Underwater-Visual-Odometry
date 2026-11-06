"""Score frozen frontend policies on the corrected references' native subsets."""
from common import *
from fast_match import fast_correspond
import pandas as pd,math
DEST=ROOT/'sequence5_results'

def main():
 summary=[];ranges=[];identities=[]
 for seq in [5,6]:
  ref=refposes(file=DEST/'reference'/f'new_archaeo_colmap_traj_sequence_{seq:02d}.txt');ids=sorted(ref);index={v:i for i,v in enumerate(ids)};folder=DEST if seq==5 else OUT
  dec=readcsv(DEST/'adaptive_frame_decisions.csv' if seq==5 else DATA/'sequence6_adaptive_v2_full/adaptive_frame_decisions.csv');raw={int(r['frame_id']):r['decision']=='raw' for r in dec}
  for alg in ['ORB','AKAZE','SIFT']:
   arrays={}
   for label in ['raw_raw','raw_c2t8','c2t8_raw','c2t8_c2t8']:
    d=pd.read_csv(folder/'frontend'/f'{alg}_{label}.csv');arrays[label]=d[d.i.isin(ids)&d.j.isin(ids)].reset_index(drop=True)
   base=arrays['raw_raw'];assert len(base)==len(ids)*(len(ids)-1)//2
   for d in arrays.values():assert np.array_equal(base[['i','j','label']],d[['i','j','label']])
   ii=np.array([index[i] for i in base.i]);jj=np.array([index[i] for i in base.j]);y=base.label.to_numpy(int);fixed=arrays['c2t8_c2t8'].inliers.to_numpy(int);observed=np.array([arrays[('raw' if raw[int(i)] else 'c2t8')+'_'+('raw' if raw[int(j)] else 'c2t8')].inliers.iloc[k] for k,(i,j) in enumerate(zip(base.i,base.j))]);rr=base.i.map(raw).to_numpy(bool);ss=base.j.map(raw).to_numpy(bool);enhanced=~rr&~ss;assert np.array_equal(observed[enhanced],fixed[enhanced])
   identities.append(dict(sequence=seq,detector=alg,pairs=len(base),positives=int(y.sum()),raw_reference_images=sum(raw[i] for i in ids),enhanced_enhanced_pairs=int(enhanced.sum()),enhanced_identity=True,mixed_pairs=int(np.sum(rr!=ss)),positive_mixed_pairs=int(np.sum((rr!=ss)&(y==1)))))
   for prep,counts in [('raw',base.inliers.to_numpy(int)),('fixed',fixed),('adaptive',observed)]:summary.append(dict(sequence=seq,detector=alg,preprocessing=prep,**score(y,counts)))
   for block in [5,10,20]:
    rng=np.random.default_rng(20261105);nblocks=math.ceil(len(ids)/block);draws=[]
    for _ in range(1000):
     included=np.isin(np.arange(len(ids))//block,np.unique(rng.integers(0,nblocks,nblocks)));mask=included[ii]&included[jj]
     if not np.any(y[mask]):continue
     a=score(y[mask],observed[mask]);b=score(y[mask],fixed[mask]);draws.append([a[k]-b[k] for k in ['ap_count','f1']])
    for col,metric in enumerate(['ap_count','f1']):
     lo,hi=np.quantile(np.array(draws)[:,col],[.025,.975]);ranges.append(dict(sequence=seq,detector=alg,metric=metric,block_keyframes=block,adaptive_minus_fixed=score(y,observed)[metric]-score(y,fixed)[metric],low=float(lo),high=float(hi),replicates=len(draws)))
 writecsv(DEST/'frontend_transfer_summary.csv',summary);writecsv(DEST/'frontend_transfer_ranges.csv',ranges);writecsv(DEST/'frontend_transfer_identity.csv',identities)
 old=refposes();new=refposes(file=DEST/'reference/new_archaeo_colmap_traj_sequence_06.txt');ids=sorted(new);checks=[]
 for i,j in zip(ids[:10],ids[1:11]):
  features=[]
  for fid in [i,j]:
   with np.load(CACHE/'reference_features/ORB/c2t8'/f'{fid}.npz') as z:features.append((z['p'],z['d']))
  x,y=fast_correspond(*features,cv2.NORM_HAMMING,True);a=sampson(undistort(x),undistort(y),fundamental(old[i],old[j]));b=sampson(undistort(x),undistort(y),fundamental(new[i],new[j]));assert np.allclose(a,b,rtol=1e-7,atol=1e-7) and np.array_equal(a<=2,b<=2);checks.append(dict(i=i,j=j,matches=len(x),inliers_identical=True,max_sampson_difference_px=float(np.max(abs(a-b))) if len(a) else 0.))
 writecsv(DEST/'sequence6_reference_epipolar_invariance.csv',checks)
 discovery=[];features=[];ids=set(new)
 for alg in ['ORB','AKAZE','SIFT']:
  for path in (OUT/'frontend').glob(f'{alg}_*.csv'):
   if '.partial.' in path.name:continue
   d=pd.read_csv(path);d=d[d.i.isin(ids)&d.j.isin(ids)];discovery.append(dict(detector=alg,mode=path.stem[len(alg)+1:],**score(d.label.to_numpy(int),d.inliers.to_numpy(int))))
  for mode in ['raw','c2t8','raw_budget','raw_matched']:
   d=pd.read_csv(OUT/'feature_counts'/f'{alg}_{mode}.csv');d=d[d.frame.isin(ids)];features.append(dict(detector=alg,mode=mode,frames=len(d),mean_features=float(d.features.mean()),median_features=float(d.features.median())))
 writecsv(DEST/'discovery_frontend_corrected.csv',discovery);writecsv(DEST/'discovery_feature_counts_corrected.csv',features)
 dump(DEST/'frontend_transfer_protocol.json',dict(sequence5='four endpoint tables recomputed on 183 corrected native reference frames',sequence6='saved endpoint tables restricted to 166 corrected native frames; corrected retained poses differ by a global Sim(3), which preserves epipolar geometry; ten correspondence-set checks verify numerical residual/inlier invariance',parameters='frozen default detector getters, response cap 1500, mutual ratio 0.75, Sampson distance <=2 px, support >=8',labels='publisher overlap labels, restricted by retained native frame identities; shared SIFT/COLMAP provenance remains',uncertainty='1000 induced-vertex resamples at block lengths 5,10,20; descriptive within-sequence sensitivities, not population confidence',not_repeated='CLAHE parameter sweeps, threshold tuning, and random-switch null families; independent transfer uses frozen policies only'))
 print('Frozen frontend transfer summarized:',len(summary),'rows',flush=True)

if __name__=='__main__':main()
