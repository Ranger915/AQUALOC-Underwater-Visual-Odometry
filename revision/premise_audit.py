from common import *
import pandas as pd,gzip

def convention():
 p=DATA/'raw_data/sequence6_mutual_matching_results/mutual_matches_ORB_clahe.csv.gz'
 df=pd.read_csv(p)
 df=df[df.overlap_label==1].copy()
 # Equal cap per positive pair prevents easy adjacent pairs dominating evidence.
 df=df.groupby(['frame_id_i','frame_id_j'],sort=True).head(100)
 result=[];pairrows=[]
 for order in ['xyzw','wxyz']:
  for direction in ['twc','tcw']:
   poses=refposes(order,direction);allerr=[];med=[]
   for (i,j),g in df.groupby(['frame_id_i','frame_id_j'],sort=True):
    a=g[['undistorted_x_i','undistorted_y_i']].to_numpy();b=g[['undistorted_x_j','undistorted_y_j']].to_numpy()
    e=sampson(a,b,fundamental(poses[int(i)],poses[int(j)]));allerr.extend(e);med.append(np.median(e))
    pairrows.append(dict(order=order,direction=direction,i=i,j=j,n=len(e),median_px=float(np.median(e)),fraction_2px=float(np.mean(e<=2))))
   result.append(dict(order=order,direction=direction,matches=len(allerr),pairs=len(med),pooled_median_px=float(np.median(allerr)),median_of_pair_medians_px=float(np.median(med)),fraction_2px=float(np.mean(np.array(allerr)<=2))))
 writecsv(OUT/'reference_convention_hypotheses.csv',result);writecsv(OUT/'reference_convention_per_pair.csv',pairrows)
 result.sort(key=lambda r:r['median_of_pair_medians_px']);dump(OUT/'reference_choice.json',{'best_geometric_hypothesis':result[0],'runner_up':result[1],'method':'same saved reciprocal-ratio ORB CLAHE image correspondences; no trajectory fitting; primary ordering by equal-pair median Sampson residual','not_proven':['metric units','exact original exporter provenance','independent physical ground truth']})
 print('CONVENTION',json.dumps(result),flush=True)

def strata():
 decisions=pd.read_csv(DATA/'sequence6_adaptive_v2_full/adaptive_frame_decisions.csv').set_index('frame_id')
 original=pd.read_csv(DATA/'raw_data/sequence6_mutual_matching_results/sequence6_mutual_feature_metrics.csv')
 adaptive=pd.read_csv(DATA/'sequence6_adaptive_v2_mutual_matching_results/sequence6_mutual_feature_metrics.csv')
 records=[];decomp=[];checks=[]
 for det in ['ORB','AKAZE_MLDB','SIFT']:
  f=original[(original.detector==det)&(original.preprocessing=='clahe')].set_index(['frame_id_i','frame_id_j']).sort_index()
  a=adaptive[adaptive.detector==det].set_index(['frame_id_i','frame_id_j']).sort_index();assert f.index.equals(a.index)
  iraw=np.array([decisions.loc[int(i),'decision']=='raw' for i,j in f.index]);jraw=np.array([decisions.loc[int(j),'decision']=='raw' for i,j in f.index])
  groups=np.where(iraw&jraw,'raw_raw',np.where(~iraw&~jraw,'enhanced_enhanced','mixed'))
  for group in ['all','raw_raw','enhanced_enhanced','mixed']:
   keep=np.ones(len(f),bool) if group=='all' else groups==group
   for policy,d in [('fixed',f),('adaptive',a)]:
    r=score(d.overlap_label.to_numpy()[keep],d.mutual_inliers_2px.to_numpy()[keep]);r.update(detector=det,policy=policy,stratum=group);records.append(r)
   delta=a.mutual_inliers_2px.to_numpy()[keep]-f.mutual_inliers_2px.to_numpy()[keep]
   decomp.append(dict(detector=det,stratum=group,n=int(keep.sum()),changed_counts=int(np.sum(delta!=0)),mean_inlier_difference=float(delta.mean()),sum_inlier_difference=int(delta.sum()),fixed_tp=records[-2]['tp'],adaptive_tp=records[-1]['tp'],fixed_fp=records[-2]['fp'],adaptive_fp=records[-1]['fp']))
  mask=groups=='enhanced_enhanced'
  checks.append(dict(detector=det,enhanced_pairs=int(mask.sum()),exact_same_counts=bool(np.all(f.mutual_inliers_2px.to_numpy()[mask]==a.mutual_inliers_2px.to_numpy()[mask])),maximum_abs_difference=int(np.max(np.abs(f.mutual_inliers_2px.to_numpy()[mask]-a.mutual_inliers_2px.to_numpy()[mask])))))
 writecsv(OUT/'saved_stratified_scores.csv',records);writecsv(OUT/'saved_stratified_decomposition.csv',decomp);dump(OUT/'enhanced_identity_checks.json',checks)
 print('IDENTITY',checks,flush=True)

def parameters():
 out={}
 for name in ['ORB','AKAZE','SIFT']:
  d,_=detector(name);params={}
  for attr in dir(d):
   if attr.startswith('get'):
    try:
     x=getattr(d,attr)()
     if isinstance(x,(int,float,str,bool)):params[attr[3:]]=x
    except Exception:pass
  out[name]=params
 dump(OUT/'detector_parameters.json',out)
 env={'python':sys.version,'python_executable':sys.executable,'numpy':np.__version__,'opencv':cv2.__version__,'platform':platform.platform(),'opencv_threads':cv2.getNumThreads(),'cpu_count':os.cpu_count(),'original_environment':'distinct from new experiment environment','source_sha256':{p.name:sha(p) for p in Path(__file__).parent.glob('*.py')}}
 dump(OUT/'new_environment.json',env);(OUT/'opencv_build.txt').write_text(cv2.getBuildInformation(),encoding='utf-8')
 x=subprocess.run([sys.executable,'-m','pip','freeze'],capture_output=True,text=True);(OUT/'pip_freeze.txt').write_text(x.stdout+x.stderr,encoding='utf-8')
 print('PARAMETERS',out,flush=True)

if __name__=='__main__':
 parameters();convention();strata()
