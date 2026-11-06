"""Frozen raw/fixed/adaptive matching test, using existing endpoint evaluator."""
from common import *
import common,frontend as f,itertools,argparse
from concurrent.futures import ThreadPoolExecutor
DEST=ROOT/'sequence5_results';IMAGEDIR=Path(r'C:\Users\Raaghav\Downloads\archaeo_sequence_5_raw_data/raw_data/images_sequence_5')

def run(alg):
 common.IMAGES=IMAGEDIR;f.OUT=DEST;f.CACHE=DEST/'cache';f.REF=refposes(file=DEST/'reference/new_archaeo_colmap_traj_sequence_05.txt');f.IDS=sorted(f.REF);loop=np.loadtxt(DEST/'reference/colmap_detected_loop_sequence_5.txt',delimiter=',');assert loop.shape==(200,200) and np.isin(loop,[0,1]).all() and np.array_equal(loop,loop.T)
 f.PAIRS=[dict(frame_id_i=i,frame_id_j=j,colmap_overlap_label=int(loop[i//20,j//20])) for i,j in itertools.combinations(f.IDS,2)];f.FUND=[fundamental(f.REF[int(p['frame_id_i'])],f.REF[int(p['frame_id_j'])]) for p in f.PAIRS]
 (DEST/'frontend').mkdir(exist_ok=True);(DEST/'feature_counts').mkdir(exist_ok=True)
 for left,right in [('raw','raw'),('c2t8','c2t8'),('raw','c2t8'),('c2t8','raw')]:f.evaluate(alg,left,right)
 fixed,norm=f.getfeatures(alg,'c2t8');checks=[]
 for i,j in zip(f.IDS[:3],f.IDS[1:4]):
  x,y=correspond(fixed[i],fixed[j],norm,True);xx,yy=f.fast_correspond(fixed[i],fixed[j],norm,True);assert np.array_equal(x,xx) and np.array_equal(y,yy);checks.append(dict(i=i,j=j,identical=True))
 if alg=='ORB':
  scores=[]
  for order,direction in itertools.product(['xyzw','wxyz'],['twc','tcw']):
   ref=refposes(order,direction,file=DEST/'reference/new_archaeo_colmap_traj_sequence_05.txt');residual=[]
   for i,j in zip(f.IDS[:-1],f.IDS[1:]):
    x,y=correspond(fixed[i],fixed[j],norm,True);x,y=x[:100],y[:100]
    if len(x):residual.append(float(np.median(sampson(undistort(x),undistort(y),fundamental(ref[i],ref[j])))))
   scores.append(dict(order=order,direction=direction,pairs=len(residual),median_pair_sampson_px=float(np.median(residual))))
  writecsv(DEST/'reference_conventions.csv',scores)
 dump(DEST/f'frontend_{alg}_validation.json',dict(matcher_checks=checks,reference='publisher-corrected updated export',labels='original publisher overlap matrix restricted by native frame IDs; shared COLMAP provenance, not independent truth',reference_poses=len(f.IDS),pairs=len(f.PAIRS),positive_pairs=sum(int(p['colmap_overlap_label']) for p in f.PAIRS),parameters='same detector getters, feature cap, ratio/mutual matching, CLAHE and 2 px Sampson threshold as Sequence 06'))

def launch():
 def task(alg):
  with (DEST/f'frontend_{alg}.log').open('a',encoding='utf-8') as log:r=subprocess.run([sys.executable,str(Path(__file__)),alg],stdout=log,stderr=subprocess.STDOUT)
  assert r.returncode==0,alg
 with ThreadPoolExecutor(max_workers=3) as pool:list(pool.map(task,['ORB','AKAZE','SIFT']))

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('action');a=p.parse_args();launch() if a.action=='launch' else run(a.action)
