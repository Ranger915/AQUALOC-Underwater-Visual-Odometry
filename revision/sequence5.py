"""Run the frozen Sequence 06 estimator on Sequence 05; keep outputs separate."""
from common import *
import common,run_vo,argparse
from concurrent.futures import ThreadPoolExecutor

DEST=ROOT/'sequence5_results';DEST.mkdir(exist_ok=True)
SOURCE=Path(r'C:\Users\Raaghav\Downloads\archaeo_sequence_5_raw_data')
IMAGEDIR=SOURCE/'raw_data/images_sequence_5'

def audit():
 files=sorted(IMAGEDIR.glob('frame*.png'));rows=readcsv(SOURCE/'raw_data/img_sequence_5.csv');assert len(files)==len(rows)==3983
 assert [list(r.values())[1].strip() for r in rows]==[p.name for p in files]
 stamps=np.array([int(next(iter(r.values()))) for r in rows],dtype=np.int64);assert np.all(np.diff(stamps)>0)
 # Both archaeological sequences use the same publisher-provided calibration.
 local=DATA/'documents-export-2026-11-4/archaeo_camera_calib.yaml';official=DEST/'reference/archaeo_camera_calib.yaml';assert sha(local)==sha(official)
 decisions=[]
 for i,p in enumerate(files):
  assert p.name==f'frame{i:06d}.png';im=cv2.imread(str(p),0);assert im is not None and im.shape==(608,968)
  contrast=float(im.reshape(8,76,8,121).std(axis=(1,3)).mean())
  if i<3:assert abs(contrast-np.mean([im[y:y+76,x:x+121].std() for y in range(0,608,76) for x in range(0,968,121)]))<1e-10
  decisions.append(dict(frame_id=i,local_std=contrast,decision='raw' if contrast>=10.5 else 'enhanced'))
 writecsv(DEST/'adaptive_frame_decisions.csv',decisions)
 dump(DEST/'protocol.json',dict(sequence=5,frames=len(files),duration_s=float((stamps[-1]-stamps[0])/1e9),nominal_hz=20,measured_median_hz=float(1e9/np.median(np.diff(stamps))),image_shape=[608,968],calibration_sha256=sha(official),timestamp_sha256=sha(SOURCE/'raw_data/img_sequence_5.csv'),adaptive_raw_frames=sum(r['decision']=='raw' for r in decisions),adaptive_threshold=10.5,clahe=dict(clipLimit=2.,tileGridSize=[8,8]),detectors={a:detector(a)[0].getDefaultName() for a in ['ORB','AKAZE','SIFT']},settings='frozen from Sequence 06; no threshold, clip, gate or source-score tuning on Sequence 05',primary_reference='reference/new_archaeo_colmap_traj_sequence_05.txt, publisher-recommended corrected scaling',historical_reference='reference/colmap_traj_sequence_5.txt for replication; reference family not selected by estimator errors',reference_source='https://seafile.lirmm.fr/d/79b03788f29148ca84e5/',source_sha256={p:sha(Path(__file__).parent/p) for p in ['sequence5.py','run_vo.py','common.py']},scope='different sequence from same archaeological acquisition domain; not independent physical truth or a new environment'))
 print('Audited',len(files),'images; raw decisions',sum(r['decision']=='raw' for r in decisions),flush=True)

def run(alg,prep):
 common.IMAGES=IMAGEDIR;run_vo.DECISIONS={int(r['frame_id']):r['decision'] for r in readcsv(DEST/'adaptive_frame_decisions.csv')}
 comparisons=[(.06,'quality',c) for c in [50.,1e9]]
 if alg=='ORB' and prep=='fixed':comparisons +=[(.06,'descriptor_first',c) for c in [50.,1e9]]
 run_vo.run(alg,prep,result_dir=DEST,frames=3983,comparisons=comparisons)

def launch():
 tasks=[(a,p) for a in ['ORB','AKAZE','SIFT'] for p in ['fixed','raw','adaptive']]
 def task(x):
  with (DEST/f'{x[0]}_{x[1]}.log').open('a',encoding='utf-8') as log:
   result=subprocess.run([sys.executable,str(Path(__file__)),*x],stdout=log,stderr=subprocess.STDOUT)
  assert result.returncode==0,(x,result.returncode)
  print('Completed',*x,flush=True)
 with ThreadPoolExecutor(max_workers=3) as pool:list(pool.map(task,tasks))

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('action');p.add_argument('prep',nargs='?');a=p.parse_args()
 if a.action=='audit':audit()
 elif a.action=='launch':launch()
 else:run(a.action,a.prep)
