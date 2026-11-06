"""Prepare and run the pinned upstream monocular system on frozen inputs."""
from common import *
from concurrent.futures import ThreadPoolExecutor
import re
DEST=ROOT/'baseline_results';TOOLS=ROOT/'baseline_tools';COMMIT='4452a3c4ab75b1cde34e5505a36ec3f9edcdc4c4'
EXE='/opt/aqualoc/ORB_SLAM3/Examples/Monocular/mono_aqualoc';VOC='/opt/aqualoc/ORB_SLAM3/Vocabulary/ORBvoc.txt'

def linux(p):
 p=Path(p).resolve();assert p.drive=='C:'
 return '/mnt/c'+p.as_posix()[2:]

def prepare():
 DEST.mkdir(exist_ok=True)
 template=(TOOLS/'upstream'/('ORB_SLAM3-'+COMMIT)/'Examples/Monocular/EuRoC.yaml').read_text()
 config={'Camera1.fx':K[0,0],'Camera1.fy':K[1,1],'Camera1.cx':K[0,2],'Camera1.cy':K[1,2],'Camera1.k1':DIST[0],'Camera1.k2':DIST[1],'Camera1.p1':DIST[2],'Camera1.p2':DIST[3],'Camera.width':968,'Camera.height':608,'Camera.fps':20,'ORBextractor.nFeatures':1500}
 for key,value in config.items():template,n=re.subn(r'(?m)^'+re.escape(key)+r':.*$',key+': '+str(value),template);assert n==1,key
 template=re.sub(r'(?m)^Camera.new(?:Width|Height):.*\n','',template)
 (DEST/'aqualoc.yaml').write_text(template)
 fs=cv2.FileStorage(str(DEST/'aqualoc.yaml'),cv2.FILE_STORAGE_READ)
 assert fs.isOpened() and fs.getNode('Camera.newWidth').empty() and fs.getNode('Camera.newHeight').empty()
 for key,value in config.items():assert abs(fs.getNode(key).real()-float(value))<1e-9,key
 for key,value in [('ORBextractor.scaleFactor',1.2),('ORBextractor.nLevels',8),('ORBextractor.iniThFAST',20),('ORBextractor.minThFAST',7)]:assert abs(fs.getNode(key).real()-value)<1e-9,key
 fs.release()
 inputs={}
 for seq,count in [(5,3983),(6,3385)]:
  source=Path(fr'C:\Users\Raaghav\Downloads\archaeo_sequence_{seq}_raw_data');stamps=readcsv(source/f'raw_data/img_sequence_{seq}.csv');assert len(stamps)==count
  assert [list(r.values())[1].strip() for r in stamps]==[f'frame{i:06d}.png' for i in range(count)]
  decision=ROOT/'sequence5_results/adaptive_frame_decisions.csv' if seq==5 else source/'sequence6_adaptive_v2_full/adaptive_frame_decisions.csv';raw={int(r['frame_id']):r['decision']=='raw' for r in readcsv(decision)}
  times=[int(next(iter(r.values()))) for r in stamps];assert np.all(np.diff(times)>0)
  paths=[source/f'raw_data/images_sequence_{seq}'/f'frame{i:06d}.png' for i in range(count)];assert all(p.exists() for p in paths)
  lines=[f'{t-times[0]} {int(raw[i])} {linux(p)}\n' for i,(t,p) in enumerate(zip(times,paths))]
  (DEST/f'sequence{seq}.manifest').write_text(''.join(lines));inputs[seq]=dict(frames=count,timestamps_sha256=sha(source/f'raw_data/img_sequence_{seq}.csv'),decision_sha256=sha(decision),manifest_sha256=sha(DEST/f'sequence{seq}.manifest'))
 (DEST/'smoke.manifest').write_text(''.join((DEST/'sequence6.manifest').read_text().splitlines(True)[:80]))
 dump(DEST/'protocol.json',dict(source='https://github.com/UZ-SLAMLab/ORB_SLAM3',commit=COMMIT,inputs=inputs,preprocessing=['raw','fixed','adaptive'],process_repeats=3,seeds=[20261105,20261106,20261107],nfeatures=1500,pyramid_scale=1.2,levels=8,FAST=[20,7],clahe=[2,8,8],viewer=False,resizing=False,imu=False,pressure=False,pacing='dataset timestamp gaps; no catch-up for slow frames',scope='monocular established-system baseline, calibrated camera, no Sequence05 tuning',export='causal per-frame state/pose/map ID and final optimized trajectory separately for every nonempty map; select largest map by keyframe count, never reference error',patch='C++14 compatibility, input adapter, read-only Atlas/thread-finish accessors; no estimator change',source_sha256={p:sha(TOOLS/p) for p in ['mono_aqualoc.cc','build_orbslam3.sh']},config_sha256=sha(DEST/'aqualoc.yaml')))

def run(seq,prep,trial,smoke=False):
 name='smoke' if smoke else f'sequence{seq}_{prep}_trial{trial}';out=DEST/name;out.mkdir(exist_ok=True)
 if (out/'done.json').exists():return
 manifest=DEST/('smoke.manifest' if smoke else f'sequence{seq}.manifest');expected=len(manifest.read_text().splitlines());seed=20261105+trial
 with (out/'run.log').open('w',encoding='utf-8') as log:
  result=subprocess.run(['wsl','-d','AqualocBaseline','--cd',linux(out),'--','env','OMP_NUM_THREADS=1','OPENBLAS_NUM_THREADS=1',EXE,VOC,linux(DEST/'aqualoc.yaml'),linux(manifest),prep,linux(out),str(seed)],stdout=log,stderr=subprocess.STDOUT)
 assert result.returncode==0,(name,result.returncode)
 rows=readcsv(out/'online.csv');assert len(rows)==expected==int((out/'complete.txt').read_text())
 assert [int(r['frame']) for r in rows]==list(range(expected))
 dump(out/'done.json',dict(name=name,sequence=seq,preprocessing=prep,trial=trial,seed=seed,frames=expected,tracking_OK=sum(int(r['state'])==2 for r in rows),maps=readcsv(out/'maps.csv'),config_sha256=sha(DEST/'aqualoc.yaml'),completed=True));print('Completed',name,flush=True)

if __name__=='__main__':
 action=sys.argv[1];assert action in ['prepare','smoke','launch']
 if action=='prepare':prepare()
 elif action=='smoke':run(6,'fixed',0,True)
 else:
  with ThreadPoolExecutor(max_workers=2) as pool:list(pool.map(lambda x:run(*x),[(s,p,t) for s in [6,5] for p in ['raw','fixed','adaptive'] for t in range(3)]))
