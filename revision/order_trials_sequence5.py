"""Repeat frozen ORB runs with the same four input-order interventions as Seq06."""
from sequence5 import *

if __name__=='__main__':
 target=DEST/'order_runs';target.mkdir(exist_ok=True)
 if len(sys.argv)==1:
  dump(target/'protocol.json',dict(seeds=[20261106,20261107,20261108,20261109],frames=3983,detector='ORB',cutoff=1e9,rho=.06,source_sha256={p:sha(ROOT/'revision'/p) for p in ['run_vo.py','common.py','order_trials_sequence5.py']},intervention='joint correspondence permutation; existing ordered run is trial zero',tuning='none'))
  def task(prep):
   with (target/(prep+'.log')).open('a',encoding='utf-8') as log:r=subprocess.run([sys.executable,__file__,prep],stdout=log,stderr=subprocess.STDOUT)
   assert r.returncode==0,prep
  with ThreadPoolExecutor(max_workers=3) as pool:list(pool.map(task,['raw','fixed','adaptive']))
 else:
  prep=sys.argv[1];assert prep in ['raw','fixed','adaptive']
  common.IMAGES=IMAGEDIR;run_vo.DECISIONS={int(r['frame_id']):r['decision'] for r in readcsv(DEST/'adaptive_frame_decisions.csv')}
  run_vo.run('ORB',prep,seed=[20261106,20261107,20261108,20261109],replicate=True,frames=3983,result_dir=target)
