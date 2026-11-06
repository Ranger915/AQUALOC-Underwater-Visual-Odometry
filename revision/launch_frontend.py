from common import *
from concurrent.futures import ThreadPoolExecutor,as_completed
def run(alg):
 with (OUT/f'frontend_{alg}.log').open('w') as f:p=subprocess.run([sys.executable,str(Path(__file__).parent/'frontend.py'),alg],stdout=f,stderr=subprocess.STDOUT)
 return alg,p.returncode
if __name__=='__main__':
 with ThreadPoolExecutor(max_workers=3) as ex:
  for f in as_completed([ex.submit(run,a) for a in (sys.argv[1:] or ['ORB','AKAZE','SIFT'])]):print('Completed',f.result(),flush=True)
