from common import *
from concurrent.futures import ThreadPoolExecutor,as_completed
tasks=[('ORB','fixed'),('ORB','raw'),('ORB','adaptive'),('AKAZE','fixed'),('SIFT','fixed'),('AKAZE','raw'),('SIFT','raw'),('AKAZE','adaptive'),('SIFT','adaptive')]
def task(x):
 log=OUT/f'vo_{x[0]}_{x[1]}.log'
 with log.open('w',encoding='utf-8') as f:
  p=subprocess.run([sys.executable,str(Path(__file__).parent/'run_vo.py'),*x],stdout=f,stderr=subprocess.STDOUT)
 return x,p.returncode
if __name__=='__main__':
 with ThreadPoolExecutor(max_workers=3) as ex:
  for f in as_completed([ex.submit(task,x) for x in tasks]):print('Completed',f.result(),flush=True)
