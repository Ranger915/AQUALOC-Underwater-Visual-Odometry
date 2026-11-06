from common import *
import ctypes
def naive(im):
 h,w=im.shape;ys=np.linspace(0,h,9,dtype=int);xs=np.linspace(0,w,9,dtype=int)
 g=float(np.std(im.astype(np.float64)));l=float(np.mean([np.std(im[ys[i]:ys[i+1],xs[j]:xs[j+1]].astype(np.float64)) for i in range(8) for j in range(8)]));return g,l
def reshape(im):
 z=im.reshape(8,76,8,121).transpose(0,2,1,3).astype(np.float64)
 return float(im.std(dtype=np.float64)),float(z.std(axis=(2,3)).mean())
def integral(im):
 a,b=cv2.integral2(im,sdepth=cv2.CV_64F,sqdepth=cv2.CV_64F);ys=np.arange(0,609,76);xs=np.arange(0,969,121)
 sums=a[ys[1:,None],xs[None,1:]]-a[ys[:-1,None],xs[None,1:]]-a[ys[1:,None],xs[None,:-1]]+a[ys[:-1,None],xs[None,:-1]]
 squares=b[ys[1:,None],xs[None,1:]]-b[ys[:-1,None],xs[None,1:]]-b[ys[1:,None],xs[None,:-1]]+b[ys[:-1,None],xs[None,:-1]]
 mu=sums/(76*121);local=np.sqrt(np.maximum(0,squares/(76*121)-mu*mu)).mean();N=im.size;globalstd=math.sqrt(max(0,b[-1,-1]/N-(a[-1,-1]/N)**2))
 return globalstd,float(local)

def main():
 import math
 old={int(r['frame_id']):r for r in readcsv(DATA/'sequence6_adaptive_v2_full/adaptive_frame_decisions.csv')};agreement=[]
 for fid in range(3385):
  im=image(fid);ref=float(old[fid]['local_std']);r=reshape(im);s=integral(im)
  agreement.append(dict(frame=fid,saved_local=ref,reshape_local=r[1],integral_local=s[1],saved_raw=old[fid]['decision']=='raw',reshape_raw=r[1]>=10.5,integral_raw=s[1]>=10.5))
 writecsv(OUT/'decision_agreement.csv',agreement)
 # Freeze a single logical-CPU affinity for this process only; no system power
 # settings are changed. Record the actual OS setting rather than claim a clock lock.
 kernel=ctypes.windll.kernel32;kernel.GetCurrentProcess.restype=ctypes.c_void_p;kernel.SetProcessAffinityMask.argtypes=[ctypes.c_void_p,ctypes.c_size_t];kernel.SetProcessAffinityMask.restype=ctypes.c_int;proc=kernel.GetCurrentProcess();mask=1<<15;ok=bool(kernel.SetProcessAffinityMask(proc,mask))
 imgs=[image(i) for i in np.linspace(0,3384,64,dtype=int)]
 clahe=cv2.createCLAHE(2.,(8,8));orb,_=detector('ORB');samples=[]
 def adaptive(im,fn):
  _,local=fn(im);return im if local>=10.5 else clahe.apply(im)
 funcs={'decision_naive':naive,'decision_reshape':reshape,'decision_integral':integral,'fixed_clahe':clahe.apply,'adaptive_naive':lambda im:adaptive(im,naive),'adaptive_integral':lambda im:adaptive(im,integral),'orb_extraction':lambda im:feature(im,orb)}
 # Interleave methods deterministically by repetition to reduce order bias.
 rng=np.random.default_rng(20261105)
 for name,fn in funcs.items():
  for im in imgs[:16]:fn(im)
 for rep in range(16):
  for name in rng.permutation(list(funcs)):
   fn=funcs[name]
   for idx,im in enumerate(imgs):
    t=time.perf_counter_ns();fn(im);ns=time.perf_counter_ns()-t;samples.append(dict(method=name,rep=rep,image_sample=idx,milliseconds=ns/1e6))
 writecsv(OUT/'timing_samples.csv',samples)
 # Essential stage uses a fixed actual descriptor pair, precomputed outside timing.
 from run_vo import pose
 pair_images=[clahe.apply(image(i)) for i in [100,120]];aa,bb=[feature(im,orb) for im in pair_images];pa,pb=correspond(aa,bb,cv2.NORM_HAMMING,False)
 for k in range(16):pose(pa,pb,50,20261105)
 for rep in range(128):
  t=time.perf_counter_ns();pose(pa,pb,50,20261105);samples.append(dict(method='essential_and_recover_pose',rep=rep,image_sample=0,milliseconds=(time.perf_counter_ns()-t)/1e6))
 writecsv(OUT/'timing_samples.csv',samples);summary=[]
 for name in sorted(set(s['method'] for s in samples)):
  a=np.array([s['milliseconds'] for s in samples if s['method']==name]);summary.append(dict(method=name,n=len(a),median_ms=float(np.median(a)),q25_ms=float(np.quantile(a,.25)),q75_ms=float(np.quantile(a,.75)),iqr_ms=float(np.quantile(a,.75)-np.quantile(a,.25))))
 writecsv(OUT/'timing_summary.csv',summary)
 power=subprocess.run(['powercfg','/getactivescheme'],capture_output=True,text=True).stdout
 dump(OUT/'timing_environment.json',dict(python=sys.version,numpy=np.__version__,opencv=cv2.__version__,opencv_threads=cv2.getNumThreads(),single_cpu_affinity_requested=mask,affinity_applied=ok,power_scheme=power,frequency_locked=False,warmup_images_per_method=16,repetitions=16,images_per_rep=64,order='deterministically randomized',seed=20261105,io='outside timed sections',timing_scope='single-process CPU timing; OS scheduling and dynamic clock variation remain'))
 dump(OUT/'decision_agreement_summary.json',dict(n=3385,reshape_disagreements=sum(r['saved_raw']!=r['reshape_raw'] for r in agreement),integral_disagreements=sum(r['saved_raw']!=r['integral_raw'] for r in agreement),reshape_max_abs_difference=max(abs(r['saved_local']-r['reshape_local']) for r in agreement),integral_max_abs_difference=max(abs(r['saved_local']-r['integral_local']) for r in agreement)))
 print(summary,flush=True)
if __name__=='__main__':
 import math
 main()
