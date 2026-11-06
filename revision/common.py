from pathlib import Path
import sys,os,json,csv,hashlib,time,platform,subprocess
import numpy as np
# Load the installed ABI-stable OpenCV extension without importing the old
# environment's Python-version-specific numpy/pandas packages.
try:
 import cv2
except ModuleNotFoundError:
 sys.path.append(os.environ.get('AQUALOC_OPENCV_SITE',r'C:\Users\Raaghav\aqualoc-venv\Lib\site-packages'))
 import cv2
cv2.setNumThreads(1)
cv2.setRNGSeed(20261105)
ROOT=Path(__file__).resolve().parents[1]
DATA=Path(os.environ.get('AQUALOC_ROOT',r'C:\Users\Raaghav\Downloads\archaeo_sequence_6_raw_data'))
OUT=ROOT/'revision_results';OUT.mkdir(exist_ok=True)
CACHE=ROOT/'revision_cache';CACHE.mkdir(exist_ok=True)
IMAGES=DATA/'raw_data/images_sequence_6'
K=np.array([[543.3327734182214,0,489.02536042247897],[0,542.398772982566,305.38727712002805],[0,0,1.]])
DIST=np.array([-.1255945656257394,.053221287232781606,9.94070021080493e-5,9.550660927242349e-5])
def readcsv(p):
 with open(p,encoding='utf-8-sig',newline='') as f:return list(csv.DictReader(f))
def writecsv(p,rr):
 rr=list(rr)
 if not rr:return
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
 with open(p,'w',encoding='utf-8',newline='') as f:
  w=csv.DictWriter(f,fieldnames=list(rr[0]));w.writeheader();w.writerows(rr)
def dump(p,x):Path(p).write_text(json.dumps(x,indent=2,allow_nan=True,default=str),encoding='utf-8')
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def image(fid):
 x=cv2.imread(str(IMAGES/f'frame{fid:06d}.png'),0)
 if x is None:raise RuntimeError(f'Missing frame {fid}')
 return x
def quatmat(q):
 x,y,z,w=np.asarray(q,float)/np.linalg.norm(q)
 return np.array([[1-2*(y*y+z*z),2*(x*y-z*w),2*(x*z+y*w)],
 [2*(x*y+z*w),1-2*(x*x+z*z),2*(y*z-x*w)],
 [2*(x*z-y*w),2*(y*z+x*w),1-2*(x*x+y*y)]])
def refposes(order='xyzw',direction='twc',file=None):
 a=np.loadtxt(file or DATA/'colmap_traj_sequence_6.txt');out={}
 for r in a:
  q=r[4:8] if order=='xyzw' else r[[5,6,7,4]];R=quatmat(q);t=r[1:4]
  if direction=='tcw':t=-R.T@t;R=R.T
  out[int(r[0])]=(R,t)
 return out
def fundamental(a,b):
 Ra,Ca=a;Rb,Cb=b;R=Rb.T@Ra;t=Rb.T@(Ca-Cb)
 x,y,z=t;sk=np.array([[0,-z,y],[z,0,-x],[-y,x,0]])
 Ki=np.linalg.inv(K);return Ki.T@sk@R@Ki
def sampson(a,b,F):
 x=np.column_stack([a,np.ones(len(a))]);y=np.column_stack([b,np.ones(len(b))]);Fx=x@F.T;Fty=y@F
 den=(Fx[:,:2]**2).sum(1)+(Fty[:,:2]**2).sum(1)
 return np.sqrt(np.divide(np.sum(y*Fx,axis=1)**2,den,out=np.full(len(x),np.inf),where=den>1e-24))
def score(y,c):
 y=np.asarray(y,int);c=np.asarray(c,float);pred=c>=8
 tp=int(np.sum((y==1)&pred));fp=int(np.sum((y==0)&pred));fn=int(np.sum((y==1)&~pred))
 precision=tp/max(1,tp+fp);recall=tp/max(1,tp+fn);f1=2*tp/max(1,2*tp+fp+fn)
 o=np.argsort(-c,kind='stable');ss=c[o];yy=y[o];idx=np.r_[np.flatnonzero(ss[:-1]!=ss[1:]),len(ss)-1]
 acc=np.cumsum(yy)[idx];pr=acc/(idx+1);dp=np.diff(np.r_[0,acc]);ap=float(np.sum(dp*pr)/sum(y)) if sum(y) else float('nan')
 return dict(n=len(y),positives=int(sum(y)),tp=tp,fp=fp,fn=fn,precision=precision,recall=recall,f1=f1,ap_count=ap)
def detector(name,low=False):
 if name=='ORB':return cv2.ORB_create(nfeatures=1500,fastThreshold=5 if low else 20),cv2.NORM_HAMMING
 if name=='AKAZE':return cv2.AKAZE_create(threshold=.0001 if low else .001),cv2.NORM_HAMMING
 if name=='SIFT':return cv2.SIFT_create(nfeatures=1500,contrastThreshold=.01 if low else .04),cv2.NORM_L2
 raise ValueError(name)
def feature(img,det):
 kp,d=det.detectAndCompute(img,None)
 if d is None:return np.empty((0,2),np.float32),np.empty((0,0),np.uint8)
 keep=np.argsort([-k.response for k in kp],kind='stable')[:1500]
 return np.array([kp[int(i)].pt for i in keep],np.float32),d[keep]
def correspond(a,b,norm,mutual=True):
 pa,da=a;pb,db=b
 if min(len(pa),len(pb))<2:return np.empty((0,2),np.float32),np.empty((0,2),np.float32)
 m=cv2.BFMatcher(norm)
 f=[x for r in m.knnMatch(da,db,k=2) if len(r)==2 for x,y in [r] if x.distance<.75*y.distance]
 if mutual:
  back={(x.trainIdx,x.queryIdx) for r in m.knnMatch(db,da,k=2) if len(r)==2 for x,y in [r] if x.distance<.75*y.distance}
  f=[x for x in f if (x.queryIdx,x.trainIdx) in back]
 return np.array([pa[x.queryIdx] for x in f],np.float32).reshape(-1,2),np.array([pb[x.trainIdx] for x in f],np.float32).reshape(-1,2)
def undistort(x):return cv2.undistortPoints(np.asarray(x,np.float32).reshape(-1,1,2),K,DIST,P=K).reshape(-1,2)
def fit(p,q):
 u=p-p.mean(0);v=q-q.mean(0);U,D,Vt=np.linalg.svd(v.T@u/len(p));Z=np.diag([1,1,np.linalg.det(U@Vt)]);R=U@Z@Vt;s=np.trace(np.diag(D)@Z)/np.mean(np.sum(u*u,axis=1));t=q.mean(0)-s*R@p.mean(0)
 return s,R,t
def degrees(R):return float(np.degrees(np.arccos(np.clip((np.trace(R)-1)/2,-1,1))))
