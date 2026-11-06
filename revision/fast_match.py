from common import *
import ctypes
native=Path(__file__).parent/'exact_match_avx2.dll'
if 'AVX2' not in cv2.getCPUFeaturesLine():native=Path(__file__).parent/'exact_match.dll'
try:lib=ctypes.CDLL(str(native if native.exists() else Path(__file__).parent/'exact_match.dll'))
except OSError:lib=None
if lib is not None:
 lib.match_exact.argtypes=[ctypes.c_void_p,ctypes.c_void_p,ctypes.c_int,ctypes.c_int,ctypes.c_int,ctypes.c_int,ctypes.c_int,ctypes.c_void_p,ctypes.c_void_p]
 lib.match_exact.restype=ctypes.c_int
def exact_indices(a,b,norm,mutual=True):
 if min(len(a),len(b))<2:return np.array([],np.int32),np.array([],np.int32)
 a=np.ascontiguousarray(a);b=np.ascontiguousarray(b);i=np.empty(len(a),np.int32);j=np.empty(len(a),np.int32)
 n=lib.match_exact(a.ctypes.data,b.ctypes.data,len(a),len(b),a.shape[1],0 if norm==cv2.NORM_HAMMING else 1,int(mutual),i.ctypes.data,j.ctypes.data)
 return i[:n],j[:n]
def fast_correspond(a,b,norm,mutual=True):
 if lib is None:return correspond(a,b,norm,mutual)
 pa,da=a;pb,db=b;i,j=exact_indices(da,db,norm,mutual)
 return pa[i],pb[j]
if __name__=='__main__':
 rng=np.random.default_rng(20261105);checks=[]
 for alg in ['ORB','AKAZE','SIFT']:
  for mode in ['raw','c2t8']:
   folder=CACHE/'reference_features'/alg/mode
   paths=sorted(folder.glob('*.npz'))
   if len(paths)<2:continue
   norm=cv2.NORM_L2 if alg=='SIFT' else cv2.NORM_HAMMING
   for k in range(30):
    x,y=rng.choice(paths,2,replace=False)
    with np.load(x) as z:a=(z['p'],z['d'])
    with np.load(y) as z:b=(z['p'],z['d'])
    aa,bb=correspond(a,b,norm);cc,dd=fast_correspond(a,b,norm)
    same=np.array_equal(aa,cc) and np.array_equal(bb,dd);checks.append(dict(detector=alg,mode=mode,pair=k,identical=same))
    if not same:raise RuntimeError(('Matcher mismatch',alg,mode,k))
 a=np.load(CACHE/'reference_features/ORB/c2t8/0.npz');b=np.load(CACHE/'reference_features/ORB/c2t8/20.npz');t=time.perf_counter()
 for k in range(100):fast_correspond((a['p'],a['d']),(b['p'],b['d']),cv2.NORM_HAMMING)
 dump(OUT/'accelerated_matcher_validation.json',dict(checks=checks,all_identical=all(r['identical'] for r in checks),orb_pair_ms=(time.perf_counter()-t)*10,method='exhaustive exact distances once per pair, simultaneous bidirectional top-two minima; strict same ratio tests; no approximate neighbor search'))
 print('PASS',len(checks),'pairs; ms',(time.perf_counter()-t)*10)
