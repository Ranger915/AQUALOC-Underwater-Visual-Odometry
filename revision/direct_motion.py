from common import *
from run_vo import pose,gate,legacy,DECISIONS
from fast_match import fast_correspond
from functools import lru_cache
import math
REF=refposes();IDS=sorted(REF)
def angle(a,b):return float(np.degrees(np.arccos(np.clip(a@b/(np.linalg.norm(a)*np.linalg.norm(b)),-1,1))))
def main():
 records=[]
 for alg in ['ORB','AKAZE','SIFT']:
  norm=cv2.NORM_L2 if alg=='SIFT' else cv2.NORM_HAMMING
  for prep in ['raw','fixed','adaptive']:
   def mode(fid):return 'raw' if prep=='raw' or (prep=='adaptive' and DECISIONS[fid]=='raw') else 'c2t8'
   @lru_cache(maxsize=3)
   def feat(fid):
    p=CACHE/'reference_features'/alg/mode(fid)/f'{fid}.npz'
    if p.exists():
     with np.load(p) as z:return z['p'],z['d']
    im=img(fid);d,_=detector(alg);return feature(im,d)
   @lru_cache(maxsize=3)
   def img(fid):
    im=image(fid);return im if mode(fid)=='raw' else cv2.createCLAHE(2.,(8,8)).apply(im)
   for i,j in zip(IDS[:-1],IDS[1:]):
    a,b=fast_correspond(feat(i),feat(j),norm,False);c,d=legacy.flow_pairs(img(i),img(j),legacy.flow_points(img(i),1800),1.5)
    for cutoff in [50.,1e9]:
     valid=[]
     for si,(source,x,y) in enumerate([('descriptor',a,b),('flow',c,d)]):
      r=pose(x,y,cutoff,(20261105+j*31+i*7+si)%2147483647);status=gate(r,.06)
      if status=='accepted':valid.append((min(r['nc'],200)*min(r['rho'],1)*min(r['parallax'],2),source,r))
      if r['R'] is None:continue
      Ri,Ci=REF[i];Rj,Cj=REF[j];refR=Rj.T@Ri;reft=Rj.T@(Ci-Cj)
      records.append(dict(detector=alg,preprocessing=prep,i=i,j=j,cutoff=cutoff,source=source,status=status,accepted=int(status=='accepted'),rotation_error_deg=degrees(refR.T@r['R']),translation_direction_error_deg=angle(reft,r['t']),ne=r['ne'],nc=r['nc'],rho=r['rho'],parallax_deg=r['parallax']))
     if valid:
      _,source,r=max(valid,key=lambda t:(t[0],t[1]=='descriptor'));Ri,Ci=REF[i];Rj,Cj=REF[j]
      records.append(dict(detector=alg,preprocessing=prep,i=i,j=j,cutoff=cutoff,source='selected',status='accepted',accepted=1,rotation_error_deg=degrees((Rj.T@Ri).T@r['R']),translation_direction_error_deg=angle(Rj.T@(Ci-Cj),r['t']),ne=r['ne'],nc=r['nc'],rho=r['rho'],parallax_deg=r['parallax']))
   writecsv(OUT/'direct_two_view_motion.csv',records);print('DIRECT',alg,prep,flush=True)
 summary=[]
 for alg in ['ORB','AKAZE','SIFT']:
  for prep in ['raw','fixed','adaptive']:
   for cut in [50.,1e9]:
    rr=[r for r in records if r['detector']==alg and r['preprocessing']==prep and r['cutoff']==cut and r['source']=='selected']
    summary.append(dict(detector=alg,preprocessing=prep,cutoff=cut,attempts=169,accepted=len(rr),median_rotation_error_deg=float(np.median([r['rotation_error_deg'] for r in rr])) if rr else float('nan'),median_translation_direction_error_deg=float(np.median([r['translation_direction_error_deg'] for r in rr])) if rr else float('nan')))
 writecsv(OUT/'direct_two_view_summary.csv',summary)
if __name__=='__main__':main()
