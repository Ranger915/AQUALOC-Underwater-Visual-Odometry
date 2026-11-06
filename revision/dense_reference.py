from common import *
R2=ROOT/'revision2_results';R2.mkdir(exist_ok=True)
REFERENCE=Path(os.environ.get('AQUALOC_REFERENCE',str(DATA/'colmap_traj_sequence_6.txt')))
REF=refposes(file=REFERENCE);IDS=np.array(sorted(REF));CENTERS=np.array([REF[i][1] for i in IDS]);RAW=np.loadtxt(REFERENCE);QUAT=RAW[:,4:8];QUAT/=np.linalg.norm(QUAT,axis=1)[:,None]
FRAMES=np.arange(IDS[-1]+1);C=np.column_stack([np.interp(FRAMES,IDS,CENTERS[:,k]) for k in range(3)]);ROTS=[]
for frame in FRAMES:
 idx=max(0,min(np.searchsorted(IDS,frame,side='right')-1,len(IDS)-2));w=max(0,(frame-IDS[idx])/(IDS[idx+1]-IDS[idx]));a=QUAT[idx];b=QUAT[idx+1].copy();dot=float(a@b)
 if dot<0:b=-b;dot=-dot
 theta=np.arccos(np.clip(dot,-1,1));q=(1-w)*a+w*b if theta<1e-8 else (np.sin((1-w)*theta)*a+np.sin(w*theta)*b)/np.sin(theta);ROTS.append(quatmat(q))
ROTS=np.array(ROTS);STEPS=np.linalg.norm(np.diff(CENTERS,axis=0),axis=1);PATH=float(STEPS.sum());EXTENT=np.ptp(CENTERS,axis=0);DIAG=float(np.linalg.norm(EXTENT));MED=float(np.median(STEPS));MAD=float(np.median(np.abs(STEPS-MED)));THRESH=MED+6*1.4826*MAD;BAD=STEPS>THRESH;CLEAN=np.ones(len(FRAMES),bool)
for a,b in zip(IDS[:-1][BAD],IDS[1:][BAD]):CLEAN[a:b+1]=False
def angle(a,b):
 na,nb=np.linalg.norm(a),np.linalg.norm(b)
 return float(np.degrees(np.arccos(np.clip(a@b/(na*nb),-1,1)))) if min(na,nb)>1e-10 else float('nan')
def model(path):
 rows=readcsv(path);n=len(FRAMES);p=np.array([[float(r[f'p{k}']) for k in range(3)] for r in rows[:n]]);rot=np.array([[[float(r[f'R{i}{j}']) for j in range(3)] for i in range(3)] for r in rows[:n]]);fresh=np.array([int(r['fresh']) for r in rows[:n]],bool)
 return dict(name=path.stem,rows=rows[:n],p=p,R=rot,fresh=fresh,path=path)
def save_reference_audit():
 rr=[]
 for k,(a,b,d) in enumerate(zip(IDS[:-1],IDS[1:],STEPS)):
  reversal=angle(CENTERS[k]-CENTERS[k-1],CENTERS[k+1]-CENTERS[k]) if k else float('nan');rr.append(dict(start=int(a),end=int(b),step_m=float(d),robust_outlier=int(BAD[k]),turn_at_start_deg=reversal))
 writecsv(R2/'reference_step_outliers.csv',rr)
 dump(R2/'reference_interpolation.json',dict(position='piecewise linear',rotation='shortest-arc quaternion SLERP',last_frame=int(IDS[-1]),extrapolation=False,path_m=PATH,extent_m=EXTENT.tolist(),diagonal_m=DIAG,median_reference_step_m=MED,scaled_MAD_m=1.4826*MAD,threshold_m=THRESH,flagged_edges=[r for r in rr if r['robust_outlier']],excluded_frame_count=int(sum(~CLEAN)),primary='all supplied reference poses; robust exclusion only a sensitivity',publisher_scale_source='https://arxiv.org/html/1910.14532 section 4: pressure-scaled metric positions; table 2 #06 length 31.8 m',interpolation_warning='Denser comparisons do not add independent ground truth and cannot resolve motion inside the one-second reference interval'))
if __name__=='__main__':save_reference_audit();print('Reference outliers',int(sum(BAD)),'excluded frames',int(sum(~CLEAN)),flush=True)
