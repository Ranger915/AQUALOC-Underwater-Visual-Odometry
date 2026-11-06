from pathlib import Path
import json,csv,hashlib
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
DATA=Path(r'C:\Users\Raaghav\Downloads\archaeo_sequence_6_raw_data')
OUT=ROOT/'IEEE_AQUALOC'
def rows(p):
 with open(p,encoding='utf-8-sig',newline='') as f:return list(csv.DictReader(f))
v=rows(DATA/'paper_research_draft/vo_full_sequence_v3/vo_sequence6_orb_clahe_v3.csv')
byid={int(r['frame']):r for r in v}
ref=np.loadtxt(DATA/'colmap_traj_sequence_6.txt')
refmap={int(r[0]):r[1:4] for r in ref}
s=json.loads((OUT/'evidence/summary.json').read_text())
train=s['train_frames'];test=s['test_frames']
checks={}
def chk(name,condition,details):
 checks[name]={'pass':bool(condition),'details':details}
p=np.array([[float(byid[i][k]) for k in ['x','y','z']] for i in train]);q=np.array([refmap[i] for i in train])
u=p-p.mean(0);w=q-q.mean(0);H=w.T@u/len(p)
U,D,Vt=np.linalg.svd(H);S=np.diag([1,1,np.linalg.det(U@Vt)])
R=U@S@Vt;scale=np.trace(np.diag(D)@S)/np.mean(np.sum(u*u,axis=1));t=q.mean(0)-scale*R@p.mean(0)
pt=np.array([[float(byid[i][k]) for k in ['x','y','z']] for i in test]);qt=np.array([refmap[i] for i in test]);e=np.linalg.norm(qt-(scale*(R@pt.T).T+t),axis=1)
rmse=float(np.sqrt(np.mean(e*e)))
chk('full_holdout_rmse',abs(rmse-s['heldout_error_colmap_units']['rmse'])<1e-9,rmse)
chk('full_alignment_scale',abs(scale-s['sim3_scale'])<1e-12,float(scale))
chk('chronological_split',max(train)<min(test) and not(set(train)&set(test)),{'train_n':len(train),'test_n':len(test)})
chk('fresh_only',all(int(byid[i]['tracking_success'])==1 for i in train+test),'All 60 evaluated poses are fresh')
nd=sum(int(r['descriptor_valid']) for r in v);nf=sum(int(r['flow_valid']) for r in v);nb=sum(int(r['both_valid']) for r in v);na=sum(int(r['tracking_success']) for r in v)
chk('source_union',nd+nf-nb==na==1188,{'descriptor':nd,'flow':nf,'both':nb,'accepted':na})
# An independent known-transform construction checks covariance order and reflection handling.
rng=np.random.default_rng(20261105);p0=rng.normal(size=(20,3));ang=.7;R0=np.array([[np.cos(ang),-np.sin(ang),0],[np.sin(ang),np.cos(ang),0],[0,0,1]])
q0=2.3*(R0@p0.T).T+np.array([3,-2,1]);a=p0-p0.mean(0);b=q0-q0.mean(0);U,D,Vt=np.linalg.svd(b.T@a/20);Z=np.diag([1,1,np.linalg.det(U@Vt)]);rr=U@Z@Vt;ss=np.trace(np.diag(D)@Z)/np.mean(np.sum(a*a,axis=1));tt=q0.mean(0)-ss*rr@p0.mean(0)
chk('sim3_known_transform',np.max(np.abs(rr-R0))<1e-12 and abs(ss-2.3)<1e-12 and np.max(np.abs(tt-[3,-2,1]))<1e-12,'Recovered known positive scale, proper rotation and translation')
assert all(c['pass'] for c in checks.values()),checks
(OUT/'evidence/independent_checks.json').write_text(json.dumps(checks,indent=2),encoding='utf-8')
print(json.dumps(checks,indent=2))
