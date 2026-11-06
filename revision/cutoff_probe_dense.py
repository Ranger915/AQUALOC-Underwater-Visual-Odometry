from dense_reference import *
from run_vo import legacy,gate,prepared
from functools import lru_cache
import math,pandas as pd
CUTS=[50.,200.,1000.,10000.,1e9]
def poses(a,b):
 result={cut:dict(ne=0,nc=0,rho=0.,parallax=float('nan'),R=None,t=None,status='few_matches') for cut in CUTS}
 if len(a)<15:return result
 u=undistort(a);v=undistort(b);E,mask=cv2.findEssentialMat(u,v,K,method=cv2.RANSAC,prob=.999,threshold=1.5,maxIters=2000)
 if E is None or mask is None:return result
 ne=int(mask.sum())
 if ne<15:
  for r in result.values():r.update(ne=ne,status='insufficient_essential_inliers')
  return result
 for cut,r in result.items():
  best=None
  for off in range(0,len(E),3):
   if E[off:off+3].shape!=(3,3):continue
   out=cv2.recoverPose(E[off:off+3],u,v,K,distanceThresh=cut,mask=mask.copy());assert len(out)==5
   nc,R,t,pm=out[:4]
   if best is None or nc>best[0]:best=(nc,R,t.reshape(3),pm)
  if best is None:continue
  nc,R,t,pm=best;r.update(ne=ne,nc=int(nc),rho=nc/ne,R=R,t=t,status='geometry');valid=pm.ravel()>0
  if np.any(valid):
   ra=cv2.undistortPoints(u[valid].reshape(-1,1,2),K,None).reshape(-1,2);rb=cv2.undistortPoints(v[valid].reshape(-1,1,2),K,None).reshape(-1,2);ra=np.column_stack([ra,np.ones(len(ra))]);rb=np.column_stack([rb,np.ones(len(rb))]);ra/=np.linalg.norm(ra,axis=1)[:,None];rb/=np.linalg.norm(rb,axis=1)[:,None];rb=rb@R;r['parallax']=float(np.median(np.degrees(np.arccos(np.clip(np.sum(ra*rb,axis=1),-1,1)))))
 return result
def main():
 baseline=readcsv(OUT/'vo/ORB_fixed_rho0.06_quality_depth1e+09.csv');det,norm=detector('ORB');pairdir=CACHE/'cutoff_fixed_pairs';pairdir.mkdir(exist_ok=True);records=[];mismatches=[]
 @lru_cache(maxsize=128)
 def feat(fid):return feature(prepared(fid,'fixed'),det)
 @lru_cache(maxsize=28)
 def corners(fid):return legacy.flow_points(prepared(fid,'fixed'),1800)
 partial=R2/'short_gap_cutoff_probe.partial.csv';done=R2/'short_gap_cutoff_probe.csv'
 if done.exists():print('Dense cutoff probe already complete',flush=True);return
 if partial.exists():records=readcsv(partial)
 last=max([int(r['frame']) for r in records],default=0)
 grid=list(range(20,3381,20))
 for fid in [f for f in grid if f>last]:
  old=baseline[fid];anchor=int(old['attempted_anchor']);g=fid-anchor;file=pairdir/f'{anchor}_{fid}.npz'
  if file.exists():
   with np.load(file) as z:a,b,c,d=[z[k] for k in ['a','b','c','d']]
  else:
   a,b=correspond(feat(anchor),feat(fid),norm,False);c,d=legacy.flow_pairs(prepared(anchor,'fixed'),prepared(fid,'fixed'),corners(anchor),1.5);np.savez_compressed(file,a=a,b=b,c=c,d=d)
  candidates={'descriptor':poses(a,b),'flow':poses(c,d)}
  for cut in CUTS:
   valid=[]
   for source,rs in candidates.items():
    r=rs[cut];status=gate(r,.06)
    if status=='accepted':valid.append((source,r))
    row=dict(frame=fid,anchor=anchor,gap=g,cutoff=cut,source=source,status=status,accepted=int(status=='accepted'),ne=r['ne'],nc=r['nc'],rho=r['rho'],parallax_deg=r['parallax'],rotation_error_deg=degrees((ROTS[fid].T@ROTS[anchor]).T@r['R']) if r['R'] is not None else float('nan'),direction_error_deg=angle(ROTS[fid].T@(C[anchor]-C[fid]),r['t']) if r['t'] is not None else float('nan'),reference_step_m=float(np.linalg.norm(C[fid]-C[anchor])),reference_clean=int(CLEAN[fid] and CLEAN[anchor]));records.append(row)
   if valid:
    source,r=max(valid,key=lambda item:(min(item[1]['nc'],200)*min(item[1]['rho'],1)*min(item[1]['parallax'],2),item[0]=='descriptor'));row=next(x for x in reversed(records) if int(x['frame'])==fid and float(x['cutoff'])==cut and x['source']==source).copy();row['source']='selected';row['selected_branch']=source
   else:
    row=records[-1].copy();row.update(source='selected',selected_branch='none',accepted=0,status='rejected',rotation_error_deg=float('nan'),direction_error_deg=float('nan'))
   # Keep the same columns for resumed and fresh rows.
   for x in records[-2:]:x.setdefault('selected_branch','')
   records.append(row)
   if cut==1e9 and (int(row['accepted'])!=int(old['fresh']) or (int(old['fresh']) and row['selected_branch']!=old['source'])):mismatches.append(dict(frame=fid,old_fresh=old['fresh'],new_fresh=row['accepted'],old_source=old['source'],new_source=row['selected_branch']))
  if fid%400==0:writecsv(partial,records);print('Cutoff short-gap probe',grid.index(fid)+1,'of',len(grid),flush=True)
 for r in records:r.setdefault('selected_branch','')
 writecsv(done,records);d=pd.DataFrame(records);d['cutoff']=d.cutoff.astype(float);d['accepted']=d.accepted.astype(int);selected=d[(d.source=='selected')&(d.accepted==1)];lowset=set(selected[selected.cutoff==50].frame.astype(int));highset=set(selected[selected.cutoff==1e9].frame.astype(int));summary=[]
 for clean in [False,True]:
  for cut in CUTS:
   a=selected[selected.cutoff==cut]
   if clean:a=a[a.reference_clean.astype(int)==1]
   for label,frameids in [('all_accepted',set(a.frame.astype(int))),('both',lowset&highset),('high_only',highset-lowset),('low_only',lowset-highset)]:
    b=a[a.frame.astype(int).isin(frameids)];summary.append(dict(mask='clean_endpoints' if clean else 'all_reference',cutoff=cut,group=label,n=len(b),median_rotation_error_deg=float(b.rotation_error_deg.astype(float).median()),p90_rotation_error_deg=float(b.rotation_error_deg.astype(float).quantile(.9)),median_direction_error_deg=float(b.direction_error_deg.astype(float).median()),p90_direction_error_deg=float(b.direction_error_deg.astype(float).quantile(.9)),median_rho=float(b.rho.astype(float).median()),fraction_flow=float(np.mean(b.selected_branch=='flow')) if len(b) else np.nan))
 writecsv(R2/'same_anchor_cutoff_probe_summary.csv',summary);dump(R2/'same_anchor_cutoff_probe_validation.json',dict(frames=len(grid),sampled_current_ids=grid,anchor_schedule='every twentieth current image, fixed to completed ORB fixed high-depth attempted anchors; short-gap gate probe, not reintegrated VO at intermediate cutoffs',essential_estimate='one essential estimate per source/pair reused at all five cutoffs',baseline_acceptance_or_source_mismatches=len(mismatches),mismatches=mismatches,geometry='linear center interpolation and quaternion SLERP; small baseline directions remain uncertain',pair_cache=str(pairdir)));print('Completed fixed-grid short-gap cutoff probe',len(records),'rows',flush=True)
if __name__=='__main__':main()
