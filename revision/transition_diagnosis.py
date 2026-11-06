from dense_reference import *
import pandas as pd
def diagnose_original():
 src=readcsv(DATA/'paper_research_draft/vo_full_sequence_v3/vo_sequence6_orb_clahe_v3.csv');by={int(r['frame']):r for r in src};split=json.loads((DATA/'paper_research_draft/full_sequence_trajectory_evaluation/summary.json').read_text());train=np.array(split['train_frames']);test=np.array(split['test_frames']);positions=np.array([[float(r[k]) for k in ['x','y','z']] for r in src]);rot=np.array([quatmat([float(r[k]) for k in ['qx','qy','qz','qw']]) for r in src]);s,A,t=fit(positions[train],C[train]);rows=[];lo=2120;hi=2360;net=(s*A@(positions[hi]-positions[lo])-(C[hi]-C[lo]));netnorm=np.linalg.norm(net);axis=net/max(netnorm,1e-20)
 for row in src:
  frame=int(row['frame'])
  if frame>3380 or not int(row['tracking_success']):continue
  anchor=frame-int(row['estimated_anchor_gap']);refR=ROTS[frame].T@ROTS[anchor];estR=rot[frame].T@rot[anchor];trel=rot[frame].T@(positions[anchor]-positions[frame]);tref=ROTS[frame].T@(C[anchor]-C[frame]);delta=s*A@(positions[frame]-positions[anchor])-(C[frame]-C[anchor]);inside=anchor>=lo and frame<=hi
  rows.append(dict(frame=frame,anchor=anchor,gap=frame-anchor,source=row['match_source'],rotation_error_deg=degrees(refR.T@estR),translation_direction_error_deg=angle(tref,trel),reference_step_m=float(np.linalg.norm(tref)),scaled_estimated_step_m=float(s*np.linalg.norm(trel)),increment_residual_x=float(delta[0]),increment_residual_y=float(delta[1]),increment_residual_z=float(delta[2]),increment_residual_norm=float(np.linalg.norm(delta)),projection_on_boundary_net=float(delta@axis),in_boundary_gap=int(inside),suspect_reference_edge=int(not (CLEAN[anchor] and CLEAN[frame]))))
 writecsv(R2/'original_per_transition_errors.csv',rows);gap=[r for r in rows if r['in_boundary_gap']];summed=np.array([[r['increment_residual_x'],r['increment_residual_y'],r['increment_residual_z']] for r in gap]).sum(0);assert np.allclose(summed,net,atol=1e-8),(summed,net);rank=sorted(gap,key=lambda r:r['projection_on_boundary_net'],reverse=True);sizes=sorted(gap,key=lambda r:r['increment_residual_norm'],reverse=True)
 sensitivity=[]
 for clean in [False,True]:
  tr=train[CLEAN[train]] if clean else train;te=test[CLEAN[test]] if clean else test;sc,al,tt=fit(positions[tr],C[tr]);e=np.linalg.norm(sc*(positions[te]@al.T)+tt-C[te],axis=1);sensitivity.append(dict(mask='reference_outlier_edges_excluded' if clean else 'all_reference',fit_n=len(tr),test_n=len(te),rmse_m=float(np.sqrt(np.mean(e*e))),rmse_fraction_path=float(np.sqrt(np.mean(e*e))/PATH),rmse_fraction_diagonal=float(np.sqrt(np.mean(e*e))/DIAG)))
 writecsv(R2/'original_split_outlier_sensitivity.csv',sensitivity)
 dump(R2/'boundary_transition_summary.json',dict(fit_last=lo,test_first=hi,boundary_accepted_steps=len(gap),boundary_net_residual_change=net.tolist(),boundary_net_norm=float(netnorm),sum_increment_check_max_abs=float(np.max(np.abs(summed-net))),largest_five_directional_contributions=rank[:5],largest_five_increment_norms=sizes[:5],top_one_projection_fraction=float(rank[0]['projection_on_boundary_net']/netnorm),top_five_projection_fraction=float(sum(r['projection_on_boundary_net'] for r in rank[:5])/netnorm),top_ten_projection_fraction=float(sum(r['projection_on_boundary_net'] for r in rank[:10])/netnorm),interpretation='Descriptive decomposition of fixed-fit residual change; interpolated reference and accumulated orientation/scale affect increments; not causal proof of a wrong match'))
def cutoff_groups():
 d=pd.read_csv(OUT/'direct_two_view_motion.csv');rows=[]
 for alg in ['ORB','AKAZE','SIFT']:
  for prep in ['raw','fixed','adaptive']:
   for source in ['selected','descriptor','flow']:
    a=d[(d.detector==alg)&(d.preprocessing==prep)&(d.source==source)];lo=a[(a.cutoff==50)&(a.accepted==1)];high=a[(a.cutoff==1e9)&(a.accepted==1)];lowset=set(zip(lo.i,lo.j));highset=set(zip(high.i,high.j))
    for label,which in [('both',lowset&highset),('high_only',highset-lowset),('low_only',lowset-highset)]:
     for cut in ([50,1e9] if label=='both' else ([1e9] if label=='high_only' else [50])):
      b=a[(a.cutoff==cut)&(a.accepted==1)];b=b[[pair in which for pair in zip(b.i,b.j)]]
      rows.append(dict(detector=alg,preprocessing=prep,source=source,accepted_group=label,cutoff=cut,n=len(b),median_rotation_error_deg=float(b.rotation_error_deg.median()),p90_rotation_error_deg=float(b.rotation_error_deg.quantile(.9)),median_direction_error_deg=float(b.translation_direction_error_deg.median()),p90_direction_error_deg=float(b.translation_direction_error_deg.quantile(.9))))
 writecsv(R2/'saved_cutoff_acceptance_groups.csv',rows)
def branch_analysis():
 mods={};rows=[];pairrows=[]
 for alg in ['ORB','AKAZE','SIFT']:
  for prep in ['raw','fixed','adaptive']:
   name=f'{alg}_{prep}_rho0.06_quality_depth1e+09';a=pd.read_csv(OUT/'vo'/f'{name}.csv');a=a[(a.fresh==1)&(a.frame<=3380)];mods[(alg,prep)]=a.set_index('frame');rows.append(dict(detector=alg,preprocessing=prep,accepted=len(a),selected_flow=int(sum(a.source=='flow')),flow_fraction=float(np.mean(a.source=='flow')),both_valid_fraction=float(np.mean((a.descriptor_valid==1)&(a.flow_valid==1)))))
 for prep in ['raw','fixed','adaptive']:
  a=mods[('AKAZE',prep)];b=mods[('SIFT',prep)];m=a.join(b,lsuffix='_a',rsuffix='_b',how='inner');same=(m.source_a=='flow')&(m.source_b=='flow')&(m.attempted_anchor_a==m.attempted_anchor_b);pairrows.append(dict(preprocessing=prep,common_fresh_frames=len(m),same_anchor_shared_flow=int(same.sum()),fraction_common_same_anchor_shared_flow=float(same.mean())))
 writecsv(R2/'selected_source_fractions.csv',rows);writecsv(R2/'shared_flow_branch.csv',pairrows)
if __name__=='__main__':
 save_reference_audit();diagnose_original();cutoff_groups();branch_analysis();print('Completed saved-transition and cutoff-group diagnosis',flush=True)
