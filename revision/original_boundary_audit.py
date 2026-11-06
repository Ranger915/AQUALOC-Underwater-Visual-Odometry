from common import *
source=readcsv(DATA/'paper_research_draft/vo_full_sequence_v3/vo_sequence6_orb_clahe_v3.csv')
split=json.loads((DATA/'paper_research_draft/full_sequence_trajectory_evaluation/summary.json').read_text());ref=refposes();by={int(r['frame']):r for r in source}
train=split['train_frames'];test=split['test_frames'];p=np.array([[float(by[i][k]) for k in ['x','y','z']] for i in train]);q=np.array([ref[i][1] for i in train]);s,R,t=fit(p,q)
reset=0;unknown=0;cum={}
for r in source:
 if int(r['anchor_reset']):reset+=1;unknown+=int(r['estimated_anchor_gap'])
 cum[int(r['frame'])]=(reset,unknown)
last=train[-1];rr=[]
for i in sorted(ref):
 r=by[i];e=s*R@np.array([float(r[k]) for k in ['x','y','z']])+t-ref[i][1];a,b=cum[i];aa,bb=cum[last]
 rr.append(dict(frame=i,fresh=int(r['tracking_success']),split='alignment' if i in train else ('heldout' if i in test else 'stale_diagnostic'),error=float(np.linalg.norm(e)),residual_x=e[0],residual_y=e[1],residual_z=e[2],resets_since_last_alignment=a-aa,unknown_frames_since_last_alignment=b-bb,frames_since_last_alignment=i-last))
writecsv(OUT/'original_boundary_diagnostic.csv',rr)
te=[r for r in rr if r['split']=='heldout'];prev=[r for r in rr if r['frame'] in train[-6:]];following=te[:6]
def corr(x,y):
 return float(np.corrcoef(x,y)[0,1]) if np.std(x)>0 and np.std(y)>0 else float('nan')
summary=dict(original_train_last=last,original_test_first=test[0],reference_frame_gap=test[0]-last,nominal_seconds_gap=(test[0]-last)/20,original_rmse=float(np.sqrt(np.mean([r['error']**2 for r in te]))),first_test=te[0],last_six_alignment_mean_error=float(np.mean([r['error'] for r in prev])),first_six_test_mean_error=float(np.mean([r['error'] for r in following])),test_error_vs_resets_pearson_descriptive=corr([r['error'] for r in te],[r['resets_since_last_alignment'] for r in te]),test_error_vs_unknown_frames_pearson_descriptive=corr([r['error'] for r in te],[r['unknown_frames_since_last_alignment'] for r in te]),test_error_vs_elapsed_frames_pearson_descriptive=corr([r['error'] for r in te],[r['frames_since_last_alignment'] for r in te]),interpretation='Sparse accepted-pose plot leaves a 240-frame unobserved interval; all-reference carried-position residuals are diagnostic only; associations do not identify lost motion as a causal explanation')
dump(OUT/'original_boundary_summary.json',summary);print(json.dumps(summary,indent=2))
