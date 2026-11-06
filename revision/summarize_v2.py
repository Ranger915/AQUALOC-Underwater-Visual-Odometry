from dense_reference import *
import pandas as pd
def main():
 seg=pd.read_csv(R2/'dense_segments.csv');summary=[]
 for (group,mask,f,name),a in seg.groupby(['group','mask','fit_frames','configuration']):
  rmse=float(np.sqrt(a.sse.sum()/a.test_n.sum()));summary.append(dict(group=group,mask=mask,fit_frames=int(f),configuration=name,segments=len(a),test_positions=int(a.test_n.sum()),rmse_m=rmse,percent_path=100*rmse/PATH,percent_diagonal=100*rmse/DIAG,median_window_rotation_error_deg=float(a.rotation_error_median_deg.median()),median_window_direction_error_deg=float(a.direction_error_median_deg.median()),median_window_relative_translation_rmse_m=float(a.relative_translation_rmse_m.median())))
 writecsv(R2/'dense_summary.csv',summary)
 labels=pd.read_csv(OUT/'geometric_label_sensitivity.csv');ranks=[]
 for (dist,view),a in labels[labels['mode']=='fixed'].groupby(['max_distance_units','max_view_angle_deg']):
  a=a.sort_values('ap_count',ascending=False);ranks.append(dict(distance=dist,view=view,ranking=' > '.join(a.detector),top_ap=float(a.ap_count.iloc[0]),last_ap=float(a.ap_count.iloc[-1])))
 writecsv(R2/'alternative_label_rankings.csv',ranks)
 primary=[r for r in summary if r['fit_frames']==800 and r['mask']=='all_reference' and r['group'] in ['high_matrix_common','source_policies_depth1e9','source_policies_depth50']]
 dump(R2/'key_results.json',dict(dense_primary=primary,label_rankings=ranks,boundary=json.loads((R2/'boundary_transition_summary.json').read_text()),reference=json.loads((R2/'reference_interpolation.json').read_text())))
 print(json.dumps(primary,indent=2),flush=True)
if __name__=='__main__':main()
