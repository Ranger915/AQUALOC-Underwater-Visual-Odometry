from dense_reference import *
sys.path.insert(0,str(ROOT/'revision_deps'))
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd,shutil
FIG=ROOT/'IEEE_AQUALOC_Revision2/figures';FIG.mkdir(exist_ok=True,parents=True)
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':8,'axes.titlesize':9,'axes.labelsize':8,'legend.fontsize':7,'xtick.labelsize':7,'ytick.labelsize':7,'axes.spines.top':False,'axes.spines.right':False})
COL=['#22679b','#b36420','#3c8658']
def save(fig,name):
 for ext in ['png','pdf']:fig.savefig(FIG/(name+'.'+ext),dpi=320,bbox_inches='tight')
 plt.close(fig)
def main():
 for ext in ['png','pdf']:shutil.copy2(ROOT/'IEEE_AQUALOC_Revision/figures'/('fixed_grid.'+ext),FIG/('fixed_grid.'+ext))
 fig,axes=plt.subplots(3,1,figsize=(3.5,5.0),layout='constrained')
 for ax,alg in zip(axes,['ORB','AKAZE','SIFT']):
  d=pd.read_csv(R2/f'switch_null_draws_{alg}.csv');iid=d[d['null']=='uniform_frame_keyframe_budgets'].ap;shift=d[d['null']=='keyframe_circular_shift'].ap;obs=float(d[d['null']=='observed_contrast'].ap.iloc[0]);rev=float(d[d['null']=='reversed_contrast_both_budgets'].ap.iloc[0]);ax.hist(iid,bins=35,density=True,histtype='step',color=COL[0],label='IID (10,000)');ax.hist(shift,bins=25,density=True,histtype='step',color=COL[1],label='Circular shifts (170)');ax.axvline(obs,color='black',lw=1.2,label='Observed');ax.axvline(rev,color=COL[2],ls=':',lw=1.2,label='Reversed contrast');ax.set(title=alg,ylabel='Density',xlim=(.45,.97))
 axes[0].legend(ncol=2,fontsize=6,loc='upper left');axes[-1].set_xlabel('Count average precision');save(fig,'temporal_nulls')
 d=pd.read_csv(R2/'paper_matrix_v2.csv');fig,ax=plt.subplots(figsize=(3.5,3.7),layout='constrained')
 for i,r in enumerate(d.itertuples()):
  c=COL[['ORB','AKAZE','SIFT'].index(r.detector)];ax.plot([r.rmse_m,r.clean_rmse_m],[i,i],color=c,lw=1);ax.scatter(r.rmse_m,i,color=c,marker='o',s=25);ax.scatter(r.clean_rmse_m,i,color=c,marker='x',s=30)
 ax.set(yticks=range(9),yticklabels=[r.detector+' '+r.preprocessing for r in d.itertuples()],xlabel='Common-set rolling RMSE (m)',title='Depth 1e9; fit 40 s; thirteen test windows',xlim=(0,.8));ax.invert_yaxis();ax.text(.03,.01,'○ All reference (691 tests)   × Edge exclusion (683)',transform=ax.transAxes,fontsize=6);save(fig,'dense_matrix')
 seg=pd.read_csv(R2/'dense_segments.csv');err=pd.read_csv(R2/'dense_per_pose.csv');group='source_policies_depth1e9';names=['ORB_fixed_rho0.06_quality_depth1e+09','ORB_fixed_rho0.06_descriptor_first_depth1e+09'];fig,axes=plt.subplots(2,1,figsize=(3.5,4.3),layout='constrained')
 for name,c,label in zip(names,COL,['Quality','Descriptor first']):
  a=seg[(seg.group==group)&(seg['mask']=='all_reference')&(seg.fit_frames==800)&(seg.configuration==name)].sort_values('test_boundary');axes[0].plot(a.test_boundary/20,a.rmse,'o-',c=c,label=label,ms=3)
 ee=[err[(err.group==group)&(err['mask']=='all_reference')&(err.fit_frames==800)&(err.configuration==name)] for name in names];a=ee[0].merge(ee[1],on=['segment','frame'],suffixes=('_q','_d'));axes[1].scatter(a.frame/20,a.error_d-a.error_q,s=3,alpha=.5,c=COL[0]);axes[1].axhline(0,c='gray',lw=.7);axes[0].set(xlabel='Test-window start (nominal s)',ylabel='Segment RMSE (m)');axes[0].legend();axes[1].set(xlabel='Nominal sequence time (s)',ylabel='Per-position error D − Q (m)');save(fig,'dense_policy')
 d=pd.read_csv(R2/'short_gap_cutoff_probe.csv');a=d[(d.source=='selected')&(d.accepted==1)];low=set(a[a.cutoff==50].frame);high=set(a[a.cutoff==1e9].frame);fig,axes=plt.subplots(2,1,figsize=(3.5,4.4),layout='constrained');cuts=[50,200,1000,10000,1e9];axes[0].plot(range(5),[sum(a.cutoff==c) for c in cuts],'o-',c=COL[0]);axes[0].set(xticks=range(5),xticklabels=['50','200','1e3','1e4','1e9'],xlabel='Explicit far-depth cutoff',ylabel='Accepted pairs / 169',ylim=(0,170))
 values=[];labels=[]
 for cutoff,ids,label in [(50,low&high,'Both\nat 50'),(1e9,low&high,'Both\nat 1e9'),(1e9,high-low,'High only\nat 1e9'),(50,low-high,'Low only\nat 50')]:
  b=a[(a.cutoff==cutoff)&a.frame.isin(ids)];values.append(b.direction_error_deg.to_numpy());labels.append(label+'\nn='+str(len(b)))
 axes[1].boxplot(values,showfliers=True,flierprops=dict(markersize=2));axes[1].set(xticks=range(1,5),xticklabels=labels,ylabel='Direction consistency error (°)',ylim=(0,185));save(fig,'cutoff_groups')
 trials=pd.read_csv(R2/'sampling_trial_summary.csv');fig,axes=plt.subplots(2,1,figsize=(3.5,4.2),layout='constrained')
 for prep,c in zip(['raw','fixed','adaptive'],COL):
  a=trials[(trials.preprocessing==prep)&(trials.policy=='quality')].sort_values('seed');axes[0].plot(range(5),a.acceptance_percent,'o-',c=c,label=prep,ms=4)
 for policy,c,label in [('quality',COL[0],'Quality'),('descriptor_first',COL[1],'Descriptor first')]:
  a=trials[(trials.preprocessing=='fixed')&(trials.policy==policy)].sort_values('seed');axes[1].plot(range(5),a.common_rmse_m,'o-',c=c,label=label,ms=4)
 for ax in axes:ax.set(xticks=range(5),xticklabels=['Ordered','P1','P2','P3','P4']);ax.legend(ncol=2,fontsize=6)
 axes[0].set_ylabel('Accepted attempts (%)');axes[1].set(xlabel='Input-order trial',ylabel='Common-set RMSE (m)',title='Same 675 test positions in eleven windows');save(fig,'sampling_trials')
 ref=pd.read_csv(R2/'reference_step_outliers.csv');gap=pd.read_csv(R2/'original_per_transition_errors.csv');gap=gap[gap.in_boundary_gap==1].sort_values('frame');fig,axes=plt.subplots(2,1,figsize=(3.5,4.4),layout='constrained');axes[0].plot(ref.end/20,ref.step_m,lw=.7,c=COL[0]);bad=ref[ref.robust_outlier==1];axes[0].scatter(bad.end/20,bad.step_m,s=14,c=COL[1]);axes[0].axhline(THRESH,c='gray',ls=':',label='Median + 6 scaled MAD');axes[0].set(yscale='log',xlabel='Nominal sequence time (s)',ylabel='Supplied reference step (m)');axes[0].legend(fontsize=6)
 for key,c in zip(['x','y','z'],COL):axes[1].plot(np.r_[2120,gap.frame],np.r_[0,np.cumsum(gap['increment_residual_'+key])],c=c,label=key)
 axes[1].set(xlabel='Image frame',ylabel='Cumulative increment residual (m)',title='72 accepted steps between 2120 and 2360');axes[1].legend(ncol=3);save(fig,'transition_audit');print('Created seven figure assets',flush=True)
if __name__=='__main__':main()
