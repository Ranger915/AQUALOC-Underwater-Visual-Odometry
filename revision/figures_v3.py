from dense_reference import *
sys.path.insert(0,str(ROOT/'revision_deps'))
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd,shutil
R3=ROOT/'revision3_results';FIG=ROOT/'IEEE_AQUALOC_Revision3/figures';FIG.mkdir(parents=True,exist_ok=True)
plt.rcParams.update({'font.size':8,'axes.spines.top':False,'axes.spines.right':False,'savefig.dpi':300})

def save(fig,name):
 fig.savefig(FIG/(name+'.png'),dpi=300,bbox_inches='tight');fig.savefig(FIG/(name+'.pdf'),bbox_inches='tight');plt.close(fig)

def main():
 for p in (ROOT/'IEEE_AQUALOC_Revision2/figures').iterdir():shutil.copy2(p,FIG/p.name)
 s=pd.read_csv(R3/'motion_control_summary.csv');names=[f'{a}_{p}_rho0.06_quality_depth1e+09' for a in ['ORB','AKAZE','SIFT'] for p in ['raw','fixed','adaptive']];fig,ax=plt.subplots(figsize=(3.5,4.4),layout='constrained')
 for clean,marker,shift in [(False,'o',-.07),(True,'s',.07)]:
  d=s[(s.group=='high_matrix_common')&(s['mask']==('robust_reference_edges_excluded' if clean else 'all_reference'))].set_index('configuration');ax.scatter([d.loc[n].rmse_m for n in names],np.arange(9)+shift,marker=marker,s=17,label='Edge exclusion (683)' if clean else 'All reference (691)')
 d=s[(s.group=='high_matrix_common')&(s['mask']=='all_reference')].set_index('configuration')
 for name,color,label in [('reference_cv_last5s','#9b3d27','Recent-velocity control'),('reference_hold','#666666','Hold control')]:ax.axvline(d.loc[name].rmse_m,c=color,ls='--',lw=.9,label=label)
 ax.set(yticks=np.arange(9),yticklabels=[n.split('_')[0]+' '+n.split('_')[1] for n in names],xlabel='Reconstruction-consistency RMSE (m)',xlim=(0,.85),title='Safe reference brackets; fit 40 s');ax.invert_yaxis();ax.legend(fontsize=6,loc='lower left');save(fig,'dense_matrix')
 seg=pd.read_csv(R3/'motion_control_segments.csv');err=pd.read_csv(R3/'motion_control_per_pose.csv');group='source_policies_depth1e9';names=['ORB_fixed_rho0.06_quality_depth1e+09','ORB_fixed_rho0.06_descriptor_first_depth1e+09'];fig,axes=plt.subplots(2,1,figsize=(3.5,4.3),layout='constrained')
 for name,label in zip(names,['Quality','Descriptor first']):
  a=seg[(seg.group==group)&(seg['mask']=='all_reference')&(seg.configuration==name)].sort_values('test_boundary');axes[0].plot(a.test_boundary/20,a.rmse,'o-',label=label,ms=3)
 e=[err[(err.group==group)&(err['mask']=='all_reference')&(err.configuration==n)] for n in names];d=e[0].merge(e[1],on=['test_boundary','frame'],suffixes=('_q','_d'));axes[1].scatter(d.frame/20,d.error_d-d.error_q,s=3,alpha=.55);axes[1].axhline(0,c='gray',lw=.7);axes[0].set(xlabel='Test-window start (nominal s)',ylabel='Segment RMSE (m)');axes[0].legend();axes[1].set(xlabel='Nominal sequence time (s)',ylabel='Per-position difference D − Q (m)');save(fig,'dense_policy')
 trials=pd.read_csv(R3/'seed_causal_summary.csv');fig,axes=plt.subplots(2,1,figsize=(3.5,4.2),layout='constrained')
 for prep in ['raw','fixed','adaptive']:
  a=trials[(trials.preprocessing==prep)&(trials.policy=='quality')].sort_values('seed');axes[0].plot(range(5),a.acceptance_percent,'o-',label=prep,ms=3)
 for policy,label in [('quality','Quality'),('descriptor_first','Descriptor first')]:
  a=trials[(trials.preprocessing=='fixed')&(trials.policy==policy)].sort_values('seed');axes[1].plot(range(5),a.common_rmse_m,'o-',label=label,ms=3)
 for ax in axes:ax.set(xticks=range(5),xticklabels=['Ordered','P1','P2','P3','P4']);ax.legend(fontsize=6)
 axes[0].set_ylabel('Accepted attempts (%)');axes[1].set(xlabel='Input-order trial',ylabel='Reconstruction RMSE (m)',title='Same 675 tests; safe reference brackets');save(fig,'sampling_trials');print('Updated causal-fit figures and control lines',flush=True)

if __name__=='__main__':main()
