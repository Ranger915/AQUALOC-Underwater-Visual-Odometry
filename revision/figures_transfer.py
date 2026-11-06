from common import *
sys.path.insert(0,str(ROOT/'revision_deps'))
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd
DEST=ROOT/'sequence5_results';FIG=ROOT/'IEEE_AQUALOC_Revision4/figures';FIG.mkdir(parents=True,exist_ok=True)
plt.rcParams.update({'font.size':8,'axes.spines.top':False,'axes.spines.right':False})

def save(fig,name):
 for ext in ['png','pdf']:fig.savefig(FIG/(name+'.'+ext),dpi=300,bbox_inches='tight')
 plt.close(fig)

def main():
 d=pd.read_csv(DEST/'discovery_frontend_corrected.csv');fig,axes=plt.subplots(3,1,figsize=(3.5,4.7),layout='constrained')
 for ax,alg in zip(axes,['ORB','AKAZE','SIFT']):
  z=np.array([[d[(d.detector==alg)&(d['mode']==f'c{clip}t{tile}_c{clip}t{tile}')].ap_count.iloc[0] for tile in [4,8,16]] for clip in [1,2,3,4]]);ax.imshow(z,cmap='Blues',vmin=.55,vmax=1,aspect='auto')
  for i in range(4):
   for j in range(3):ax.text(j,i,f'{z[i,j]:.3f}',ha='center',va='center',fontsize=7,color='white' if z[i,j]>.88 else 'black')
  ax.set(xticks=range(3),xticklabels=['4 × 4','8 × 8','16 × 16'],yticks=range(4),yticklabels=[1,2,3,4],ylabel='Clip limit',title=alg+' count AP')
 axes[-1].set_xlabel('CLAHE tile grid');save(fig,'fixed_grid')
 fig,axes=plt.subplots(3,1,figsize=(3.5,4.5),layout='constrained')
 for ax,alg in zip(axes,['ORB','AKAZE','SIFT']):
  d=pd.read_csv(DEST/'discovery_nulls_corrected'/f'switch_null_draws_{alg}.csv')
  for kind,label in [('uniform_frame_keyframe_budgets','IID (10,000)'),('keyframe_circular_shift','Circular shifts (166)')]:ax.hist(d[d['null']==kind].ap,bins=28,density=True,histtype='step',label=label)
  ax.axvline(d[d['null']=='observed_contrast'].ap.iloc[0],c='black',lw=.9,label='Observed');ax.axvline(d[d['null']=='reversed_contrast_both_budgets'].ap.iloc[0],c='green',ls=':',lw=.9,label='Reversed contrast');ax.set(title=alg,ylabel='Density')
 axes[0].legend(fontsize=5.5,ncol=2);axes[-1].set_xlabel('Count average precision');save(fig,'temporal_nulls')
 fig,axes=plt.subplots(2,1,figsize=(3.5,6.2),layout='constrained');names=[f'{a}_{p}_rho0.06_quality_depth1e+09' for a in ['ORB','AKAZE','SIFT'] for p in ['raw','fixed','adaptive']]
 for ax,seq in zip(axes,[6,5]):
  d=pd.read_csv(DEST/('sequence6_updated' if seq==6 else 'evaluation_updated')/'summary.csv');d=d[d.group=='high_matrix_common'];allref=d[d['mask']=='all_reference'].set_index('configuration');clean=d[d['mask']=='robust_reference_edges_excluded'].set_index('configuration')
  ax.scatter([allref.loc[n].rmse_m for n in names],np.arange(9)-.06,s=14,label='All reference');ax.scatter([clean.loc[n].rmse_m for n in names],np.arange(9)+.06,s=14,marker='s',label='Edge exclusion')
  for name,color,label in [('reference_cv_last5s','#9b3d27','Past-ref CV (5 s)'),('reference_hold','#666666','Past-ref hold')]:ax.axvline(allref.loc[name].rmse_m,c=color,ls='--',lw=.8,label=label)
  ax.set(yticks=np.arange(9),yticklabels=[n.split('_')[0]+' '+n.split('_')[1][0].upper() for n in names],xlabel='Reconstruction-consistency RMSE (m)',title=f'Sequence {seq:02d}; {int(allref.loc[names[0]].test_positions)} common tests');ax.invert_yaxis();ax.set_xlim(left=0)
 axes[0].legend(fontsize=5.5,ncol=2,loc='lower left');save(fig,'transfer_reconstruction');print('Created three corrected-reference figure sets',flush=True)

if __name__=='__main__':main()
