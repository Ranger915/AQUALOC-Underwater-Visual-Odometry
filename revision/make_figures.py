from common import *
sys.path.insert(0,str(ROOT/'revision_deps'))
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd
FIG=ROOT/'IEEE_AQUALOC_Revision/figures';FIG.mkdir(parents=True,exist_ok=True)
plt.rcParams.update({'font.size':8,'axes.titlesize':9,'axes.labelsize':8,'legend.fontsize':7,'xtick.labelsize':7,'ytick.labelsize':7,'font.family':'DejaVu Sans','axes.spines.top':False,'axes.spines.right':False})
ALGS=['ORB','AKAZE','SIFT'];COL=['#22679b','#b36420','#3c8658']
def save(fig,name):
 fig.savefig(FIG/(name+'.png'),dpi=320,bbox_inches='tight');fig.savefig(FIG/(name+'.pdf'),bbox_inches='tight');plt.close(fig)
def main():
 sw=pd.read_csv(OUT/'switch_threshold_random_controls.csv');fr=pd.read_csv(OUT/'frontend_primary_summary.csv');mat=pd.read_csv(OUT/'paper_matrix_summary.csv');seg=pd.read_csv(OUT/'rolling_segment_metrics.csv');err=pd.read_csv(OUT/'rolling_per_pose_errors.csv');vo=pd.read_csv(OUT/'vo_configuration_summary.csv')
 fig,ax=plt.subplots(2,1,figsize=(3.5,4.4),layout='constrained')
 for z,(alg,c) in enumerate(zip(ALGS,COL)):
  a=sw[sw.detector==alg];r=a[a.type=='random_matched_keyframes'];box=ax[0].boxplot(r.ap_count,positions=[z],widths=.4,patch_artist=True);box['boxes'][0].set_facecolor(c);box['boxes'][0].set_alpha(.3)
  ap=a[a.type=='contrast_rule'].ap_count.iloc[0];fixed=fr[(fr.detector==alg)&(fr['mode']=='c2t8_c2t8')].ap_count.iloc[0];ax[0].scatter(z,ap,c=c,marker='o',zorder=4);ax[0].scatter(z,fixed,c=c,marker='x',zorder=4)
  t=a[a.type=='threshold'].sort_values('fraction_raw_keyframes');ax[1].plot(t.fraction_raw_keyframes,t.ap_count,c=c,label=alg);ax[1].scatter(a[a.type=='contrast_rule'].fraction_raw_keyframes,[ap],c=c,s=22)
 ax[0].set(xticks=range(3),xticklabels=ALGS,ylabel='Count average precision',title='24 random seeds: matched frame/keyframe budgets');ax[0].text(.01,.02,'Box: random; dot: contrast rule; ×: fixed',transform=ax[0].transAxes,fontsize=7)
 ax[1].set(xlabel='Fraction of raw reference images',ylabel='Count average precision',title='Contrast-threshold sweep (0–20)');ax[1].legend(loc='lower left');save(fig,'switch_controls')
 fig,axes=plt.subplots(3,1,figsize=(3.5,5.0),layout='constrained')
 for ax,alg in zip(axes,ALGS):
  vals=np.array([[fr[(fr.detector==alg)&(fr['mode']==f'c{c}t{t}_c{c}t{t}')].ap_count.iloc[0] for t in [4,8,16]] for c in [1,2,3,4]])
  ax.imshow(vals,cmap='Blues',vmin=.5,vmax=1,aspect='auto')
  for i in range(4):
   for j in range(3):ax.text(j,i,f'{vals[i,j]:.3f}',ha='center',va='center',color='white' if vals[i,j]>.84 else 'black',fontsize=8)
  ax.set(xticks=range(3),xticklabels=['4 × 4','8 × 8','16 × 16'],yticks=range(4),yticklabels=[1,2,3,4],ylabel='Clip limit',title=alg+' count AP')
 axes[-1].set_xlabel('CLAHE tile grid');save(fig,'fixed_grid')
 fig,axes=plt.subplots(2,1,figsize=(3.5,4.1),layout='constrained')
 for ax,dep,title in zip(axes,['50','1e+09'],['Far-depth cutoff 50','Far-depth cutoff 1e9']):
  for alg,c in zip(ALGS,COL):
   for prep,m in [('raw','o'),('fixed','s'),('adaptive','^')]:
    r=mat[(mat.detector==alg)&(mat.preprocessing==prep)].iloc[0];ax.scatter(r.ap,r['accept_'+dep]*100,c=c,marker=m,s=35)
  ax.set(xlabel='Frontend count AP',ylabel='Accepted attempts (%)',title=title)
 axes[0].text(.03,.94,'ORB / AKAZE / SIFT: blue / orange / green\nRaw ○   Fixed □   Adaptive △',transform=axes[0].transAxes,va='top',fontsize=7);save(fig,'matrix_outcomes')
 group='paired_with_ORB_fixed__ORB_fixed_rho0.06_descriptor_first_depth50';names=['ORB_fixed_rho0.06_quality_depth50','ORB_fixed_rho0.06_descriptor_first_depth50'];fig,axes=plt.subplots(2,1,figsize=(3.5,4.2),layout='constrained')
 for name,c,label in zip(names,COL,['Quality','Descriptor first']):
  a=seg[(seg.group==group)&(seg.k==24)&(seg.configuration==name)].sort_values('segment');axes[0].plot(a.segment+1,a.rmse,'o-',c=c,label=label)
 a=err[(err.group==group)&(err.k==24)&(err.configuration==names[0])];b=err[(err.group==group)&(err.k==24)&(err.configuration==names[1])];p=a.merge(b,on=['segment','frame'],suffixes=('_q','_d'));axes[1].scatter(p.frame,p.error_d-p.error_q,c=COL[0],s=14);axes[1].axhline(0,c='gray',lw=.7)
 axes[0].set(xlabel='Chronological test segment',ylabel='Position RMSE (ref. units)');axes[0].legend();axes[1].set(xlabel='Image frame',ylabel='Paired position error: D − Q');save(fig,'paired_policy')
 fig,axes=plt.subplots(2,1,figsize=(3.5,4.2),layout='constrained')
 for ax,dep in zip(axes,['50','1e+09']):
  group='rho_sweep_common_depth'+dep
  for rho in [.03,.06,.10,.15]:
   name=f'ORB_fixed_rho{rho:.2f}_quality_depth{dep}';v=vo[vo.configuration==name].iloc[0];s=seg[(seg.group==group)&(seg.k==24)&(seg.configuration==name)]
   if len(s):
    rmse=np.sqrt(np.mean(s.rmse**2));ax.scatter(v.acceptance*100,rmse,s=35)
    if rho in [.03,.15]:ax.annotate('0.03 / 0.06 / 0.10' if rho==.03 else '0.15',(v.acceptance*100,rmse),xytext=(-80,10) if rho==.03 else (5,-2),textcoords='offset points',fontsize=7)
  count=int(s.test_n.sum()) if len(s) else 0;ax.set(xlabel='Accepted attempts (%)',ylabel='Common-set RMSE',title=f'Depth {dep}; k=24; {count} test positions');ax.margins(.25)
  from matplotlib.ticker import MaxNLocator,FormatStrFormatter
  ax.xaxis.set_major_locator(MaxNLocator(3));ax.yaxis.set_major_locator(MaxNLocator(4))
  if count:ax.xaxis.set_major_formatter(FormatStrFormatter('%.2f'));ax.yaxis.set_major_formatter(FormatStrFormatter('%.4f'))
  if not count:
   vals=[float(vo[vo.configuration==f'ORB_fixed_rho{r:.2f}_quality_depth{dep}'].iloc[0].acceptance*100) for r in [.03,.06,.10,.15]];ax.bar(range(4),vals,color=COL[0]);ax.set(xticks=range(4),xticklabels=['0.03','0.06','0.10','0.15'],xlabel='Recovered / essential inlier fraction',ylabel='Accepted attempts (%)');ax.text(.5,.94,'No common fit/test window at k=24',ha='center',va='top',transform=ax.transAxes,fontsize=7)
 save(fig,'gate_sweep')
 b=pd.read_csv(OUT/'original_boundary_diagnostic.csv');fig,axes=plt.subplots(2,1,figsize=(3.5,4.3),layout='constrained')
 axes[0].plot(b.frame,b.error,c='gray',lw=.7);fresh=b[b.fresh==1];axes[0].scatter(fresh.frame,fresh.error,c=COL[0],s=12,label='Fresh');stale=b[b.fresh==0];axes[0].scatter(stale.frame,stale.error,s=12,facecolors='none',edgecolors='gray',label='Carried pose')
 for key,c in zip(['residual_x','residual_y','residual_z'],COL):axes[1].plot(b.frame,b[key],c=c,label=key[-1])
 for ax in axes:ax.axvspan(2120,2360,color='#e6c880',alpha=.35);ax.axvline(2120,c='black',ls=':',lw=.8);ax.legend(loc='upper left',ncol=2)
 axes[0].set(ylabel='Original aligned error',title='Original split: 240-frame sampling gap');axes[1].set(xlabel='Image frame',ylabel='Signed residual (ref. units)');save(fig,'boundary_diagnostic')
 print('Created six scientific figures',flush=True)
if __name__=='__main__':main()
