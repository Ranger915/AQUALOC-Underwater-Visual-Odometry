from common import *
sys.path.insert(0,str(ROOT/'revision_deps'))
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd

def main():
 folder=ROOT/'IEEE_AQUALOC_Revision5/figures';folder.mkdir(parents=True,exist_ok=True)
 d=pd.concat([pd.read_csv(ROOT/f'baseline_results/evaluation_sequence{s}/native_all_fit_metrics.csv') for s in [6,5]])
 d=d[d.scope=='paired_native_all_fit'];plt.rcParams.update({'font.size':8,'axes.spines.top':False,'axes.spines.right':False})
 fig,ax=plt.subplots(figsize=(3.5,4.1),layout='constrained');labels=[]
 for y,(seq,prep) in enumerate((s,p) for s in [6,5] for p in ['raw','fixed','adaptive']):
  labels.append(f'{seq:02d} {prep}')
  for method,offset,color,label in [('ORB_SLAM3',-.11,'#176a97','ORB-SLAM3'),('pairwise_ORB',.11,'#b35b1c','Imposed-step ORB')]:
   x=d[(d.sequence==seq)&(d.preprocessing==prep)&(d.method==method)].rmse_m.to_numpy()
   if len(x):ax.plot([x.min(),x.max()],[y+offset]*2,c=color,lw=1);ax.scatter(x,np.full(len(x),y+offset),color=color,s=17,alpha=.75,label=label if y==0 else None)
 ax.set(yticks=range(6),yticklabels=labels,xlabel='Native paired all-fit RMSE (m)');ax.invert_yaxis();ax.set_xlim(left=0);ax.legend(fontsize=7);ax.grid(axis='x',alpha=.15)
 for ext in ['png','pdf']:fig.savefig(folder/('baseline_native.'+ext),dpi=300,bbox_inches='tight')
 plt.close(fig)

if __name__=='__main__':main()
