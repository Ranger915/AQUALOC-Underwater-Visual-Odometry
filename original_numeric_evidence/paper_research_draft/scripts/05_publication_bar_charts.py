"""High-DPI publication-style comparisons built from measured CSV summaries."""
import argparse
import numpy as np
import matplotlib.pyplot as plt
from common import summary,point,output,export,style,DETECTORS,PREPS

def bars(root,kind,metric,scope):
    df=summary(root,kind);out=output(root,'05_publication_charts')
    x=np.arange(3);w=.25
    fig,ax=plt.subplots(figsize=(9,5))
    for j,pr in enumerate(PREPS):
        vals=[point(df,d,pr,metric,scope,kind=kind) for d in DETECTORS]
        rects=ax.bar(x+(j-1)*w,vals,w,label=pr.replace('_',' '))
        for rect,value in zip(rects,vals):
            ax.annotate(f'{value:.3f}',(rect.get_x()+rect.get_width()/2,rect.get_height()),
                        xytext=(0,3),textcoords='offset points',ha='center',fontsize=8,rotation=0)
    ax.set_xticks(x,DETECTORS);ax.set_ylim(0,1.08);ax.grid(axis='y')
    ax.set(ylabel=metric.replace('_',' ').title(),title=f'{kind.capitalize()} {metric.replace("_"," ")} | {scope} | 2 px, min 8')
    ax.legend(loc='lower right',fontsize=9)
    export(fig,out,f'{kind}_{metric}_{scope}_bar_comparison')

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',required=True);args=p.parse_args();style()
    for kind in ['calibrated','mutual']:
        df=summary(args.root,kind)
        metrics=['f1','precision','recall','average_precision_count' if kind=='mutual' else 'average_precision_2px']
        for metric in metrics:
            if metric not in df.columns:
                print('[SKIP]',kind,metric,'not in summary');continue
            # AP is threshold-independent ranking metric, but retained alongside the 2px operating-point label
            for scope in ['all_pairs','local_gap_le_100','nonlocal_gap_ge_200']:
                bars(args.root,kind,metric,scope)

if __name__=='__main__':main()
