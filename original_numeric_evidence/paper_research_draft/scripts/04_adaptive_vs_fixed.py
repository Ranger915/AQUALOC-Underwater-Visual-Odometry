"""Paired contrast (adaptive v2 - fixed CLAHE) plots and experimental uncertainty."""
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from common import style, summary, point, output, export, DETECTORS, SCOPES, require

def contrast(root,kind,metric):
    data=summary(root,kind);out=output(root,'04_adaptive_vs_fixed')
    x=np.arange(len(DETECTORS));w=.25
    fig,ax=plt.subplots(figsize=(10,5))
    for j,scope in enumerate(SCOPES):
        ys=[point(data,d,'adaptive_v2',metric,scope,kind=kind)-point(data,d,'clahe',metric,scope,kind=kind) for d in DETECTORS]
        ax.bar(x+(j-1)*w,ys,w,label=scope.replace('_',' '))
    ax.axhline(0,color='black',lw=1);ax.set_xticks(x,DETECTORS);ax.set(ylabel=f'Δ {metric} (adaptive − fixed)',title=f'{kind} | {metric} difference | 2px / 8 inliers')
    ax.legend(fontsize=8);ax.grid(axis='y');export(fig,out,f'{kind}_{metric}_adaptive_minus_fixed_by_scope')

def ap_ci(root):
    p=Path(root)/'sequence6_paper_validation'/'adaptive_v2_mutual_ap_paired_bootstrap.csv'
    df=pd.read_csv(p);require(df,['detector','scope','metric','paired_difference_adaptive_minus_fixed','ci95_lower','ci95_upper'],str(p))
    df=df[df.metric=='average_precision_count']
    f1path=Path(root)/'sequence6_paper_validation'/'adaptive_reference_f1_bootstrap.csv'
    out=output(root,'04_adaptive_vs_fixed')
    files=[('mutual_average_precision',df)]
    if f1path.exists():
        f=pd.read_csv(f1path)
        if {'detector','scope','ci95_lower','ci95_upper'}.issubset(f.columns):files.append(('mutual_f1',f))
    for name,d in files:
        val='paired_difference_adaptive_minus_fixed' if 'paired_difference_adaptive_minus_fixed' in d else 'difference' if 'difference' in d else None
        if val is None:continue
        fig,ax=plt.subplots(figsize=(9,5));centers=np.arange(3); offsets=[-.23,0,.23]
        for i,scope in enumerate(SCOPES):
            dd=d[d.scope==scope].set_index('detector');pp=[];neg=[];pos=[]
            for detector in DETECTORS:
                if detector not in dd.index:raise ValueError(f'{name}: missing {detector} / {scope}')
                r=dd.loc[detector]
                if isinstance(r,pd.DataFrame):raise ValueError(f'Duplicate bootstrap records for {name} {detector} {scope}')
                v=float(r[val]);lo=float(r.ci95_lower);hi=float(r.ci95_upper)
                pp.append(v);neg.append(max(0,v-lo));pos.append(max(0,hi-v))
            ax.errorbar(centers+offsets[i],pp,yerr=np.array([neg,pos]),fmt='o',capsize=4,label=scope.replace('_',' '))
        ax.axhline(0,color='black',linestyle='--');ax.set_xticks(centers,DETECTORS)
        ax.set(ylabel='Adaptive − fixed (95% paired bootstrap CI)',title=f'{name} uncertainty')
        ax.grid(axis='y');ax.legend(fontsize=8);export(fig,out,f'{name}_bootstrap_ci')

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',required=True);args=p.parse_args();style()
    for kind in ['calibrated','mutual']:
        for metric in ['f1','precision','recall']:contrast(args.root,kind,metric)
    ap_ci(args.root)

if __name__=='__main__':main()
