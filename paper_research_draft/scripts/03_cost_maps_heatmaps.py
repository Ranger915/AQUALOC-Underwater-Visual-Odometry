"""Parameter-response surfaces for F1, precision, recall (not robot occupancy/planning cost maps)."""
import argparse
import numpy as np
import matplotlib.pyplot as plt
from common import style, summary, output, export, DETECTORS, PREPS, SCOPES

def plot_kind(root,kind,scope,metric):
    df=summary(root,kind)
    if metric not in df.columns:raise ValueError(f'No {metric} field in {kind} summary')
    out=output(root,'03_parameter_response_heatmaps')
    minc='minimum_mutual_inliers' if kind=='mutual' else 'minimum_reference_inliers'
    fig,axes=plt.subplots(3,3,figsize=(13.5,11.5),sharex=True,sharey=True)
    for i,det in enumerate(DETECTORS):
        for j,pr in enumerate(PREPS):
            ax=axes[i,j]
            sub=df[(df.detector==det)&(df.preprocessing==pr)&(df.scope==scope)]
            table=sub.pivot(index=minc,columns='sampson_threshold_px',values=metric).sort_index().sort_index(axis=1)
            if table.empty:raise ValueError(f'Missing response surface for {kind}/{det}/{pr}/{scope}')
            z=table.to_numpy(float)
            obj=ax.imshow(z,vmin=0,vmax=1,cmap='viridis',origin='lower',aspect='auto')
            for r in range(len(table.index)):
                for c in range(len(table.columns)):
                    value=z[r,c]
                    if np.isfinite(value):ax.text(c,r,f'{value:.3f}',ha='center',va='center',fontsize=8,color='white' if value<.55 else 'black')
            ax.set_xticks(range(len(table.columns)),[str(x) for x in table.columns]);ax.set_yticks(range(len(table.index)),[str(x) for x in table.index]);
            ax.set_title(f'{det} | {pr}');ax.set_xlabel('Sampson distance threshold [px]');ax.set_ylabel('Minimum verified inliers')
    fig.colorbar(obj,ax=axes.ravel().tolist(),shrink=.6,label=f'{metric} (larger is better)')
    fig.suptitle(f'{kind.capitalize()} reference matching | {scope} | {metric}',fontsize=15)
    fig.subplots_adjust(top=.91,bottom=.07,left=.07,right=.89,hspace=.35,wspace=.3)
    export(fig,out,f'{kind}_{scope}_{metric}_response_surface')

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',required=True)
    p.add_argument('--scopes',nargs='+',default=SCOPES,choices=SCOPES)
    p.add_argument('--metrics',nargs='+',default=['f1','precision','recall'],choices=['f1','precision','recall'])
    args=p.parse_args();style()
    for kind in ['calibrated','mutual']:
        for scope in args.scopes:
            for metric in args.metrics:plot_kind(args.root,kind,scope,metric)
    print('[NOTE] Maps are benchmark parameter surfaces, NOT spatial robot cost maps.')

if __name__=='__main__':main()
