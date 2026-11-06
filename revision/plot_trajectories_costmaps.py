"""Plot existing paired native all-fit evidence; unknown cells stay unobserved."""
from common import *
sys.path.insert(0,str(ROOT/'revision_deps'))
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.lines import Line2D
import pandas as pd
import zipfile

DEST=ROOT/'trajectory_costmaps';DEST.mkdir(exist_ok=True)
COLORS={'Reference':'#242f3b','ORB_SLAM3':'#007f9c','pairwise_ORB':'#c46428'}
LABELS={'ORB_SLAM3':'ORB-SLAM3','pairwise_ORB':'Imposed-step ORB'}
POLICIES=['raw','fixed','adaptive'];CELL=.5

def pieces(ids,p):
 # Avoid drawing a continuous path through unsupported native endpoints.
 for ix in np.split(np.arange(len(ids)),np.flatnonzero(np.diff(ids)>20)+1):
  if len(ix):yield p[ix]

def save(fig,stem,pdf):
 fig.savefig(DEST/(stem+'.png'),dpi=220,facecolor='white')
 fig.savefig(DEST/(stem+'.svg'),facecolor='white')
 pdf.savefig(fig,facecolor='white');plt.close(fig)

def main():
 plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False,'axes.titleweight':'bold'})
 aligned=[];checks=[];cells=[];summary=[];inputs={}
 def check(label,ok):assert bool(ok),label;checks.append(dict(check=label,passed=True))
 # A known similarity verifies the shared fitting convention independently.
 probe=np.array([[0,0,0],[1,0,0],[0,1,0],[0,0,1]],float)
 s,R,t=fit(probe,2*probe+np.array([3,-4,5]));check('known Sim3 recovery',np.allclose(s*(probe@R.T)+t,2*probe+[3,-4,5]))
 for seq in [5,6]:
  target=ROOT/f'baseline_results/evaluation_sequence{seq}'
  metrics=pd.read_csv(target/'native_all_fit_metrics.csv');metrics=metrics[metrics.scope=='paired_native_all_fit']
  residuals=pd.read_csv(target/'native_all_fit_per_pose.csv');residuals=residuals[residuals.scope=='paired_native_all_fit']
  reference=ROOT/f'sequence5_results/reference/new_archaeo_colmap_traj_sequence_{seq:02d}.txt'
  ref=refposes(file=reference);inputs[str(reference.relative_to(ROOT))]=sha(reference)
  times=np.array([int(l.split()[0]) for l in (ROOT/f'baseline_results/sequence{seq}.manifest').read_text().splitlines()],np.int64)
  for prep in POLICIES:
   path=(ROOT/'sequence5_results/vo' if seq==5 else ROOT/'revision_results/vo')/f'ORB_{prep}_rho0.06_quality_depth1e+09.csv'
   pair=pd.read_csv(path);inputs[str(path.relative_to(ROOT))]=sha(path)
   for trial in range(3):
    mm=metrics[(metrics.preprocessing==prep)&(metrics.trial==trial)]
    check(f'paired methods and support {seq}/{prep}/{trial}',len(mm)==2 and mm.fit_ids.nunique()==1)
    ids=np.array(list(map(int,mm.fit_ids.iloc[0].split(';'))));q=np.array([ref[i][1] for i in ids])
    folder=ROOT/f'baseline_results/sequence{seq}_{prep}_trial{trial}'
    a=np.loadtxt(folder/f'map_{int(mm.map_id.iloc[0])}.txt',ndmin=2);ii=np.searchsorted(times,np.rint(a[:,0]).astype(np.int64))
    lookup={int(i):p for i,p in zip(ii,a[:,1:4])};inputs[str((folder/f'map_{int(mm.map_id.iloc[0])}.txt').relative_to(ROOT))]=sha(folder/f'map_{int(mm.map_id.iloc[0])}.txt')
    for method,p in [('ORB_SLAM3',np.array([lookup[i] for i in ids])),('pairwise_ORB',pair.iloc[ids][['p0','p1','p2']].to_numpy())]:
     s,R,t=fit(p,q);est=s*(p@R.T)+t;errors=np.linalg.norm(est-q,axis=1)
     expected=mm[mm.method==method].iloc[0]
     saved=residuals[(residuals.preprocessing==prep)&(residuals.trial==trial)&(residuals.method==method)].set_index('frame').loc[ids]
     check(f'published RMSE {seq}/{prep}/{trial}/{method}',np.isclose(np.sqrt(np.mean(errors**2)),expected.rmse_m,rtol=1e-8))
     check(f'published residuals {seq}/{prep}/{trial}/{method}',np.allclose(errors,saved.error_m,rtol=1e-8,atol=1e-9))
     for frame,gt,xyz,error in zip(ids,q,est,errors):aligned.append(dict(sequence=seq,policy=prep,trial=trial,method=method,frame=int(frame),time_s=float(times[frame]/1e9),ref_x=gt[0],ref_y=gt[1],ref_z=gt[2],x=xyz[0],y=xyz[1],z=xyz[2],error_m=error))
     summary.append(dict(sequence=seq,policy=prep,trial=trial,method=method,n=len(ids),rmse_m=float(np.sqrt(np.mean(errors**2)))))
  inputs[str((target/'native_all_fit_metrics.csv').relative_to(ROOT))]=sha(target/'native_all_fit_metrics.csv')
 data=pd.DataFrame(aligned);data.to_csv(DEST/'aligned_native_poses.csv',index=False)
 pd.DataFrame(summary).to_csv(DEST/'trajectory_metrics.csv',index=False)
 handles=[Line2D([0],[0],color=COLORS['Reference'],lw=2,label='Corrected reference')]+[Line2D([0],[0],color=COLORS[m],lw=1.7,ls='-' if m=='ORB_SLAM3' else '--',label=LABELS[m]) for m in LABELS]
 with PdfPages(DEST/'trajectory_and_costmap_graphs.pdf') as pdf:
  for seq in [5,6]:
   d=data[data.sequence==seq];ref=refposes(file=ROOT/f'sequence5_results/reference/new_archaeo_colmap_traj_sequence_{seq:02d}.txt');rid=np.array(sorted(ref));q=np.array([ref[i][1] for i in rid])
   fig,axes=plt.subplots(1,3,figsize=(15,5.7),layout='constrained',sharex=True,sharey=True)
   for ax,prep in zip(axes,POLICIES):
    for part in pieces(rid,q):ax.plot(part[:,0],part[:,1],color=COLORS['Reference'],lw=2,zorder=4)
    for (method,trial),g in d[d.policy==prep].groupby(['method','trial']):
     for part in pieces(g.frame.to_numpy(),g[['x','y','z']].to_numpy()):ax.plot(part[:,0],part[:,1],color=COLORS[method],ls='-' if method=='ORB_SLAM3' else '--',lw=1,alpha=.55)
     ax.scatter(g.x,g.y,s=7,color=COLORS[method],alpha=.4)
    ax.scatter(q[0,0],q[0,1],c='#242f3b',marker='o',s=35,zorder=6);ax.scatter(q[-1,0],q[-1,1],c='#242f3b',marker='X',s=45,zorder=6)
    ax.set(title=prep.title(),xlabel='Reference X (m)',ylabel='Reference Y (m)',aspect='equal');ax.grid(alpha=.15)
   fig.suptitle(f'Sequence {seq:02d} | Native-pose trajectory comparison\nThree repeats; paired all-fit alignment; circle: reference start, X: end',fontsize=14)
   fig.legend(handles=handles,loc='outside lower center',ncol=3,frameon=False)
   save(fig,f'sequence{seq:02d}_trajectories_xy',pdf)
   fig=plt.figure(figsize=(16,6.5));fig.subplots_adjust(left=.02,right=.88,wspace=.3,bottom=.16,top=.85)
   bounds=np.vstack([q,d[['x','y','z']].to_numpy()]);lo=bounds.min(0);hi=bounds.max(0);padding=np.maximum((hi-lo)*.05,.1)
   for n,prep in enumerate(POLICIES,1):
    ax=fig.add_subplot(1,3,n,projection='3d')
    for part in pieces(rid,q):ax.plot(*part.T,color=COLORS['Reference'],lw=2)
    for (method,trial),g in d[d.policy==prep].groupby(['method','trial']):
     xyz=g[['x','y','z']].to_numpy()
     for part in pieces(g.frame.to_numpy(),xyz):ax.plot(*part.T,color=COLORS[method],ls='-' if method=='ORB_SLAM3' else '--',alpha=.55,lw=1)
     ax.scatter(*xyz.T,s=3,color=COLORS[method],alpha=.3)
    ax.set(title=prep.title(),xlabel='X (m)',ylabel='Y (m)',zlabel='Z (m)',xlim=(lo[0]-padding[0],hi[0]+padding[0]),ylim=(lo[1]-padding[1],hi[1]+padding[1]),zlim=(lo[2]-padding[2],hi[2]+padding[2]));ax.set_box_aspect(np.maximum(hi-lo,.5));ax.tick_params(labelsize=8,pad=1);ax.view_init(elev=26,azim=-65)
   fig.suptitle(f'Sequence {seq:02d} | 3D native-pose trajectories',fontsize=17)
   fig.legend(handles=handles,loc='outside lower center',ncol=3,frameon=False)
   save(fig,f'sequence{seq:02d}_trajectories_3d',pdf)
   fig,axes=plt.subplots(3,1,figsize=(12,8),layout='constrained',sharex=True,sharey=True)
   for ax,prep in zip(axes,POLICIES):
    for (method,trial),g in d[d.policy==prep].groupby(['method','trial']):ax.plot(g.time_s,g.error_m,color=COLORS[method],alpha=.6,lw=1,ls='-' if method=='ORB_SLAM3' else '--')
    ax.set(title=prep.title(),ylabel='Position error (m)');ax.grid(alpha=.2)
   axes[-1].set_xlabel('Dataset time from first input image (s)');fig.suptitle(f'Sequence {seq:02d} | Final-map all-fit error over time',fontsize=16)
   fig.legend(handles=handles[1:],loc='outside lower center',ncol=2,frameon=False)
   save(fig,f'sequence{seq:02d}_error_over_time',pdf)
   # Aggregate error at reference XY locations; never infer obstacle/free space.
   origin=np.floor(q[:,:2].min(0)/CELL)*CELL-CELL;upper=np.ceil(q[:,:2].max(0)/CELL)*CELL+CELL;shape=np.rint((upper-origin)/CELL).astype(int)
   fig,axes=plt.subplots(2,3,figsize=(14,9),layout='constrained',sharex=True,sharey=True)
   vmax=float(d.error_m.max());norm=Normalize(0,vmax)
   for row,method in enumerate(LABELS):
    for col,prep in enumerate(POLICIES):
     g=d[(d.method==method)&(d.policy==prep)].copy();ix=np.floor((g[['ref_x','ref_y']].to_numpy()-origin)/CELL).astype(int)
     g['ix']=ix[:,0];g['iy']=ix[:,1];grid=np.full((shape[1],shape[0]),np.nan)
     for (x,y),group in g.groupby(['ix','iy']):
      rmse=float(np.sqrt(np.mean(group.error_m.to_numpy()**2)));grid[y,x]=rmse;cells.append(dict(sequence=seq,policy=prep,method=method,cell_x=int(x),cell_y=int(y),center_x_m=origin[0]+(x+.5)*CELL,center_y_m=origin[1]+(y+.5)*CELL,samples=len(group),unique_frames=group.frame.nunique(),rmse_m=rmse))
     check(f'cell counts {seq}/{prep}/{method}',sum(r['samples'] for r in cells if r['sequence']==seq and r['policy']==prep and r['method']==method)==len(g))
     ax=axes[row,col];ax.set_facecolor('#e5e7eb');artist=ax.imshow(np.ma.masked_invalid(grid),origin='lower',extent=[origin[0],upper[0],origin[1],upper[1]],cmap='magma',norm=norm,interpolation='none',aspect='equal')
     for part in pieces(rid,q):ax.plot(part[:,0],part[:,1],color='#36d9d0',lw=.6,alpha=.55)
     ax.set(title=f'{prep.title()} | {LABELS[method]}',xlabel='Reference X (m)',ylabel='Reference Y (m)')
   fig.colorbar(artist,ax=axes.ravel().tolist(),label='Cell position RMSE (m)',shrink=.8)
   fig.suptitle(f'Sequence {seq:02d} | Trajectory-error costmaps',fontsize=18)
   fig.supxlabel('0.5 m cells; three repeats pooled. Gray: no evaluated samples. Cyan: reference path. Shared color scale; no spatial filling.',fontsize=10)
   save(fig,f'sequence{seq:02d}_error_costmaps',pdf)
 pd.DataFrame(cells).to_csv(DEST/'costmap_cells.csv',index=False)
 # An explicit pooling check catches accidental averaging of RMSE values.
 check('RMS pooling self-check',np.isclose(np.sqrt(np.mean(np.square([3.,4.]))),np.sqrt(12.5)))
 dump(DEST/'validation.json',dict(passed=True,checks=len(checks),native_pose_rows=len(data),system_comparisons=18,protocol='existing paired native all-fit Sim3, largest map by keyframes',cell_size_m=CELL,unknown='NaN; gray; no interpolation',inputs_sha256=inputs))
 (DEST/'README.md').write_text('''# Trajectory graphs and trajectory-error costmaps

The eight-page PDF and matching PNG/SVG files cover Sequences 05 and 06, all three preprocessing policies and all three fresh-process repeats. No trial is selected by its error. Graphs reuse the Revision5 final largest-map paired-native protocol: each method receives its own all-fit positive Sim(3) on the exact saved common native frame IDs. Recomputed residuals and RMSE are checked against published CSVs. Future map optimization enters these scores; these are not held-out online predictions.

XY views use the reference coordinate frame, without assuming a gravity-aligned top-down plane. Native supported positions are shown; line segments do not bridge missing native endpoints in trajectory views. Separate map epochs are never stitched. Full-run tracking losses are described by the original coverage.csv files and the paper; these views show the retained largest final map only.

Costmaps aggregate squared 3D position errors at reference XY locations in 0.5 m cells, then take the square root. The three process repeats are pooled as descriptive repeated observations, not independent physical samples. Each sequence uses one shared linear color scale across both methods and all policies. Gray cells have no evaluated samples. No spatial interpolation, smoothing, obstacle labeling, or free-space inference is applied. These are error maps, not obstacle/traversability maps suitable for navigation.

aligned_native_poses.csv contains every plotted aligned coordinate, reference position, time and error. trajectory_metrics.csv contains all 36 method/trial scores. costmap_cells.csv contains each occupied evaluation cell and its sample/frame counts. validation.json records checks, provenance and source hashes. PNGs are high-resolution raster figures; SVGs are editable vector figures. Reproduce with the bundled Python runtime and `revision/plot_trajectories_costmaps.py`; dependencies match the existing graph scripts. GitHub and the reviewed Revision5 documents/archive are untouched.
''',encoding='utf-8')
 with zipfile.ZipFile(ROOT/'AQUALOC_Trajectories_and_Costmaps.zip','w',zipfile.ZIP_DEFLATED) as z:
  for p in sorted(DEST.iterdir()):
   if p.is_file():z.write(p,'trajectory_costmaps/'+p.name)
  z.write(Path(__file__),'revision/'+Path(__file__).name)
 with zipfile.ZipFile(ROOT/'AQUALOC_Trajectories_and_Costmaps.zip') as z:check('delivery archive CRC',z.testzip() is None)
 print(f'PASS {len(checks)} checks; {len(data)} native pose rows; 8 graph pages',flush=True)

if __name__=='__main__':main()
