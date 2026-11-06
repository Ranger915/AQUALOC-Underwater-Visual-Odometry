#!/usr/bin/env python3
"""AQUALOC Sequence 06: explicit-convention, coverage-aware VO vs COLMAP evaluator.

Do not treat COLMAP reconstruction as independent ground truth. Sim(3) ATE
measures reconstruction-relative consistency after global scale fitting.
"""
import argparse
import csv
import json
import math
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

CONFIGS = ('orb_raw', 'orb_clahe', 'akaze_raw', 'akaze_clahe')


def quaternion_matrix(q):
    x, y, z, w = np.asarray(q, dtype=float)
    norm = np.linalg.norm([x, y, z, w])
    if not np.isfinite(norm) or norm < 1e-12:
        raise ValueError('Invalid reference quaternion')
    x, y, z, w = (np.array([x, y, z, w]) / norm)
    return np.array([
        [1-2*(y*y+z*z), 2*(x*y-z*w), 2*(x*z+y*w)],
        [2*(x*y+z*w), 1-2*(x*x+z*z), 2*(y*z-x*w)],
        [2*(x*z-y*w), 2*(y*z+x*w), 1-2*(x*x+y*y)],
    ])


def read_reference(path, convention):
    points = {}
    with open(path, encoding='utf-8-sig') as fh:
        for line_no, line in enumerate(fh, 1):
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            parts = line.replace(',', ' ').split()
            if len(parts) != 8:
                raise ValueError(f'Reference line {line_no}: expected 8 columns, got {len(parts)}')
            a = [float(v) for v in parts]
            frame = int(round(a[0]))
            if abs(a[0]-frame) > 1e-5 or frame in points:
                raise ValueError(f'Invalid/duplicate reference frame at line {line_no}')
            t = np.array(a[1:4], dtype=float)
            if convention == 'twc':
                center = t
            else:
                R_cw = quaternion_matrix(a[4:8])
                center = -R_cw.T @ t
            if not np.all(np.isfinite(center)):
                raise ValueError(f'Nonfinite position for frame {frame}')
            points[frame] = center
    if not points:
        raise ValueError('Reference has no poses')
    return points


def read_estimate(path):
    result = {}
    with open(path, encoding='utf-8-sig', newline='') as fh:
        reader = csv.DictReader(fh)
        req = {'frame','x','y','z','tracking_success'}
        if not reader.fieldnames or not req.issubset(reader.fieldnames):
            raise ValueError(f'{path.name}: missing required fields {req-set(reader.fieldnames or [])}')
        for row in reader:
            frame = int(float(row['frame']))
            if frame in result:
                raise ValueError(f'{path.name}: duplicate frame {frame}')
            position = np.array([float(row[k]) for k in ('x', 'y', 'z')], dtype=float)
            if not np.all(np.isfinite(position)):
                raise ValueError(f'{path.name}: nonfinite position at frame {frame}')
            success = str(row['tracking_success']).strip().lower() in ('1','true','yes')
            result[frame] = {'position':position, 'accepted':success,
                             'status':row.get('status',''),
                             'parallax':parse_num(row.get('median_parallax_deg')),
                             'essential':parse_num(row.get('essential_inliers')),
                             'cheirality':parse_num(row.get('cheirality_inliers'))}
    return result


def parse_num(v):
    try:
        n = float(v)
        return n if math.isfinite(n) else None
    except (TypeError, ValueError):
        return None


def align_sim3(source, target):
    """Least-squares Umeyama; target ~= s R source + t."""
    n = len(source)
    if n < 3:
        raise ValueError('At least 3 paired points required')
    src_mean, dst_mean = source.mean(axis=0), target.mean(axis=0)
    src, dst = source-src_mean, target-dst_mean
    var = np.sum(src*src)/n
    if var < 1e-12:
        raise ValueError('VO has insufficient spatial variation for Sim(3) alignment')
    cov = dst.T @ src/n
    U, singular, Vt = np.linalg.svd(cov)
    diag = np.ones(3)
    if np.linalg.det(U@Vt) < 0:
        diag[-1] = -1
    R = U @ np.diag(diag) @ Vt
    scale = float(np.dot(singular, diag)/var)
    if scale <= 0 or not np.isfinite(scale):
        raise ValueError('Invalid fitted scale')
    t = dst_mean-scale*R@src_mean
    return scale, R, t


def aligned(points, scale, R, t):
    return scale * (points @ R.T) + t


def metrics(errors):
    if not len(errors):
        return {}
    return { 'rmse':float(np.sqrt(np.mean(errors**2))),
             'mean':float(np.mean(errors)), 'median':float(np.median(errors)),
             'p95':float(np.percentile(errors,95)), 'max':float(np.max(errors)),
             'evaluated_poses':int(len(errors)) }


def write_rows(path, rows):
    if not rows: return
    with open(path,'w',newline='',encoding='utf-8') as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def saveplot(path):
    plt.tight_layout()
    plt.savefig(path,dpi=300,bbox_inches='tight')
    plt.close()


def run_config(name, estimate_path, reference, output_dir, fit_mode):
    estimate = read_estimate(estimate_path)
    common = sorted(set(reference)&set(estimate))
    accepted = [f for f in common if estimate[f]['accepted'] or f == 0]
    strictly_accepted = [f for f in common if estimate[f]['accepted']]
    # All associated frame rows includes carried-forward poses. Primary report
    # fits and evaluates only accepted keyframes; carried-forward poses are
    # supplementary and explicitly labelled as such.
    if fit_mode == 'accepted':
        fit_frames = strictly_accepted
    elif fit_mode == 'all_associated':
        fit_frames = common
    else:
        raise ValueError(fit_mode)
    summary = {
        'configuration':name, 'estimate_file':str(estimate_path),
        'total_estimate_frames':len(estimate),
        'reference_keyframes':len(reference),
        'associated_keyframes':len(common),
        'accepted_associated_keyframes':len(strictly_accepted),
        'accepted_associated_coverage_pct':100*len(strictly_accepted)/len(reference),
        'all_frame_acceptance_pct':100*sum(int(v['accepted']) for v in estimate.values())/max(1,len(estimate)-1),
        'fit_mode':fit_mode, 'fit_points':len(fit_frames),
    }
    if len(fit_frames) < 3:
        summary['evaluation_status']='insufficient_accepted_reference_keyframes'
        summary['note']='Run full Sequence 06 or use a longer segment; do not report ATE.'
        return summary
    src = np.stack([estimate[f]['position'] for f in fit_frames]); target = np.stack([reference[f] for f in fit_frames])
    try:
        scale, R, t = align_sim3(src, target)
    except ValueError as ex:
        summary['evaluation_status']='alignment_failed'
        summary['note']=str(ex)
        return summary
    fit_err = np.linalg.norm(aligned(src,scale,R,t)-target,axis=1)
    summary['evaluation_status']='computed'
    summary['sim3_scale']=scale
    summary['fit_ate'] = metrics(fit_err)
    summary['warning']='In-sample Sim(3) fitted and evaluated on same poses; COLMAP reconstruction reference, not independent measured ground truth.'
    out = output_dir/name
    out.mkdir(parents=True,exist_ok=True)
    coords=[]
    for f in common:
        pos=estimate[f]['position']
        match=reference[f]
        p=aligned(pos.reshape(1,3),scale,R,t)[0]
        err=float(np.linalg.norm(p-match))
        coords.append({'frame':f,'accepted_transition':int(estimate[f]['accepted']),
                       'status':estimate[f]['status'], 'error':err,
                       'est_aligned_x':p[0],'est_aligned_y':p[1],'est_aligned_z':p[2],
                       'ref_x':match[0],'ref_y':match[1],'ref_z':match[2],
                       'median_parallax_deg':estimate[f]['parallax'],
                       'essential_inliers':estimate[f]['essential'],
                       'cheirality_inliers':estimate[f]['cheirality']})
    write_rows(out/'associated_poses_errors.csv',coords)
    # Secondary errors include held poses and must be interpreted with caution.
    errors_all=np.array([r['error'] for r in coords])
    summary['all_associated_including_held_poses_ate']=metrics(errors_all)
    summary['accepted_keyframes_ate']=metrics(np.array([r['error'] for r in coords if r['accepted_transition']]))

    eval_rows=[r for r in coords if r['accepted_transition']]
    if eval_rows:
        frames=np.array([r['frame'] for r in eval_rows]); errors=np.array([r['error'] for r in eval_rows])
        plt.figure(figsize=(9,4));plt.plot(frames,errors,'.-');plt.xlabel('Frame ID');plt.ylabel('Position error (COLMAP units)');
        plt.title(name+' — accepted keyframe ATE after Sim(3)');saveplot(out/'ATE_accepted_vs_frame.png')
        plt.figure(figsize=(7,4));plt.hist(errors,bins=min(30,max(5,int(math.sqrt(len(errors))))));
        plt.xlabel('Position error (COLMAP units)');plt.ylabel('Count');plt.title(name+' — ATE histogram');saveplot(out/'ATE_histogram.png')
        ss=np.sort(errors);plt.figure(figsize=(7,4));plt.plot(ss,np.arange(1,len(ss)+1)/len(ss));
        plt.xlabel('Position error (COLMAP units)');plt.ylabel('CDF');plt.title(name+' — ATE CDF');saveplot(out/'ATE_CDF.png')
        E=np.array([[r[f'est_aligned_{a}'] for a in 'xyz'] for r in eval_rows]);
        G=np.array([[r[f'ref_{a}'] for a in 'xyz'] for r in eval_rows]);
        fig,axes=plt.subplots(1,3,figsize=(15,4));
        for ax,(i,j,axeslab) in zip(axes,[(0,1,'XY'),(0,2,'XZ'),(1,2,'YZ')]):
            ax.plot(G[:,i],G[:,j],'.-',label='COLMAP reference')
            ax.plot(E[:,i],E[:,j],'.-',label='VO (Sim3 aligned)')
            ax.set_xlabel(axeslab[0]);ax.set_ylabel(axeslab[1]);ax.set_title(axeslab)
            ax.axis('equal');ax.legend(fontsize=8)
        fig.suptitle(name+' — accepted reference-frame positions');saveplot(out/'trajectory_projections.png')

    frames=sorted(estimate)
    y=np.array([int(estimate[f]['accepted']) for f in frames])
    plt.figure(figsize=(12,2.7));plt.scatter(frames,y,s=7);
    plt.yticks([0,1],['not accepted','accepted']);plt.xlabel('Frame ID');
    plt.title(name+' — frame-transition acceptance');saveplot(out/'tracking_acceptance.png')
    diag=[(f,estimate[f]['parallax'],estimate[f]['essential'],estimate[f]['cheirality']) for f in frames]
    diag=[d for d in diag if d[1] is not None]
    if diag:
        plt.figure(figsize=(9,4));plt.scatter([d[0] for d in diag],[d[1] for d in diag],s=8)
        plt.xlabel('Frame ID');plt.ylabel('Median parallax (deg)');plt.title(name+' — parallax');saveplot(out/'parallax_vs_frame.png')
    with open(out/'evaluation_summary.json','w',encoding='utf-8') as fh:
        json.dump(summary,fh,indent=2)
    return summary


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',required=True)
    p.add_argument('--reference',help='COLMAP reference text file; defaults to first export under root')
    p.add_argument('--reference-convention',choices=['twc','tcw'],required=True,
                   help='twc = XYZ camera center in world; tcw = world-to-camera t, converted using quaternion. Verify from metadata before publishing.')
    p.add_argument('--fit-mode',choices=['accepted','all_associated'],default='accepted',
                   help='accepted: fit on accepted transitions only; all_associated: includes stale held poses (diagnostics only).')
    p.add_argument('--estimates-dir')
    p.add_argument('--output-dir')
    p.add_argument('--configs',nargs='+',choices=CONFIGS,default=list(CONFIGS))
    args=p.parse_args()
    root=Path(args.root)
    if not root.exists(): p.error(f'Project root does not exist: {root}')
    if args.reference:
        ref_path=Path(args.reference)
    else:
        possible=list(root.rglob('colmap_traj_sequence_6.txt'))
        if not possible: p.error('Cannot locate colmap_traj_sequence_6.txt; specify --reference')
        ref_path=possible[0]
    reference=read_reference(ref_path,args.reference_convention)
    est_dir=Path(args.estimates_dir) if args.estimates_dir else root/'paper_research_draft'/'trajectories_v2'
    out_dir=Path(args.output_dir) if args.output_dir else root/'paper_research_draft'/'figures'/'trajectory_evaluation'
    out_dir.mkdir(parents=True,exist_ok=True)
    print('Reference:',ref_path, '| convention:',args.reference_convention,'| keyframes:',len(reference))
    summaries=[]
    for name in args.configs:
        path=est_dir/f'vo_sequence6_{name}_v2.csv'
        if not path.exists():
            print('[MISSING]',path)
            continue
        try:
            s=run_config(name,path,reference,out_dir,args.fit_mode)
            summaries.append(s)
            print(f"[{s['evaluation_status']}] {name} | accepted associated={s['accepted_associated_keyframes']}/{s['reference_keyframes']} | fit_points={s['fit_points']}")
            if s['evaluation_status']=='computed':
                print(f"   Sim3 scale={s['sim3_scale']:.6g} | ATE RMSE={s['accepted_keyframes_ate'].get('rmse',float('nan')):.6g} (COLMAP units)")
        except Exception as ex:
            print('[ERROR]',name,ex)
            summaries.append({'configuration':name,'evaluation_status':'error','error':str(ex)})
    if not summaries:
        p.error(f'No trajectory CSVs found in {est_dir}')
    with open(out_dir/'evaluation_manifest.json','w',encoding='utf-8') as fh:
        json.dump({'reference':str(ref_path),'reference_convention':args.reference_convention,
                   'fit_mode':args.fit_mode,'configurations':summaries},fh,indent=2)
    print('Saved reports and PNGs to:',out_dir)
    print('NOTE: COLMAP-based reconstruction consistency, not independent ground-truth ATE.')

if __name__=='__main__':main()
