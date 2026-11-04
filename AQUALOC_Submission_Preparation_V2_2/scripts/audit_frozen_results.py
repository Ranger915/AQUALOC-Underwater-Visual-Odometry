#!/usr/bin/env python3
"""Read-only AQUALOC Sequence 06 CSV/source and environment audit.
Checks exact table/score definitions on mutual and calibrated pair metrics,
file SHA-256, pose/VO coverage, and signs off ONLY checks actually run.
Works without adding any Python package dependencies.
"""
import argparse, collections, csv, datetime, hashlib, json, math, os, platform, subprocess, sys
from pathlib import Path

EXPECTED = {
    'ORB': {'raw':(.294030,.548765),'clahe':(.841633,.918386),'adaptive_v2':(.837581,.912467)},
    'AKAZE_MLDB': {'raw':(.169600,.357551),'clahe':(.547755,.815782),'adaptive_v2':(.496389,.796251)},
    'SIFT': {'raw':(.394406,.653586),'clahe':(.879668,.943451),'adaptive_v2':(.867317,.932795)}
}

def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda:f.read(1024*1024),b''): h.update(block)
    return h.hexdigest()

def read(path):
    with path.open(encoding='utf-8-sig',newline='') as f: return list(csv.DictReader(f))

def ap(labels,scores):
    # sklearn-compatible non-interpolated AP with full tie blocks
    if not labels or sum(labels)==0: return float('nan')
    order=sorted(range(len(labels)),key=lambda i:scores[i],reverse=True)
    tp=fp=old_tp=0; acc=0.0
    for j,i in enumerate(order):
        if labels[i]:tp+=1
        else:fp+=1
        if j==len(order)-1 or scores[i]!=scores[order[j+1]]:
            acc+=(tp-old_tp)*tp/(tp+fp);old_tp=tp
    return acc/sum(labels)

def score_rows(rows, threshold=8):
    tp=fp=fn=tn=0; labels=[];scores=[];fracs=[]
    for r in rows:
        label=int(float(r['overlap_label']))
        v=float(r['mutual_inliers_2px'])
        fraction=float(r['mutual_fraction_2px'])
        positive=v>=threshold
        if label not in (0,1) or v<0 or not math.isfinite(v) or not math.isfinite(fraction):
            raise ValueError('Invalid label or score')
        labels.append(label);scores.append(v);fracs.append(fraction)
        if positive and label:tp+=1
        elif positive:fp+=1
        elif label:fn+=1
        else:tn+=1
    prec=tp/(tp+fp) if tp+fp else 0
    recall=tp/(tp+fn) if tp+fn else 0
    f1=2*tp/(2*tp+fp+fn) if 2*tp+fp+fn else 0
    return dict(tp=tp,fp=fp,fn=fn,tn=tn,precision=prec,recall=recall,f1=f1,
                average_precision_count=ap(labels,scores),average_precision_fraction=ap(labels,fracs),n=len(rows))

def run(root,report,include_images=False):
    inputs={
      'pair_manifest':root/'raw_data/sequence6_calibrated_reference/sequence6_calibrated_pair_manifest.csv',
      'mutual_original':root/'raw_data/sequence6_mutual_matching_results/sequence6_mutual_feature_metrics.csv',
      'mutual_original_summary':root/'raw_data/sequence6_mutual_matching_results/sequence6_mutual_feature_summary.csv',
      'mutual_adaptive':root/'sequence6_adaptive_v2_mutual_matching_results/sequence6_mutual_feature_metrics.csv',
      'mutual_adaptive_summary':root/'sequence6_adaptive_v2_mutual_matching_results/sequence6_mutual_feature_summary.csv',
      'calibrated_original':root/'raw_data/sequence6_calibrated_feature_results/sequence6_reference_feature_metrics.csv',
      'calibrated_original_summary':root/'raw_data/sequence6_calibrated_feature_results/sequence6_reference_feature_summary.csv',
      'calibrated_adaptive':root/'sequence6_adaptive_v2_calibrated_reference_results/sequence6_reference_feature_metrics.csv',
      'calibrated_adaptive_summary':root/'sequence6_adaptive_v2_calibrated_reference_results/sequence6_reference_feature_summary.csv',
      'frontend_original':root/'raw_data/sequence6_frontend_results/feature_pair_metrics.csv',
      'frontend_adaptive':root/'sequence6_adaptive_v2_full/adaptive_feature_pair_metrics.csv',
      'frontend_control':root/'sequence6_fixed_clahe_control_full/adaptive_feature_pair_metrics.csv',
      'ap_bootstrap':root/'sequence6_paper_validation/adaptive_v2_mutual_ap_paired_bootstrap.csv',
      'f1_bootstrap':root/'sequence6_paper_validation/adaptive_reference_f1_bootstrap.csv',
      'trajectory':root/'paper_research_draft/vo_full_sequence_v3/vo_sequence6_orb_clahe_v3.csv',
      'reference':root/'colmap_traj_sequence_6.txt',
      'reference_alternate':root/'new_archaeo_colmap_traj_sequence_06.txt',
      'full_vo_evaluation':root/'paper_research_draft/full_sequence_trajectory_evaluation/summary_flat.csv',
      'v3_engine':root/'aqualoc_vo_v3_pilot/slam_engine_sequence6_v3.py',
    }
    # Plus all text/python/yaml/json files in the relevant reproducibility folders, without rehashing the images by default.
    for folder in ('aqualoc_vo_v3_pilot','sequence6_paper_validation','paper_research_draft/scripts'):
        d=root/folder
        if d.exists():
            for p in d.rglob('*'):
                if p.is_file() and p.suffix.lower() in ('.py','.yaml','.yml','.json','.md'):
                    inputs.setdefault('supplemental:'+str(p.relative_to(root)),p)
    missing=[k for k,p in inputs.items() if not p.exists()]
    files=[dict(key=k,path=str(p.relative_to(root)),bytes=p.stat().st_size,sha256=sha(p))
           for k,p in inputs.items() if p.is_file()]
    if include_images:
        img=root/'raw_data/images_sequence_6'
        if img.exists():
            for p in sorted(img.glob('*.png')):
                files.append(dict(key='source_image',path=str(p.relative_to(root)),bytes=p.stat().st_size,sha256=sha(p)))
    checks=[];tables=[]
    def check(name,ok,details,required=True):
        checks.append(dict(check=name,status='PASS' if ok else 'FAIL',details=details,required=required))
    for k in ('mutual_original','mutual_adaptive','calibrated_original','calibrated_adaptive','frontend_original'):
        p=inputs[k]
        if p.exists():
            n=sum(1 for _ in p.open(encoding='utf-8-sig'))-1
            expected={'mutual_original':86190,'mutual_adaptive':43095,'calibrated_original':86190,'calibrated_adaptive':43095,'frontend_original':60774}[k]
            check(k+'_row_count',n==expected,f'{n}/{expected}')
        else: check(k+'_row_count',False,'Missing file')
    # All original/adaptive detector scores are recomputed from per-pair CSV, not summaries.
    if inputs['mutual_original'].exists() and inputs['mutual_adaptive'].exists():
        original=read(inputs['mutual_original']);adaptive=read(inputs['mutual_adaptive'])
        for name,rows,preps in [('original',original,['raw','clahe']),('adaptive',adaptive,['adaptive_v2'])]:
            groups=collections.defaultdict(list)
            for r in rows:
                if r['preprocessing'].lower() in preps:
                    k=(r['detector'].upper(),r['preprocessing'].lower())
                    groups[k].append(r)
            for (det,prep),rr in sorted(groups.items()):
                pids=[(int(r['frame_id_i']),int(r['frame_id_j'])) for r in rr]
                check(f'{name}_{det}_{prep}_unique_pairs',len(set(pids))==14365 and len(rr)==14365,
                      f'unique={len(set(pids))},rows={len(rr)}')
                vals=score_rows(rr)
                tables.append(dict(detector=det,preprocessing=prep,scope='all_pairs',sampson_threshold_px=2,minimum_mutual_inliers=8,**vals))
                want=EXPECTED.get(det,{}).get(prep)
                if want:
                    check(f'{det}_{prep}_reproduces_manuscript',
                          abs(vals['f1']-want[0])<1e-5 and abs(vals['average_precision_count']-want[1])<1e-5,
                          f"F1={vals['f1']:.7f} vs {want[0]}; AP={vals['average_precision_count']:.7f} vs {want[1]}")
    else: check('mutual_table_recalculation',False,'Missing original/adaptive mutual pair CSVs')
    # Recompute calibrated non-mutual operating point (2px / 8 reference inliers)
    cal_expected={
      'ORB': {'raw':(.414938,.597611),'clahe':(.886394,.930927),'adaptive_v2':(.882088,.926415)},
      'AKAZE_MLDB': {'raw':(.184127,.337309),'clahe':(.611111,.828474),'adaptive_v2':(.563557,.814632)},
      'SIFT': {'raw':(.433538,.661262),'clahe':(.858618,.944442),'adaptive_v2':(.852941,.935690)}
    }
    calibrated_tables=[]
    if inputs['calibrated_original'].exists() and inputs['calibrated_adaptive'].exists():
        for name,path,preps in [('original',inputs['calibrated_original'],['raw','clahe']),
                                ('adaptive',inputs['calibrated_adaptive'],['adaptive_v2'])]:
            groups=collections.defaultdict(list)
            for r in read(path):
                if r['preprocessing'].lower() in preps:
                    groups[r['detector'].upper(),r['preprocessing'].lower()].append(r)
            for (det,prep),rr in sorted(groups.items()):
                pairids={(int(r['frame_id_i']),int(r['frame_id_j'])) for r in rr}
                check(f'calibrated_{det}_{prep}_unique_pairs',len(rr)==14365 and len(pairids)==14365,
                      f'rows={len(rr)}, unique_pairs={len(pairids)}')
                # Use exact saved reference inlier count and overlap labels; AP is 2px count score.
                labels=[int(float(r['overlap_label'])) for r in rr]
                scores=[float(r['reference_inliers_2px']) for r in rr]
                tp=sum(y==1 and score>=8 for y,score in zip(labels,scores))
                fp=sum(y==0 and score>=8 for y,score in zip(labels,scores))
                fn=sum(y==1 and score<8 for y,score in zip(labels,scores))
                tn=sum(y==0 and score<8 for y,score in zip(labels,scores))
                f1=2*tp/(2*tp+fp+fn) if tp+fp+fn else 0
                cap=ap(labels,scores)
                calibrated_tables.append(dict(detector=det,preprocessing=prep,scope='all_pairs',
                                              sampson_threshold_px=2,minimum_reference_inliers=8,
                                              TP=tp,FP=fp,FN=fn,TN=tn,f1=f1,average_precision_2px=cap))
                want=cal_expected.get(det,{}).get(prep)
                if want:
                    check(f'calibrated_{det}_{prep}_reproduces_manuscript',
                          abs(f1-want[0])<1e-5 and abs(cap-want[1])<1e-5,
                          f'F1={f1:.7f} vs {want[0]}; AP={cap:.7f} vs {want[1]}')
    else: check('calibrated_table_recalculation',False,'Missing original/adaptive calibrated pair CSVs')
    # Confirm full VO count/coverage directly.
    if inputs['trajectory'].exists():
        rows=read(inputs['trajectory']); ids=[int(r['frame']) for r in rows]
        accepted=sum(r['tracking_success']=='1' for r in rows)
        fresh=sum(int(r['frame'])%20==0 and r['tracking_success']=='1' for r in rows)
        check('vo_full_run',len(rows)==3385 and len(set(ids))==3385 and accepted==1188 and fresh==60,
              f'frames={len(rows)} unique={len(set(ids))} accepted={accepted} fresh_keyframes={fresh}')
    else:check('vo_full_run',False,'Missing trajectory CSV')
    if inputs['reference'].exists():
        r=[line for line in inputs['reference'].read_text(encoding='utf-8-sig').splitlines() if line.strip() and not line.startswith('#')]
        check('reference_count',len(r)==170,f'poses={len(r)}')
    else:check('reference_count',False,'Missing COLMAP reference')
    env={'os':platform.platform(),'python':sys.version,'executable':sys.executable,
         'machine':platform.machine(),'processor':platform.processor(),
         'cpu_count':os.cpu_count(),'seed_annotation':'AP bootstrap seed from command/script; OpenCV RANSAC calls may be nondeterministic'}
    for mod in ('numpy','pandas','matplotlib','cv2','scipy','sklearn'):
        try:
            m=__import__(mod);env[mod]=getattr(m,'__version__','available')
            if mod=='cv2':
                env['opencv_build_excerpt']='\n'.join(m.getBuildInformation().splitlines()[:65])
        except Exception as exc:env[mod]='not available: '+str(exc)
    if sys.platform=='win32':
        try:
            cmd=['powershell','-NoProfile','-Command','Get-CimInstance Win32_Processor | Select-Object -ExpandProperty Name; Get-CimInstance Win32_VideoController | Select-Object -ExpandProperty Name; Get-CimInstance Win32_PhysicalMemory | Measure-Object -Property Capacity -Sum | Select-Object -ExpandProperty Sum']
            env['windows_hardware']=subprocess.run(cmd,capture_output=True,text=True,timeout=30).stdout.strip()
        except Exception as exc:env['windows_hardware_error']=str(exc)
    report.mkdir(parents=True,exist_ok=True)
    manifest={'generated_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),
              'root':str(root),'input_files':files,'missing_expected_files':missing,
              'checks':checks,'note':'Read-only audit. No result is claimed verified if data are absent.'}
    for name,obj in [('reproducibility_manifest.json',manifest),('environment.json',env)]:
        (report/name).write_text(json.dumps(obj,indent=2,ensure_ascii=False),encoding='utf-8')
    if calibrated_tables:
        with (report/'recomputed_calibrated_operating_point.csv').open('w',newline='',encoding='utf-8') as f:
            w=csv.DictWriter(f,fieldnames=list(calibrated_tables[0]));w.writeheader();w.writerows(calibrated_tables)
    if tables:
        with (report/'recomputed_mutual_operating_point.csv').open('w',newline='',encoding='utf-8') as f:
            w=csv.DictWriter(f,fieldnames=list(tables[0]));w.writeheader();w.writerows(tables)
    manifest['generated_analysis_outputs']=[dict(path=str(p.relative_to(report)),bytes=p.stat().st_size,sha256=sha(p))
        for p in sorted(report.glob('recomputed_*_operating_point.csv')) if p.is_file()]
    (report/'reproducibility_manifest.json').write_text(json.dumps(manifest,indent=2,ensure_ascii=False),encoding='utf-8')
    fails=[c for c in checks if c['status']=='FAIL']
    lines=['# Sequence 06 frozen-source audit','',f"Run: {manifest['generated_utc']}",
           f"Checks passed: {len(checks)-len(fails)}/{len(checks)}",'',
           '| Check | Status | Evidence |','|---|---|---|']
    for c in checks:lines.append(f"| {c['check']} | {c['status']} | {c['details']} |")
    lines+=['','Missing expected inputs: '+(', '.join(missing) if missing else 'none'),
            '','**Not audited here:** reference pose semantics; physical scale of local export; independent-sequence generalization; substantive source-code correspondence to user-run configurations; full literature novelty; image insertion; author declarations.']
    (report/'audit_report.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    print(f'Audit complete: {len(checks)-len(fails)} PASS, {len(fails)} FAIL. Reports: {report}')
    return 0 if not fails else 2

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',required=True,type=Path)
    p.add_argument('--report',type=Path)
    p.add_argument('--hash-images',action='store_true')
    a=p.parse_args();root=a.root.expanduser().resolve()
    report=a.report or root/'paper_research_draft'/'submission_audit'
    raise SystemExit(run(root,report,a.hash_images))
