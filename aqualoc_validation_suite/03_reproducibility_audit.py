#!/usr/bin/env python3
"""Read-only file, metadata, v2-policy, decision/pair, source provenance audit."""
import argparse,csv,hashlib,json,platform,statistics,sys
from collections import Counter,defaultdict
from pathlib import Path

def hash_file(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for chunk in iter(lambda:f.read(1024*1024),b''):h.update(chunk)
    return h.hexdigest()

def read_rows(p):
    with p.open(newline='',encoding='utf-8-sig') as f:return list(csv.DictReader(f))

def main():
    p=argparse.ArgumentParser();p.add_argument('--root',required=True);p.add_argument('--adaptive-package',required=True);p.add_argument('--output',required=True)
    args=p.parse_args();root=Path(args.root);pkg=Path(args.adaptive_package)
    out=Path(args.output);out.mkdir(parents=True,exist_ok=True)
    locations={
      'original_metrics':root/'raw_data/sequence6_frontend_results/feature_pair_metrics.csv',
      'adaptive_v2_metrics':root/'sequence6_adaptive_v2_full/adaptive_feature_pair_metrics.csv',
      'fixed_control_metrics':root/'sequence6_fixed_clahe_control_full/adaptive_feature_pair_metrics.csv',
      'frame_decisions':root/'sequence6_adaptive_v2_full/adaptive_frame_decisions.csv',
      'adaptive_metadata':root/'sequence6_adaptive_v2_full/adaptive_metadata.json',
      'adaptive_preprocessing_source':pkg/'adaptive_clahe.py',
      'adaptive_benchmark_source':pkg/'aqualoc_adaptive_benchmark.py',
      'original_frontend_source':pkg/'aqualoc_sequence6_frontend_original.py',
      'fixed_control_source':pkg/'aqualoc_fixed_clahe_control.py',
    }
    manifest={k:{'path':str(v),'exists':v.is_file(),'bytes':v.stat().st_size if v.is_file() else None,'sha256':hash_file(v) if v.is_file() else None} for k,v in locations.items()}
    checks={}
    if locations['frame_decisions'].is_file():
        rows=read_rows(locations['frame_decisions'])
        counts=Counter(r['decision'] for r in rows)
        checks['decision_rows']=len(rows)
        checks['decision_counts']=dict(counts)
        checks['unique_frame_ids']=len({r['frame_id'] for r in rows})==len(rows)
        checks['v2_reported_counts_match']=(len(rows)==3385 and counts=={'raw':808,'moderate_clahe':2577})
        checks['strong_clahe_count']=counts.get('strong_clahe',0)
    for k in ('original_metrics','fixed_control_metrics','adaptive_v2_metrics'):
        if locations[k].is_file():
            rows=read_rows(locations[k]);checks[k+'_rows']=len(rows)
            checks[k+'_detector_counts']=dict(Counter(r['detector'] for r in rows))
    if locations['adaptive_metadata'].is_file():
        meta=json.loads(locations['adaptive_metadata'].read_text(encoding='utf-8'))
        checks['v2_parameters_as_recorded']=meta.get('parameters',{})
        checks['v2_decision_rule_verified_against_claude_original']=False
    try:
        import cv2,numpy
        software={'python':sys.version,'opencv':cv2.__version__,'numpy':numpy.__version__,'platform':platform.platform()}
    except ImportError:software={'python':sys.version,'platform':platform.platform()}
    report={'provenance':'AQUALOC archaeological site Sequence 06; dataset source URL / citation must be verified from official dataset documentation',
            'reproducibility':{'baseline':'fixed clipLimit 2.0 grid 8x8','adaptive_v1':'global >=35 AND local >=15 -> RAW; else enhanced unless severe','adaptive_v2':'mean local std >=10.5 -> RAW; otherwise severe if global <25 AND local <10; else moderate CLAHE clipLimit=2.0; inferred from user notes and synthetic-tool implementation','caveat':'v2 threshold selected from Sequence 06 local contrast histogram after inspecting v1; no generalization claim'},
            'manifest':manifest,'checks':checks,'software':software,
            'known_limits':['Sequence-06-only post-hoc tuning','Temporal dependence and 52 mixed pairs','SIFT original/control 40 inlier count and 42 ratio count mismatches (from earlier observed audit)','COLMAP trajectory exports Sim(3) equivalent, not independent ground truth','Frontend estimated-F epipolar residual is NOT reference Sampson error','No independent SLAM poses or trajectory ATE from frontend metrics','Missing calibrated/mutual matching scripts prevent reference AP/F1/Sampson recomputation until sourced']}
    dest=out/'reproducibility_manifest.json';dest.write_text(json.dumps(report,indent=2),encoding='utf-8')
    print('[SAVED]',dest)
    print('DECISIONS',checks.get('decision_counts'))
    for k,v in manifest.items():print(('FOUND' if v['exists'] else 'MISSING'),k,v['path'])

if __name__=='__main__':main()
