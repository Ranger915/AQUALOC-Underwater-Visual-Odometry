#!/usr/bin/env python3
"""AQUALOC Sequence 06 image-sequence monocular visual odometry.

Reuses the user's unmodified slam_engine.py functions for feature detection,
matching, essential-matrix pose recovery, pose composition, and CSV row format.
This is monocular VO (arbitrary scale), NOT a complete SLAM system.
"""
import argparse
import csv
import importlib.util
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np

FX, FY, CX, CY = 543.3327734182214, 542.398772982566, 489.02536042247897, 305.38727712002805
DIST = [-0.1255945656257394, 0.053221287232781606, 9.94070021080493e-05, 9.550660927242349e-05]

def load_engine(path):
    if not path.is_file():
        raise FileNotFoundError(f'Original slam_engine.py missing: {path}')
    spec = importlib.util.spec_from_file_location('user_slam_engine', str(path))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod

def find_frames(root):
    directory = root / 'raw_data' / 'images_sequence_6'
    paths = sorted(directory.glob('frame*.png'))
    if not paths:
        raise FileNotFoundError(f'No frame*.png images in {directory}')
    return paths

def frame_id(path):
    return int(path.stem.replace('frame', ''))

def run_one(engine, frames, output, detector_name, preprocessing, max_frames, ratio, min_inliers, min_ratio, ransac_px):
    K = np.array([[FX, 0.0, CX], [0.0, FY, CY], [0.0, 0.0, 1.0]], dtype=np.float64)
    dist = np.array(DIST, dtype=np.float64).reshape(-1, 1)
    detector = engine.make_detector(detector_name, 1500)
    norm = detector.defaultNorm()
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)) if preprocessing == 'clahe' else None
    T = np.eye(4, dtype=np.float64)
    prev_kps = prev_desc = None
    rows, last_frame_id, attempts, successful = [], None, 0, 0
    use_frames = frames[:max_frames] if max_frames else frames
    for idx, path in enumerate(use_frames):
        t0 = time.perf_counter()
        fid = frame_id(path)
        if last_frame_id is not None and fid != last_frame_id + 1:
            raise ValueError(f'Nonconsecutive frames: {last_frame_id}, {fid}; avoid invalid pose accumulation')
        last_frame_id = fid
        gray = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
        if gray is None:
            raise RuntimeError(f'Cannot read image: {path}')
        if gray.shape != (608, 968):
            raise ValueError(f'Expected calibration resolution 968x608, got {gray.shape[::-1]} for {path}')
        t_pre = time.perf_counter()
        proc = clahe.apply(gray) if clahe else gray
        preprocess_ms = (time.perf_counter() - t_pre) * 1000
        t_det = time.perf_counter()
        kps, desc, raw_count = engine.detect_features(proc, detector, 1500)
        detection_ms = (time.perf_counter() - t_det) * 1000
        matches, inliers, fraction, match_ms, pose_ms, success = [], 0, 0.0, 0.0, 0.0, False
        if prev_kps is not None and prev_desc is not None:
            attempts += 1
            t_match = time.perf_counter()
            matches = engine.match_descriptors(prev_desc, desc, norm, ratio)
            match_ms = (time.perf_counter() - t_match) * 1000
            t_pose = time.perf_counter()
            T_curr_prev, inliers, _, fraction = engine.estimate_monocular_pose(
                prev_kps, kps, matches, K, dist, min_inliers, min_ratio,
                ransac_px, 1.0)
            if T_curr_prev is not None:
                T = T @ np.linalg.inv(T_curr_prev)
                success = True
                successful += 1
            pose_ms = (time.perf_counter() - t_pose) * 1000
        row = engine.make_row(fid, 20.0, T, raw_count, len(kps), len(matches),
                              0, inliers, fraction, success, 'monocular_arbitrary',
                              preprocess_ms, detection_ms, match_ms, pose_ms,
                              (time.perf_counter() - t0)*1000)
        rows.append(row)
        prev_kps, prev_desc = kps, desc
        if (idx+1) % 250 == 0:
            print(f'  {detector_name}/{preprocessing}: {idx+1}/{len(use_frames)} frames', flush=True)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open('w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=engine.CSV_FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    keyframe_file = output.with_name(output.stem + '_keyframes.csv')
    with keyframe_file.open('w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=engine.CSV_FIELDS)
        writer.writeheader()
        writer.writerows([r for r in rows if int(r['frame']) % 20 == 0])
    meta = {'source':'original slam_engine.py functions', 'algorithm':detector_name,
            'preprocessing':preprocessing, 'camera':'AQUALOC Sequence 06 calibrated camera',
            'intrinsics':[FX,FY,CX,CY], 'distortion':DIST,
            'motion_estimator':'essential matrix + recoverPose',
            'output_pose':'T_world_camera', 'translation_scale':'arbitrary 1.0 per accepted step',
            'units':'NOT metres', 'first_frame':int(rows[0]['frame']),
            'last_frame':int(rows[-1]['frame']), 'frames_processed':len(rows),
            'attempted_transitions':attempts, 'accepted_transitions':successful,
            'acceptance_rate':successful / attempts if attempts else None,
            'tracking_failure_behavior':'pose held constant; subsequent matching against previous image',
            'warning':'This pipeline is monocular visual odometry, not loop-corrected SLAM.'}
    output.with_suffix('.json').write_text(json.dumps(meta,indent=2),encoding='utf-8')
    print(f'[SAVED] {output} | accepted {successful}/{attempts} ({100*successful/max(1,attempts):.1f}%)')
    return meta

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root', type=Path, required=True)
    p.add_argument('--engine', type=Path, default=Path.home()/'Desktop'/'deep_sea'/'slam_engine.py')
    p.add_argument('--max-frames',type=int,default=0,help='0 for full 3385-frame sequence; e.g. 100 for smoke test')
    p.add_argument('--algorithms', nargs='+', choices=['ORB','AKAZE'],default=['ORB','AKAZE'])
    p.add_argument('--preprocessings',nargs='+',choices=['raw','clahe'],default=['raw','clahe'])
    p.add_argument('--ratio',type=float,default=0.80)
    p.add_argument('--min-inliers',type=int,default=12)
    p.add_argument('--min-inlier-ratio',type=float,default=0.20)
    p.add_argument('--ransac-px',type=float,default=1.0)
    args=p.parse_args()
    root=args.root.expanduser().resolve()
    engine=load_engine(args.engine.expanduser())
    frames=find_frames(root)
    if frame_id(frames[0])!=0:
        raise ValueError('Sequence must start at frame000000.png for correct reference association')
    out=root/'paper_research_draft'/'trajectories'
    print(f'[INFO] Using {len(frames)} source frames, writing to {out}')
    print('[IMPORTANT] Monocular VO poses have ARBITRARY TRANSLATION SCALE. Use Sim(3) for aligned comparison.')
    all_meta=[]
    for method in args.algorithms:
        for prep in args.preprocessings:
            outfile=out/f'vo_sequence6_{method.lower()}_{prep}.csv'
            all_meta.append(run_one(engine,frames,outfile,method,prep,args.max_frames,args.ratio,args.min_inliers,args.min_inlier_ratio,args.ransac_px))
    (out/'run_manifest.json').write_text(json.dumps(all_meta,indent=2),encoding='utf-8')
    print('[COMPLETE] Generated all requested VO runs; no full-SLAM or metric-ATE claim is implied.')

if __name__=='__main__':
    main()
