"""Run every available non-ATE plotting module; ATE needs a real SLAM trajectory."""
import argparse, subprocess, sys
from pathlib import Path

p=argparse.ArgumentParser(description=__doc__)
p.add_argument('--root',required=True)
p.add_argument('--frame-a',type=int,default=100)
p.add_argument('--frame-b',type=int,default=120)
args=p.parse_args()
base=Path(__file__).resolve().parent
jobs=[('02_feature_maps_matches.py',['--frame-a',str(args.frame_a),'--frame-b',str(args.frame_b)]),
      ('03_cost_maps_heatmaps.py',[]),('04_adaptive_vs_fixed.py',[]),('05_publication_bar_charts.py',[])]
failed=[]
for name,extra in jobs:
    cmd=[sys.executable,str(base/name),'--root',args.root,*extra]
    print('\n'+'='*65+'\nRUNNING '+name+'\n'+'='*65,flush=True)
    try:subprocess.run(cmd,check=True)
    except subprocess.CalledProcessError as exc:
        print('[ERROR]',name,'exit',exc.returncode,flush=True);failed.append(name)
print('\nSummary:',len(jobs)-len(failed),'passed,',len(failed),'failed')
if failed:print('Failed:',', '.join(failed))
sys.exit(1 if failed else 0)
