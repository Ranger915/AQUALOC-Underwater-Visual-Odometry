"""Regression check: changing run length does not change saved Sequence 06 poses."""
from common import *
import run_vo

dest=ROOT/'sequence5_results/sequence6_regression'
run_vo.run('ORB','fixed',result_dir=dest,frames=20,comparisons=[(.06,'quality',50.),(.06,'quality',1e9)])
for cutoff in ['50','1e+09']:
 name=f'ORB_fixed_rho0.06_quality_depth{cutoff}';old=readcsv(OUT/'vo'/f'{name}.csv')[:20];new=readcsv(dest/'vo'/f'{name}.csv');assert len(new)==len(old)==20
 for a,b in zip(old,new):
  for field in ['frame','attempted_anchor','fresh','source','status','ne','nc']:assert a[field]==b[field],(field,a[field],b[field])
  for field in [f'p{k}' for k in range(3)]+[f'R{i}{j}' for i in range(3) for j in range(3)]:assert abs(float(a[field])-float(b[field]))<1e-10
dump(dest/'validation.json',dict(passed=True,frames=20,cutoffs=[50,1e9],scope='new run-length argument preserves the original ordered Sequence 06 output'))
print('PASS: both 20-frame Sequence 06 prefixes reproduce',flush=True)
