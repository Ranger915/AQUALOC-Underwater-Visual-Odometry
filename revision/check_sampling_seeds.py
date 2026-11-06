from common import *
from run_vo import pose
rr=[]
for file in sorted((CACHE/'cutoff_fixed_pairs').glob('*.npz'))[:20]:
 with np.load(file) as z:a,b=z['a'],z['b']
 if len(a)<15:continue
 baseline=pose(a,b,1e9,20261105)
 for seed in [20261105,20261106,20261107,20261108,20261109]:
  for perm in [False,True]:
   r=pose(a,b,1e9,seed,permute=perm);same=(r['R'] is None and baseline['R'] is None) or (r['R'] is not None and baseline['R'] is not None and np.array_equal(r['R'],baseline['R']) and np.array_equal(r['t'],baseline['t']) and r['nc']==baseline['nc']);rr.append(dict(pair=file.stem,seed=seed,permuted=perm,identical_to_ordered_baseline=same,ne=r['ne'],nc=r['nc']))
target=ROOT/'revision2_results';writecsv(target/'sampling_seed_validation.csv',rr);dump(target/'sampling_seed_validation.json',dict(pairs=len(set(r['pair'] for r in rr)),global_seed_only_all_identical=all(r['identical_to_ordered_baseline'] for r in rr if not r['permuted']),permuted_different=sum(not r['identical_to_ordered_baseline'] for r in rr if r['permuted']),source='https://github.com/opencv/opencv/blob/4.11.0/modules/calib3d/src/ptsetreg.cpp#L158',interpretation='OpenCV legacy RANSAC uses a fixed local RNG; seeded joint input permutations alter sampled hypotheses without changing the correspondence set; this also probes order/tie sensitivity'))
print('Validated actual sampling intervention',flush=True)
