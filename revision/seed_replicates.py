from common import *
import run_vo,argparse
p=argparse.ArgumentParser();p.add_argument('prep');a=p.parse_args()
run_vo.run('ORB',a.prep,seed=[20261106,20261107,20261108,20261109],result_dir=ROOT/'revision2_results/seed_order_runs',replicate=True)
