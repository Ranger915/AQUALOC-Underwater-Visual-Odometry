from common import *
sys.path.insert(0,str(ROOT/'source'))
import build_package as b
from write_paper_v3 import TITLE
b.OUT=ROOT/'IEEE_AQUALOC_Revision3';b.FIG=b.OUT/'figures';b.TITLE=TITLE;b.SUBTITLE='A methodological audit of pairwise motion and reconstruction consistency'
b.SUPPLEMENT_TITLE='Detailed Derivations for Feature Matching and Motion Evaluation'
b.KEEP_TABLES={'Fixed-detector reconstruction differences with paired ranges'}
b.TABLE_WIDTHS={'Nine-way common-set rolling position errors at depth 1e9':[.57,.38,.55,.71,1.19],'Common-set motion controls on the same 691 test positions':[1.55,.48,1.37],'Fixed-detector reconstruction differences with paired ranges':[1.02,.65,1.73]}
for main,src,stem in [(True,'manuscript.md','ieee_manuscript'),(False,'supplement.md','detailed_derivations')]:b.Builder(main).build((b.OUT/src).read_text(encoding='utf-8'),stem)
