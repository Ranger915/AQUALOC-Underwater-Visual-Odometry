from common import *
sys.path.insert(0,str(ROOT/'source'))
import build_package as b
from write_paper_v2 import TITLE
b.OUT=ROOT/'IEEE_AQUALOC_Revision2';b.FIG=b.OUT/'figures';b.TITLE=TITLE;b.SUBTITLE='A controlled study on AQUALOC Sequence 06'
b.TABLE_WIDTHS={'Nine-way common-set rolling position errors at depth 1e9':[.57,.38,.55,.71,1.19]}
for main,src,stem in [(True,'manuscript.md','ieee_manuscript'),(False,'supplement.md','detailed_derivations')]:b.Builder(main).build((b.OUT/src).read_text(encoding='utf-8'),stem)
