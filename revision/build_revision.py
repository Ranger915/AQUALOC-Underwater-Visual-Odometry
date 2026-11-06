from common import *
sys.path.insert(0,str(ROOT/'source'))
import build_package as b
from write_redraft import TITLE
b.OUT=ROOT/'IEEE_AQUALOC_Revision';b.FIG=b.OUT/'figures';b.TITLE=TITLE;b.SUBTITLE='A controlled single sequence study of contrast enhancement and motion estimation'
for main,src,stem in [(True,'manuscript.md','ieee_manuscript_final'),(False,'supplement.md','detailed_derivations')]:
 b.Builder(main).build((b.OUT/src).read_text(encoding='utf-8'),stem)
