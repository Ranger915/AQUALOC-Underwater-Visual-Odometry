from common import *
sys.path.insert(0,str(ROOT/'source'))
import build_package as b
from write_transfer_paper import TITLE
b.OUT=ROOT/'IEEE_AQUALOC_Revision4';b.FIG=b.OUT/'figures';b.TITLE=TITLE;b.SUBTITLE='Frozen-policy transfer and reference-aware reconstruction diagnostics'
b.SUPPLEMENT_TITLE='Detailed Derivations for Feature Matching and Motion Evaluation'
b.TABLE_WIDTHS={'Frozen-policy matching AP and adaptive-minus-fixed sensitivity':[.75,.4,.4,.5,1.35],'Frozen reconstructions on corrected-reference common sets':[.68,.62,.48,.39,1.23],'Motion controls on the same nine-way common tests':[1.8,.8,.8],'Fixed-detector paired reconstruction contrasts':[.35,.95,.65,1.45]}
b.KEEP_TABLES={'Sequence 06 development count AP on corrected native IDs','Positive-pair mean consistent-match loss on Sequence 06','Frozen-policy matching AP and adaptive-minus-fixed sensitivity','Motion controls on the same nine-way common tests','Paired source-policy reconstructions with corrected references','Fixed-detector paired reconstruction contrasts'}
b.KEEP_TABLES.update({'Twenty-vertex adaptive-minus-fixed matching sensitivity','Reference-conditioned controls on matched test sets'})
for main,src,stem in [(True,'manuscript.md','ieee_manuscript'),(False,'supplement.md','detailed_derivations')]:b.Builder(main).build((b.OUT/src).read_text(encoding='utf-8'),stem)
