from common import *
p=ROOT/'IEEE_AQUALOC_Revision/manuscript.md';s=p.read_text(encoding='utf-8')
for a,b in {
'Rolling similarity alignment replaces the small pilots:':'Rolling similarity alignment provides chronological held-out errors:',
'The original five- and six-test-pose pilots are replaced by larger chronological comparisons. ':'',
'Full sequence source policy comparisons replace the small pilots':'Full sequence source policy comparisons',
'compared with six test positions in the original pilot':'with fit windows and test IDs shared by both methods',
'The result is smaller and less certain than the pilot percentage suggested.':'The main interval spans zero and the effect depends on the fit-window length.',
"We therefore remove the pilot’s acceptance-versus-error claim from the abstract.":'These sensitivities do not establish a general acceptance-versus-error relationship.',
"We therefore remove the pilot's acceptance-versus-error claim from the abstract.":'These sensitivities do not establish a general acceptance-versus-error relationship.',
'For the original Fig. 5,':'For a diagnostic long chronological split with 36 fit positions and 24 test positions,',
'Audit of the original sparse-error figure using all reference IDs':'Audit of the long chronological split using all reference IDs',
'while the rolling comparisons replace the small policy pilots':'while rolling comparisons quantify source-policy sensitivity',
'The revision freezes a primary correspondence metric':'The experiment specifies a primary correspondence metric',
"It retains the original experiment's exploratory status.":'The study is exploratory because the sequence informed the original threshold and engine.',
'and the larger source-policy comparison does not support the original pilot\'s strong improvement wording':'and the source-policy comparison remains sensitive to alignment-window length',
}.items():s=s.replace(a,b)
s=s.replace('Its settings and common-pair counts are listed in the supplement.','At the default depth cutoff, 140 pairs are accepted by every configuration. Median rotation errors range from 1.04° to 2.46° and direction errors from 8.92° to 25.00° on that common set. Fixed and adaptive SIFT have the lowest angular errors in this probe; high frontend AP therefore does not imply poor motion estimation merely because integrated update coverage is lower. Detailed settings and the high-depth comparison are listed in the supplement.')
p.write_text(s,encoding='utf-8')
for a,b in {
'The revision fixes all-pair, mutual, count AP':'We specify all-pair, mutual, count AP',
'Unlike the earlier product-of-multiplicities scheme, this check uses pair inclusion without multiplicity weights.':'This check uses pair inclusion without multiplicity weights.',
'These revised runs replace the old pilot numbers; differences from the legacy saved run can reflect the newly fixed seed schedule and environment as well as an intervention, so the legacy run is retained only as a separate diagnostic.':'All comparisons use this seed schedule and environment. The separate long-split diagnostic is identified as a historical run.',
'the second panel gives F1 versus':'the second panel gives count AP versus',
}.items():s=s.replace(a,b)
p.write_text(s,encoding='utf-8')
print('Finished editorial pass',flush=True)
