"""Prepare, then hash-check a bounded copy to the explicitly requested folder."""
from common import *
import shutil,zipfile
from pypdf import PdfReader
from lxml import etree
SOURCE=ROOT/'IEEE_AQUALOC_Revision5_Illustrated'
TARGET=Path(r'C:\Users\Raaghav\Desktop\Research\AQUALOC_Complete_Findings_Revision5').resolve()
STAGE=ROOT/'qa/illustrated_deployment'
BACKUP=ROOT/'qa/before_desktop_illustrated'
PLAN=ROOT/'qa/illustrated_copy_plan.json'

def prepare():
 pdf=ROOT/'qa/illustrated_manuscript_final/ieee_manuscript.pdf';assert len(PdfReader(pdf).pages)==11
 shutil.copy2(pdf,SOURCE/'ieee_manuscript.pdf')
 with zipfile.ZipFile(SOURCE/'ieee_manuscript.docx') as z:tree=etree.fromstring(z.read('word/document.xml'))
 ns={'w':'http://schemas.openxmlformats.org/wordprocessingml/2006/main','m':'http://schemas.openxmlformats.org/officeDocument/2006/math','a':'http://schemas.openxmlformats.org/drawingml/2006/main'}
 assert len(tree.findall('.//m:oMath',ns))==3 and len(tree.findall('.//w:tbl',ns))==10 and len(tree.findall('.//a:blip',ns))==8
 old=(ROOT/'IEEE_AQUALOC_Revision5/manuscript.md').read_text(encoding='utf-8');new=(SOURCE/'manuscript.md').read_text(encoding='utf-8')
 assert all(line in new for line in old.splitlines() if line.strip())
 rendered=ROOT/'qa/illustrated_manuscript_final'
 assert len(list(rendered.glob('page-*.png')))==11
 for i in range(1,8):assert sha(rendered/f'page-{i}.png')==sha(ROOT/f'qa/v5_main_final/page-{i}.png')
 status=json.loads((SOURCE/'delivery_status.json').read_text());status['documents']['ieee_manuscript'].update(pages=11,docx_sha256=sha(SOURCE/'ieee_manuscript.docx'),pdf_sha256=sha(pdf));status.update(figures=8,illustration_update='Four new full-width figures with Section VI A discussion; scientific metrics unchanged',visual_review='All changed pages 8–11 reviewed; pages 1–7 pixel-identical to reviewed prior render')
 dump(SOURCE/'delivery_status.json',status)
 review=json.loads((SOURCE/'visual_review.json').read_text());review.update(article_pages=11,article_visual_review='Pages 8–11 visually reviewed; pages 1–7 pixel-matched',article_rendered_pages={p.name:sha(p) for p in rendered.glob('page-*.png')});dump(SOURCE/'visual_review.json',review)
 meta=json.loads((SOURCE/'paper_metadata.json').read_text());meta.update(article_pages=11,article_figures=8,illustration_update=True);dump(SOURCE/'paper_metadata.json',meta)
 note='\n\n## Trajectory and spatial error figures\n\nThe article now has eleven pages and eight figures. Figures 5–8 show paired-native trajectory overlays and 0.5 m spatial error costmaps for both sequences. Section VI A defines alignment, pooled repeats, reference XY projection, missing cells and the limits of interpreting reconstructed-reference error. All eight standalone graphs and four manuscript figure variants are in figures/ as PNG and SVG. figures/trajectory_graph_data/ preserves their CSV data, validation, and README. The source numerical experiments and the seventeen-page supplement are unchanged. This extracted folder has been updated; the previously generated ZIP remains the earlier snapshot.\n'
 for name in ['README_FINDINGS.md','FIXES_AND_FINDINGS.md']:
  s=(SOURCE/name).read_text(encoding='utf-8').replace('nine pages','eleven pages').replace('nine-page','eleven-page');(SOURCE/name).write_text(s+note,encoding='utf-8')
 files={}
 for name in ['ieee_manuscript.docx','ieee_manuscript.pdf','ieee_manuscript.md','manuscript.md','delivery_status.json','visual_review.json','paper_metadata.json','pdf_equation_rendering.json','figure_integration.json','README_FINDINGS.md','FIXES_AND_FINDINGS.md']:
  files['IEEE_AQUALOC_Revision5/'+name]=SOURCE/name
 for p in (SOURCE/'figures').glob('*'):
  if p.is_file() and (p.name.startswith('sequence0') or p.name.startswith('paper_sequence')):files['IEEE_AQUALOC_Revision5/figures/'+p.name]=p
 for p in (ROOT/'trajectory_costmaps').iterdir():
  if p.suffix in {'.csv','.json','.md'}:files['IEEE_AQUALOC_Revision5/figures/trajectory_graph_data/'+p.name]=p
 files['IEEE_AQUALOC_Revision5/figures/trajectory_and_costmap_graphs.pdf']=ROOT/'trajectory_costmaps/trajectory_and_costmap_graphs.pdf'
 for name in ['illustrate_manuscript.py','install_illustrated_manuscript.py','plot_trajectories_costmaps.py']:files['revision/'+name]=ROOT/'revision'/name
 STAGE.mkdir(parents=True,exist_ok=True)
 for rel,source in files.items():
  out=STAGE/rel;out.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(source,out)
 rootreadme=(TARGET/'README.md').read_text(encoding='utf-8').replace('nine pages','eleven pages').replace('nine-page','eleven-page')
 (STAGE/'README.md').write_text(rootreadme+note,encoding='utf-8');files['README.md']=STAGE/'README.md'
 rows={r['path']:r for r in readcsv(TARGET/'MANIFEST.csv')}
 for rel in files:
  p=STAGE/rel;rows[rel]=dict(path=rel,bytes=p.stat().st_size,sha256=sha(p))
 writecsv(STAGE/'MANIFEST.csv',[rows[k] for k in sorted(rows)]);files['MANIFEST.csv']=STAGE/'MANIFEST.csv'
 plan=[]
 for rel in sorted(files):
  dest=(TARGET/rel).resolve();assert dest.is_relative_to(TARGET)
  plan.append(dict(relative=rel,new_sha256=sha(STAGE/rel),prior_sha256=sha(dest) if dest.exists() else None))
 dump(PLAN,dict(target=str(TARGET),files=plan,backup=str(BACKUP),article_pages=11,figures=8,prior_zip_unchanged=True))
 print('Prepared',len(plan),'files; native equations, tables, figures, original text and all reviewed page hashes verified',flush=True)

def apply():
 plan=json.loads(PLAN.read_text());assert Path(plan['target']).resolve()==TARGET
 for r in plan['files']:
  p=(TARGET/r['relative']).resolve();assert p.is_relative_to(TARGET)
  assert (sha(p) if p.exists() else None)==r['prior_sha256'],f'Concurrent edit: {p}'
  assert sha(STAGE/r['relative'])==r['new_sha256']
 for r in plan['files']:
  p=TARGET/r['relative']
  if p.exists():
   backup=BACKUP/r['relative'];backup.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(p,backup)
  p.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(STAGE/r['relative'],p)
 for r in plan['files']:assert sha(TARGET/r['relative'])==r['new_sha256']
 dump(ROOT/'qa/illustrated_install_verified.json',dict(passed=True,target=str(TARGET),files_verified=len(plan['files']),pages=11,figures=8,backup=str(BACKUP)))
 print('Installed and SHA256-verified',len(plan['files']),'files; prior versions backed up in workspace',flush=True)

if __name__=='__main__':
 if len(sys.argv)>1 and sys.argv[1]=='--apply':apply()
 else:prepare()
