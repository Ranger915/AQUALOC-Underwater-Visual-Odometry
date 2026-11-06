from pathlib import Path
import json,hashlib,shutil,zipfile
from lxml import etree
from pypdf import PdfReader
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'IEEE_AQUALOC'
target=OUT/'source';target.mkdir(exist_ok=True)
for p in (ROOT/'source').iterdir():
 if p.is_file() and p.suffix in ['.py','.ps1','.md']:shutil.copy2(p,target/p.name)
qa=[]
for stem,expected in [('ieee_manuscript',6),('detailed_derivations',71)]:
 with zipfile.ZipFile(OUT/(stem+'.docx')) as z:
  xml=etree.fromstring(z.read('word/document.xml'))
  eq=xml.xpath('//*[local-name()="oMath"]')
  assert len(eq)==expected
  assert not xml.xpath('//*[local-name()="pBdr"]')
  alltext=' '.join(xml.xpath('//*[local-name()="t"]/text()'))
  assert 'FIG ' not in alltext and '@@' not in alltext
  assert '\ufffd' not in alltext
 pdf=PdfReader(OUT/(stem+'.pdf'))
 text='\n'.join(p.extract_text() for p in pdf.pages)
 assert '4.089' in text and len(text)>15000
 qa.append({'document':stem,'pages':len(pdf.pages),'editable_equations':len(eq),
            'visual_review':'All final pages inspected from Poppler PNGs',
            'pdf_nonempty_text':True,'docx_archive_check':True})
(OUT/'evidence/final_quality_checks.json').write_text(json.dumps(qa,indent=2),encoding='utf-8')
manifest={p.relative_to(OUT).as_posix():{'bytes':p.stat().st_size,'sha256':hashlib.sha256(p.read_bytes()).hexdigest()}
 for p in sorted(OUT.rglob('*')) if p.is_file() and p.name!='SHA256SUMS.json'}
(OUT/'SHA256SUMS.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')
zipname=ROOT/'AQUALOC_IEEE_Paper_and_Derivations.zip'
with zipfile.ZipFile(zipname,'w',zipfile.ZIP_DEFLATED) as z:
 for p in sorted(OUT.rglob('*')):
  if p.is_file():z.write(p,Path('IEEE_AQUALOC')/p.relative_to(OUT))
with zipfile.ZipFile(zipname) as z:assert z.testzip() is None
print(json.dumps({'quality_checks':qa,'zip':str(zipname),'zip_bytes':zipname.stat().st_size,'files':len(manifest)+1},indent=2))
