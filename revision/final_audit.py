from common import *
from pypdf import PdfReader
from zipfile import ZipFile
from lxml import etree
paper=ROOT/'IEEE_AQUALOC_Revision';rr=[]
for stem,docstem,eqs,imgs in [('ieee_manuscript','ieee_manuscript_final',4,10),('detailed_derivations','detailed_derivations',64,64)]:
 with ZipFile(paper/(docstem+'.docx')) as z:
  root=etree.fromstring(z.read('word/document.xml'));native=len(root.xpath('//*[local-name()="oMath"]'));assert native==eqs
 pdf=PdfReader(paper/(stem+'.pdf'));images={}
 for page in pdf.pages:
  for name,obj in page['/Resources'].get('/XObject',{}).items():
   if obj.get_object().get('/Subtype')=='/Image':images[obj.idnum]=name
 assert len(images)>=imgs,(stem,len(images),imgs)
 assert len(pdf.pages)==(8 if stem=='ieee_manuscript' else 17)
 rr.append(dict(document=stem,pages=len(pdf.pages),native_editable_equations=native,pdf_image_objects=len(images),pdf_minimum_expected_images=imgs,visual_review='All pages inspected; revised gate sweep inspected separately and integrated',pdf_equations='Visible formula images from the same equation source; native editable math remains in DOCX',checks='PASS'))
dump(paper/'delivery_status.json',dict(artifacts=rr,priority_experiments='Items 1, 2 and 6 completed within stated single-sequence scope',numerical_validation=json.loads((OUT/'revision_validation_summary.json').read_text()),repository_release_tag='pending; no new revision tag published',word_web='Manuscript uploaded; supplement upload and web PDF export pending',raw_images='not duplicated; all completed numerical outputs and available original non-image evidence are included'))
env=json.loads((OUT/'new_environment.json').read_text());env['final_source_sha256']={p.name:sha(p) for p in sorted((ROOT/'revision').glob('*')) if p.suffix in ['.py','.json','.cpp']};dump(OUT/'new_environment.json',env)
print('PASS: 8-page manuscript, 17-page supplement, 68 native editable equations and visible PDF formulas',flush=True)
