"""Extend the verified local evidence snapshot; never touches Git."""
from common import *
import zipfile,io,re
from pypdf import PdfReader
from lxml import etree
PAPER=ROOT/'IEEE_AQUALOC_Revision4';RESULTS=ROOT/'sequence5_results'
BASE=ROOT/'AQUALOC_Complete_Findings_Revision3.zip'
PACKAGE=ROOT/'AQUALOC_Complete_Findings_Revision4.zip'

def main():
 validation=json.loads((RESULTS/'validation_summary.json').read_text());assert validation['passed'] and validation['checks']==3054
 markers=list((RESULTS/'vo').glob('*.done.json'));assert len(markers)==20 and all(json.loads(p.read_text())['frames']==3983 for p in markers)
 documents={}
 for stem,pages,equations,images in [('ieee_manuscript',7,3,6),('detailed_derivations',15,72,72)]:
  doc=PAPER/(stem+'.docx');pdf=PAPER/(stem+'.pdf')
  with zipfile.ZipFile(doc) as z:tree=etree.fromstring(z.read('word/document.xml'))
  count=len(tree.findall('.//{http://schemas.openxmlformats.org/officeDocument/2006/math}oMath'));assert count==equations
  reader=PdfReader(pdf);assert len(reader.pages)==pages
  objects=set()
  for page in reader.pages:
   for ref in page['/Resources'].get('/XObject',{}).get_object().values():
    if ref.get_object().get('/Subtype')=='/Image':objects.add((ref.idnum,ref.generation))
  assert len(objects)==images,(stem,len(objects))
  documents[stem]=dict(pdf_pages=pages,native_equations=count,unique_pdf_image_objects=len(objects),docx_sha256=sha(doc),pdf_sha256=sha(pdf))
 text=(PAPER/'manuscript.md').read_text(encoding='utf-8');body,bib=text.split('## REFERENCES',1)
 citations=list(dict.fromkeys(map(int,re.findall(r'\[(\d+)\]',body))));assert citations==list(range(1,19)) and len(re.findall(r'(?m)^\[\d+\]',bib))==18
 assert all(s in body for s in ['0.810','0.827','754','691','nc','ρc','α'])
 dump(PAPER/'delivery_status.json',dict(current=True,revision=4,documents=documents,validation=validation,visual_review='All 7 article and 15 supplement pages reviewed; unchanged final pages pixel-matched to reviewed render',pdf_math='75 native Word equations retained; same-source visible PDF equation fallback',sequence_transfer='20 complete frozen runs; same archaeological site',established_system_baseline='not completed',sequence5_order_repeats='not run',git='paused; no mutations or publication'))
 dump(PAPER/'visual_review.json',dict(passed=True,article_pages=7,supplement_pages=15,method='Every page reviewed; final changed pages reviewed anew; unchanged pages pixel-compared',rendered_pages={p.relative_to(ROOT).as_posix():sha(p) for kind in ['main','supp'] for p in sorted((ROOT/'qa'/('v4_final_'+kind)).glob('page-*.png'))}))
 assert sha(BASE)=='06c9c0076de78d802458a0c298ec9d68bbd012c5e8fed99a5624009d974fa8ed','Unexpected historical base'
 overrides={}
 def addtree(folder,prefix,ok):
  for p in folder.rglob('*'):
   if p.is_file() and ok(p):overrides[prefix+'/'+p.relative_to(folder).as_posix()]=p
 addtree(PAPER,'IEEE_AQUALOC_Revision4',lambda p:not p.name.startswith('~$') and p.name!='package_audit.json')
 addtree(RESULTS,'sequence5_results',lambda p:p.suffix in {'.csv','.json','.md','.txt','.yaml'} and 'cache' not in p.relative_to(RESULTS).parts and 'progress' not in p.name and '.partial.' not in p.name)
 addtree(ROOT/'revision','revision',lambda p:p.suffix in {'.py','.json','.cpp'})
 addtree(ROOT/'source','source',lambda p:p.suffix in {'.py','.md','.ps1'})
 for p in Path(r'C:\Users\Raaghav\Downloads\archaeo_sequence_5_raw_data\raw_data').glob('*.csv'):overrides['original_sequence5_numeric/raw_data/'+p.name]=p
 assert len([p for p in overrides if p.startswith('original_sequence5_numeric/')])==4
 overrides['WORK_STATUS.md']=ROOT/'PAUSED_WORK.md'
 manifest=[]
 def put(z,name,data):
  z.writestr(name,data);manifest.append(dict(path=name,bytes=len(data),sha256=hashlib.sha256(data).hexdigest()))
 with zipfile.ZipFile(BASE) as old,zipfile.ZipFile(PACKAGE,'w',zipfile.ZIP_DEFLATED,compresslevel=6) as new:
  for name in old.namelist():
   if name not in overrides and name not in {'README.md','MANIFEST.csv'} and not name.endswith('/package_audit.json'):put(new,name,old.read(name))
  for name,path in sorted(overrides.items()):put(new,name,path.read_bytes())
  put(new,'README.md',(PAPER/'README_FINDINGS.md').read_bytes())
  buf=io.StringIO(newline='');w=csv.DictWriter(buf,fieldnames=['path','bytes','sha256']);w.writeheader();w.writerows(sorted(manifest,key=lambda r:r['path']));new.writestr('MANIFEST.csv',buf.getvalue())
 print('Packed',len(manifest)+1,'files; verifying CRC and all payload hashes',flush=True)
 with zipfile.ZipFile(PACKAGE) as z:
  assert z.testzip() is None;assert len(z.namelist())==len(set(z.namelist()))==len(manifest)+1
  assert not any('/.git/' in n or '.partial.' in n or '/seed_runs/' in n or 'transcript' in n.lower() for n in z.namelist())
  for r in manifest:
   data=z.read(r['path']);assert len(data)==r['bytes'] and hashlib.sha256(data).hexdigest()==r['sha256'],r['path']
 audit=dict(file=str(PACKAGE),files=len(manifest)+1,archive_bytes=PACKAGE.stat().st_size,sha256=sha(PACKAGE),crc='PASS',all_payload_sha256='PASS',current_checks=3054,complete_sequence5_runs=20,git_push=False)
 dump(PAPER/'package_audit.json',audit);print(json.dumps(audit,indent=2),flush=True)

if __name__=='__main__':main()
