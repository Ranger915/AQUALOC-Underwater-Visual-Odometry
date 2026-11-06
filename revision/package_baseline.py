"""Verify and extend the local evidence archive without touching Git."""
from common import *
import zipfile,io,re,shutil
import pandas as pd
from pypdf import PdfReader
from lxml import etree
PAPER=ROOT/'IEEE_AQUALOC_Revision5'
BASE=ROOT/'AQUALOC_Complete_Findings_Revision4.zip'
PACKAGE=ROOT/'AQUALOC_Complete_Findings_Revision5.zip'

def main():
 validations=[json.loads(p.read_text()) for p in [ROOT/'sequence5_results/validation_summary.json',ROOT/'sequence5_results/order_runs/validation_summary.json']+[ROOT/f'baseline_results/evaluation_sequence{s}/validation_summary.json' for s in [6,5]]]
 assert all(v['passed'] for v in validations)
 assert [v['checks'] for v in validations]==[3054,235,293,325]
 checks=sum(v['checks'] for v in validations)
 markers=list((ROOT/'baseline_results').glob('sequence*/done.json'));assert len(markers)==18
 assert all(json.loads(p.read_text())['completed'] and json.loads(p.read_text())['frames']==(3983 if 'sequence5' in p.parent.name else 3385) for p in markers)
 native=pd.concat([pd.read_csv(ROOT/f'baseline_results/evaluation_sequence{s}/native_all_fit_metrics.csv') for s in [6,5]])
 native=native[native.scope=='paired_native_all_fit'];scores=native.pivot(index=['sequence','preprocessing','trial'],columns='method',values='rmse_m')
 assert len(scores)==18 and (scores.ORB_SLAM3<scores.pairwise_ORB).all()
 causal=pd.concat([pd.read_csv(ROOT/f'baseline_results/evaluation_sequence{s}/causal_summary.csv') for s in [6,5]])
 causal=causal[causal['mask']=='all_reference'];wins=0
 for _,g in causal.groupby('run'):
  assert len(g)==2;wins+=int(g[g.method=='ORB_SLAM3'].rmse_m.iloc[0]<g[g.method!='ORB_SLAM3'].rmse_m.iloc[0])
 assert wins==17
 documents={}
 for stem,kind,pages,equations in [('ieee_manuscript','main_final',9,3),('detailed_derivations','supp',17,72)]:
  source=ROOT/'qa'/('v5_'+kind);shutil.copy2(source/(stem+'.pdf'),PAPER/(stem+'.pdf'))
  reader=PdfReader(PAPER/(stem+'.pdf'));assert len(reader.pages)==pages
  with zipfile.ZipFile(PAPER/(stem+'.docx')) as z:tree=etree.fromstring(z.read('word/document.xml'))
  assert len(tree.findall('.//{http://schemas.openxmlformats.org/officeDocument/2006/math}oMath'))==equations
  assert len(list(source.glob('page-*.png')))==pages
  documents[stem]=dict(pages=pages,native_equations=equations,docx_sha256=sha(PAPER/(stem+'.docx')),pdf_sha256=sha(PAPER/(stem+'.pdf')))
 body,bib=(PAPER/'manuscript.md').read_text(encoding='utf-8').split('## REFERENCES',1)
 assert list(dict.fromkeys(map(int,re.findall(r'\[(\d+)\]',body))))==list(range(1,20))
 assert len(re.findall(r'(?m)^\[\d+\]',bib))==19
 dump(PAPER/'delivery_status.json',dict(revision=5,current=True,documents=documents,checks=checks,validation_groups=[v['checks'] for v in validations],baseline_runs=18,sequence5_new_order_runs=16,all_fit_comparisons_lower=18,causal_comparisons_lower=17,visual_review='Every page reviewed; unchanged final article pages pixel-matched',git='paused; no Git mutations'))
 dump(PAPER/'visual_review.json',dict(passed=True,article_pages=9,supplement_pages=17,rendered_pages={p.relative_to(ROOT).as_posix():sha(p) for folder in ['v5_main_final','v5_supp'] for p in sorted((ROOT/'qa'/folder).glob('page-*.png'))},math='75 native Word equations retained; same-source visible PDF fallback'))
 status='''# Local Revision5 completed

Current documents: IEEE_AQUALOC_Revision5/. Current baseline evidence: baseline_results/evaluation_sequence5/ and evaluation_sequence6/. Sequence 05 order evidence: sequence5_results/order_runs/.

All eighteen full monocular ORB-SLAM3 runs completed; sixteen additional Sequence 05 ORB order runs completed. Validation groups: 3,054 existing transfer checks, 235 order checks, 618 baseline checks; 3,907 total. Article: nine pages. Supplement: seventeen pages. Editable Word retains 75 native equations; all final PDF pages visually reviewed.

ORB-SLAM3 has lower final-map all-fit error in 18/18 matched native comparisons and lower online chronological error in 17/18. Raw runs vary substantially; two Sequence 06 enhancement runs form two map epochs, with 80.9% and 82.5% tracking coverage. Coverage, exact support, and online versus optimized-map results remain explicit. Both sequences share one archaeological site and reconstruction-derived references. Independent physical truth, sensor fusion, and broad site generalization remain outside this evidence.

GitHub remains paused. Do not stage, commit, amend, tag, or push until explicitly resumed. The future repository update is code plus README only, excluding findings and transcripts. Never publish this complete local findings ZIP as that update.
'''
 (ROOT/'PAUSED_WORK.md').write_text(status,encoding='utf-8')
 (ROOT/'sequence5_results/STATUS.md').write_text(status,encoding='utf-8')
 log=ROOT/'baseline_results/WORK_LOG.md';log.write_text(log.read_text(encoding='utf-8').split('\n\n## Completion on 7 November 2026')[0]+'\n\n## Completion on 7 November 2026\n\n'+status,encoding='utf-8')
 fixes=(ROOT/'IEEE_AQUALOC_Revision4/FIXES_AND_FINDINGS.md').read_text(encoding='utf-8')
 fixes=fixes.replace('Revision 4:','Revision 5:').replace('An established ORB-SLAM3/UVS-class baseline has not been completed.','Eighteen upstream monocular ORB-SLAM3 replays now complete the declared system comparison. UVS was not run.')
 fixes=fixes.replace('no Sequence 05 or AKAZE/SIFT order-repeat claim is made.','Sequence 05 repeats the same four ORB order interventions; AKAZE/SIFT remain unrepeated.')
 fixes=fixes.replace('The article is seven pages; the detailed derivations are fifteen pages.','The article is nine pages; the detailed derivations are seventeen pages.')
 fixes=fixes.replace('**3,054 numerical and implementation checks pass**','**3,907 numerical and implementation checks pass across the transfer, order, and baseline validation groups**')
 fixes=fixes.replace('Not completed: an established odometry baseline, independent physical trajectory reference, sensor fusion, persistent optimized motion estimation, and Sequence 05 correspondence-order repeats.','Completed additions: eighteen ORB-SLAM3 runs and sixteen Sequence 05 ORB order runs. Independent physical trajectory reference, sensor fusion, broader site validation, and persistent optimized estimation inside the pairwise engine remain outside scope.')
 (PAPER/'FIXES_AND_FINDINGS.md').write_text(fixes+'\n\n## Established system and order results\n\n'+status,encoding='utf-8')
 readme='''# AQUALOC local evidence package Revision5

The current article and supplement are in IEEE_AQUALOC_Revision5/ as PDF, editable Word, and Markdown. Earlier revisions are historical. The article has nine pages and the supplement seventeen; Word retains 75 native equations. All final PDF pages were reviewed.

Eighteen full upstream monocular ORB-SLAM3 replays cover both sequences, raw/fixed/adaptive preprocessing, and three fresh processes per setting. Sixteen additional Sequence 05 ORB order runs extend the ordered run to five trials. All 3,907 current checks pass (3,054 transfer, 235 order, 618 baseline).

ORB-SLAM3 has lower final-map all-fit RMSE in all eighteen paired native comparisons and lower online chronological RMSE in seventeen. Sequence 05 raw trial 0 is the chronological exception. Raw final-map scores vary substantially. Two Sequence 06 runs split maps and lose tracking coverage. Scores use reconstruction-derived references from two sequences at one archaeological site. Final-map all-fit scores use future optimization and are distinct from chronological online scores; support and coverage must accompany both.

baseline_results/ contains protocol, calibration, manifests, complete per-frame tracking/poses, all exported nonempty final maps, completion markers, coverage, native metrics, exact chronological fit/test IDs, residuals, paired ranges, and validation records. baseline_tools/ contains the adapter, source patch, build script, dependency versions, compiler flags, and hashes. Upstream ORB-SLAM3 is pinned to 4452a3c4ab75b1cde34e5505a36ec3f9edcdc4c4; Pangolin v0.6 and consistent C++14 are used. No estimator parameter was retuned after the smoke check.

sequence5_results/order_runs/ contains full order outputs and evaluation; the existing transfer evidence and original numerical evidence remain included. Historical README/status files inside older manuscript folders describe those snapshots, not current completion.

Reproduction uses the dependencies documented in IEEE_AQUALOC_Revision4/README_FINDINGS.md for the pairwise work. Evaluate saved baseline outputs with `python revision/evaluate_orbslam3.py 6` then `python revision/evaluate_orbslam3.py 5`. Evaluate saved order outputs with `python revision/evaluate_order_sequence5.py`. Full replays require separately downloaded raw AQUALOC images and the documented Linux ORB-SLAM3 build; scripts retain original local dataset paths and WSL distribution names, which must be adjusted when relocating. Absolute manifest paths describe original provenance. Source generation uses write_baseline_paper.py, figures_baseline.py, build_baseline_paper.py, and render_math_for_pdf.py --paper IEEE_AQUALOC_Revision5. PDFs use same-source visible math rendering copies because the available renderer omits inline OMML; editable Word keeps native math. Packaging uses package_baseline.py after render and visual review.

MANIFEST.csv records each payload size and SHA-256 except itself. CRC and every payload hash are verified after packing. The external package_audit.json records the archive hash. Raw images, partial runs, smoke attempts, transcripts, runtimes, VHDX, installers, cloned dependency trees, and source archives are excluded from the additions.

GitHub remains paused. This complete findings ZIP must never be used as the future code-only GitHub publication.
'''
 (PAPER/'README_FINDINGS.md').write_text(readme,encoding='utf-8')
 assert sha(BASE)=='42575a9bdad2e23ea5ff39ad4bf40f2a79f251676755dfa5880a2867104a7f03'
 overrides={}
 def addtree(folder,prefix,ok):
  for p in folder.rglob('*'):
   if p.is_file() and ok(p):overrides[prefix+'/'+p.relative_to(folder).as_posix()]=p
 addtree(PAPER,PAPER.name,lambda p:p.name!='package_audit.json' and not p.name.startswith('~$'))
 complete_names={p.parent.name for p in markers}
 addtree(ROOT/'baseline_results','baseline_results',lambda p:p.suffix in {'.csv','.json','.txt','.yaml','.manifest','.md'} and not p.name.startswith('smoke') and (p.parent==ROOT/'baseline_results' or p.relative_to(ROOT/'baseline_results').parts[0] in complete_names|{'evaluation_sequence5','evaluation_sequence6'}))
 addtree(ROOT/'sequence5_results','sequence5_results',lambda p:p.suffix in {'.csv','.json','.md','.txt','.yaml'} and 'cache' not in p.parts and 'progress' not in p.name and '.partial.' not in p.name)
 addtree(ROOT/'revision','revision',lambda p:p.suffix in {'.py','.json','.cpp','.ps1'})
 for name in ['mono_aqualoc.cc','build_orbslam3.sh','build_flags.txt','build_sha256.txt','cpu.json','environment.txt','pangolin_commit.txt','ubuntu_packages.txt','ubuntu.SHA256SUMS','upstream_adapter.patch','upstream_commit.json']:overrides['baseline_tools/'+name]=ROOT/'baseline_tools'/name
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
 print('Packed',len(manifest)+1,'files; verifying',flush=True)
 with zipfile.ZipFile(PACKAGE) as z:
  assert z.testzip() is None and len(z.namelist())==len(set(z.namelist()))==len(manifest)+1
  assert len([n for n in z.namelist() if n.startswith('baseline_results/sequence') and n.endswith('/done.json')])==18
  assert not any(n.startswith('baseline_results/smoke') for n in z.namelist())
  assert not any('/.git/' in n or '.partial.' in n or 'transcript' in n.lower() or n.endswith('.vhdx') for n in z.namelist())
  for r in manifest:
   data=z.read(r['path']);assert len(data)==r['bytes'] and hashlib.sha256(data).hexdigest()==r['sha256'],r['path']
 audit=dict(file=str(PACKAGE),files=len(manifest)+1,archive_bytes=PACKAGE.stat().st_size,sha256=sha(PACKAGE),crc='PASS',all_payload_sha256='PASS',current_checks=checks,baseline_runs=18,git_push=False)
 dump(PAPER/'package_audit.json',audit);print(json.dumps(audit,indent=2),flush=True)

if __name__=='__main__':main()
