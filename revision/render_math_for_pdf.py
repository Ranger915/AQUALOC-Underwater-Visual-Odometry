from common import *
from PIL import Image,ImageDraw,ImageFont
from docx import Document
from docx.shared import Inches
from docx.oxml.ns import qn
import re,copy,argparse
parser=argparse.ArgumentParser();parser.add_argument('--paper',default='IEEE_AQUALOC_Revision');args=parser.parse_args()
paper=ROOT/args.paper;target=ROOT/'qa'/('pdf_render_inputs' if args.paper=='IEEE_AQUALOC_Revision' else args.paper+'_pdf_inputs');target.mkdir(parents=True,exist_ok=True);assets=target/'equations';assets.mkdir(exist_ok=True)
fontfile=r'C:\Windows\Fonts\cambria.ttc'
try:font=ImageFont.truetype(fontfile,64,index=1)
except OSError:font=ImageFont.truetype(fontfile,64)
audit=[]
for src,stem,maximum in [('manuscript.md','ieee_manuscript',3.0),('supplement.md','detailed_derivations',6.5)]:
 source=(paper/src).read_text(encoding='utf-8');equations=[m.strip().rstrip('.') for m in re.findall(r'\$\$(.*?)\$\$',source)];doc=Document(paper/(stem+'_final.docx' if (paper/(stem+'_final.docx')).exists() else stem+'.docx'));nodes=list(doc._element.iter(qn('m:oMath')));assert len(equations)==len(nodes)
 for idx,(eq,node) in enumerate(zip(equations,nodes),1):
  # Rendering fallback only: preserve the exact formula text in visible math
  # images because this portable LibreOffice build omits inline OMML entirely.
  # The distributed editable DOCX retains all native Office equations.
  probe=Image.new('RGB',(1,1),'white');draw=ImageDraw.Draw(probe);box=draw.textbbox((0,0),eq,font=font);w=box[2]-box[0]+32;h=box[3]-box[1]+24;im=Image.new('RGB',(w,h),'white');ImageDraw.Draw(im).text((16-box[0],12-box[1]),eq,font=font,fill='black');path=assets/f'{stem}_{idx:02}.png';im.save(path,dpi=(400,400))
  p=doc.add_paragraph();r=p.add_run();r.add_picture(str(path),width=Inches(min(maximum,w/400)));replacement=copy.deepcopy(r._r);p._p.getparent().remove(p._p);parent=node.getparent();parent.replace(node,replacement)
 doc.save(target/(stem+'.docx'));audit.append(dict(document=stem,equations=len(nodes),editable_source='native OMML retained',pdf_fallback='same formula source rendered as visible Cambria Math text images; compact matrices retain semicolon row notation'))
dump(paper/'pdf_equation_rendering.json',audit);print('Prepared visible equations in PDF rendering copies',flush=True)
