from pathlib import Path
import csv, json, re, shutil, hashlib, zipfile, math, copy
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from docx import Document
from docx.shared import Inches, Pt, RGBColor
from docx.enum.section import WD_SECTION_START
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from lxml import etree
from equation_math import mathml
XSL=etree.XSLT(etree.parse(r'C:\Program Files\Microsoft Office\root\Office16\MML2OMML.XSL'))

ROOT=Path(__file__).resolve().parents[1]
DATA=Path(r'C:\Users\Raaghav\Downloads\archaeo_sequence_6_raw_data')
OUT=ROOT/'IEEE_AQUALOC'; OUT.mkdir(exist_ok=True)
FIG=OUT/'figures'; FIG.mkdir(exist_ok=True)
EVID=OUT/'evidence'; EVID.mkdir(exist_ok=True)
AUTHOR='B. Raaghavan Mookiah'
TITLE='From Feature Correspondence to Trajectory Reliability in Underwater Visual Odometry'
SUBTITLE='A calibrated study of contrast enhancement on AQUALOC Sequence 06'
TABLE_WIDTHS={}
KEEP_TABLES=set()
SUPPLEMENT_TITLE='Detailed Derivations for Underwater Visual Odometry Evaluation'
def rows(p):
    with open(p,encoding='utf-8-sig',newline='') as f:return list(csv.DictReader(f))
M=rows(ROOT/'evidence_audit/recomputed_mutual_operating_point.csv')
C=rows(ROOT/'evidence_audit/recomputed_calibrated_operating_point.csv')
V=rows(DATA/'paper_research_draft/vo_full_sequence_v3/vo_sequence6_orb_clahe_v3.csv')
S=json.loads((DATA/'paper_research_draft/full_sequence_trajectory_evaluation/summary.json').read_text())
BS=rows(DATA/'paper_research_draft/submission_final_evidence/bootstrap_sensitivity/bootstrap_block_sensitivity.csv')
def metric(det,pre,key):return float(next(r for r in M if r['detector']==det and r['preprocessing']==pre)[key])

FONT=r'C:\Windows\Fonts\times.ttf'; BOLD=r'C:\Windows\Fonts\timesbd.ttf'
def font(n,b=False):return ImageFont.truetype(BOLD if b else FONT,n)
COL=['#606C79','#1F5F99','#BF6326']
def canvas(title,subtitle='',h=1200):
    im=Image.new('RGB',(1800,h),'white');d=ImageDraw.Draw(im)
    d.text((95,35),title,font=font(49,True),fill='black')
    if subtitle:d.text((95,100),subtitle,font=font(32),fill='#333333')
    return im,d
def savefig(im,name):im.save(FIG/(name+'.png'),dpi=(400,400))
def barchart():
    im,d=canvas('Reference overlap discrimination','Mutual matching | 2 pixel Sampson threshold | count score')
    for panel,key,label in [(0,'f1','F1'),(1,'average_precision_count','Average precision')]:
        x0=125+panel*865;y0=930;w=710;h=680
        for j in range(6):
            y=y0-j*h/5;d.line((x0,y,x0+w,y),fill='#DDDDDD',width=2);d.text((x0-65,y-18),f'{j/5:.1f}',font=font(30),fill='black')
        d.text((x0,y0-h-65),label,font=font(38,True),fill='black')
        for i,det in enumerate(['ORB','AKAZE_MLDB','SIFT']):
            for k,pre in enumerate(['raw','clahe','adaptive_v2']):
                val=metric(det,pre,key);x=x0+45+i*230+k*55
                d.rectangle((x,y0-h*val,x+45,y0),fill=COL[k])
            d.text((x0+30+i*230,y0+20),'AKAZE' if i==1 else det,font=font(31),fill='black')
    for i,l in enumerate(['Raw','Fixed CLAHE','Adaptive v2']):
        x=300+i*480;d.rectangle((x,1070,x+45,1095),fill=COL[i]);d.text((x+65,1060),l,font=font(35),fill='black')
    savefig(im,'01_matching')
def tracking():
    im,d=canvas('Accepted motion across the sequence','Fresh accepted updates; rejected rows do not add measured displacement')
    x0,y0,w,h=150,910,1500,680
    cum=np.cumsum([int(r['tracking_success']) for r in V])
    for k in range(5):
        y=y0-k*h/4;d.line((x0,y,x0+w,y),fill='#DDDDDD',width=2);d.text((35,y-20),str(k*300),font=font(31),fill='black')
    pts=[(x0+i/3384*w,y0-float(v)/1200*h) for i,v in enumerate(cum)]
    d.line(pts,fill=COL[1],width=7)
    for k in [0,600,1200,1800,2400,3000,3384]:
        x=x0+k/3384*w;d.text((x-35,y0+30),str(k),font=font(31),fill='black')
    d.text((700,1010),'Image frame index',font=font(38),fill='black')
    d.text((150,160),'Cumulative accepted updates',font=font(35),fill='black')
    d.text((1160,260),'1,188 / 3,384 = 35.11%',font=font(36,True),fill=COL[1])
    savefig(im,'02_tracking')
def errors():
    im,d=canvas('Reconstruction relative position errors','One Sim(3) fit on 36 earlier poses; 24 later poses held out')
    rr=rows(DATA/'paper_research_draft/full_sequence_trajectory_evaluation/fresh_pose_errors.csv')
    x0,y0,w,h=145,910,1500,670
    for k in range(7):
        y=y0-k*h/6;d.line((x0,y,x0+w,y),fill='#DDDDDD',width=2);d.text((65,y-20),str(k),font=font(31),fill='black')
    for r in rr:
        x=x0+int(r['frame'])/3384*w;y=y0-float(r['error_colmap_units'])/6*h
        d.ellipse((x-9,y-9,x+9,y+9),fill=COL[1] if r['split']=='train' else COL[2])
    for k in [0,600,1200,1800,2400,3000]:d.text((x0+k/3384*w-20,y0+22),str(k),font=font(31),fill='black')
    d.text((145,160),'Position error in reconstruction units',font=font(35),fill='black')
    d.text((1180,300),'Test RMSE 4.089',font=font(36,True),fill=COL[2])
    d.text((540,1050),'Blue: alignment poses     Orange: held out poses',font=font(35),fill='black')
    savefig(im,'03_errors')
def sensitivity():
    im,d=canvas('Bootstrap sensitivity of count AP','Adaptive minus fixed CLAHE | paired endpoint block resampling')
    x0,y0,w,h=310,940,1310,700;lo=-.09;hi=.02
    def xx(v):return x0+(v-lo)/(hi-lo)*w
    for v in [-.08,-.06,-.04,-.02,0,.02]:
        x=xx(v);d.line((x,200,x,y0),fill='#BBBBBB' if v==0 else '#E5E5E5',width=3);d.text((x-42,y0+20),f'{v:.2f}',font=font(29),fill='black')
    for i,det in enumerate(['ORB','AKAZE_MLDB','SIFT']):
        d.text((20,250+i*220),'AKAZE' if i==1 else det,font=font(37,True),fill='black')
        for j,B in enumerate([1,5,10,20]):
            r=next(r for r in BS if r['detector']==det and r['scope']=='all_pairs' and r['metric']=='ap_count' and int(r['block_keyframes'])==B)
            y=235+i*220+j*47;a=xx(float(r['ci95_low']));b=xx(float(r['ci95_high']));p=xx(float(r['observed_adaptive_minus_fixed']))
            d.line((a,y,b,y),fill=COL[i],width=5);d.line((a,y-9,a,y+9),fill=COL[i],width=3);d.line((b,y-9,b,y+9),fill=COL[i],width=3);d.ellipse((p-7,y-7,p+7,y+7),fill=COL[i]);d.text((220,y-18),str(B),font=font(28),fill='black')
    d.text((120,160),'Block size',font=font(29),fill='black')
    d.text((420,1060),'400 replicates per setting; heuristic 95% percentile intervals',font=font(34),fill='black')
    savefig(im,'04_sensitivity')
def pipeline():
    im,d=canvas('Separate evaluation stages','Correspondence support and trajectory reliability answer different questions',h=1090)
    labels=[('Image processing','Raw / fixed CLAHE / adaptive v2'),('Reference pair scoring','ORB / AKAZE / SIFT; F1 and AP'),('Two view pose candidates','Descriptors + Lucas-Kanade; separate validation'),('Accepted monocular updates','Quality selection; arbitrary gap-based steps'),('Conditional trajectory evaluation','Fresh reference poses; earlier fit / later test')]
    for i,(a,b) in enumerate(labels):
        y=190+i*165;d.rounded_rectangle((145,y,1650,y+122),radius=15,fill='#F0F3F6',outline='#777777',width=2)
        d.text((180,y+12),a,font=font(39,True),fill='black');d.text((180,y+66),b,font=font(33),fill='black')
        if i<4:d.line((900,y+123,900,y+157),fill='black',width=4);d.polygon([(890,y+146),(910,y+146),(900,y+160)],fill='black')
    savefig(im,'00_pipeline')

def prepare():
    for fn in ['audit_report.md','recomputed_mutual_operating_point.csv','recomputed_calibrated_operating_point.csv','reproducibility_manifest.json']:
        shutil.copy2(ROOT/'evidence_audit'/fn,EVID/fn)
    for rel in ['paper_research_draft/full_sequence_trajectory_evaluation/summary.json','paper_research_draft/full_sequence_trajectory_evaluation/fresh_pose_errors.csv','paper_research_draft/submission_final_evidence/bootstrap_sensitivity/bootstrap_block_sensitivity.csv','paper_research_draft/submission_final_evidence/bootstrap_sensitivity/bootstrap_methodology.json','paper_research_draft/vo_selection_comparison/comparison_summary.csv','paper_research_draft/cheirality_threshold_comparison/comparison_summary.csv']:
        shutil.copy2(DATA/rel,EVID/Path(rel).name if 'comparison_summary' not in rel else EVID/(Path(rel).parent.name+'.csv'))
    for f in [pipeline,barchart,tracking,errors,sensitivity]:f()

class Builder:
    def __init__(self,main):
        self.main=main;self.doc=Document();self.eqn=0;self.fign=0;self.tabn=0
        sec=self.doc.sections[0];sec.page_width=Inches(8.5);sec.page_height=Inches(11)
        sec.top_margin=Inches(.7);sec.bottom_margin=Inches(.7);sec.left_margin=sec.right_margin=Inches(.7)
        for sn in ['Normal','Title','Subtitle','Heading 1','Heading 2','Caption']:
            st=self.doc.styles[sn];st.font.name='Times New Roman';st.font.color.rgb=RGBColor(0,0,0)
            st._element.get_or_add_rPr().rFonts.set(qn('w:eastAsia'),'Times New Roman')
        normal=self.doc.styles['Normal'];normal.font.size=Pt(10 if main else 11);normal.paragraph_format.space_after=Pt(4 if main else 6);normal.paragraph_format.line_spacing=1
        normal.paragraph_format.widow_control=True
        self.doc.styles['Title'].font.size=Pt(21 if main else 20);self.doc.styles['Title'].font.bold=False
        for sn,sz in [('Heading 1',10 if main else 13),('Heading 2',10 if main else 11)]:
            st=self.doc.styles[sn];st.font.size=Pt(sz);st.font.bold=True;st.paragraph_format.space_before=Pt(10);st.paragraph_format.space_after=Pt(5);st.paragraph_format.keep_with_next=True
        self.doc.styles['Caption'].font.size=Pt(8.5 if main else 9)
        self.doc.core_properties.author=AUTHOR;self.doc.core_properties.title=TITLE if main else SUPPLEMENT_TITLE
        p=self.doc.add_paragraph(TITLE if main else SUPPLEMENT_TITLE,'Title');p.alignment=WD_ALIGN_PARAGRAPH.CENTER
        p=self.doc.add_paragraph(SUBTITLE);p.alignment=WD_ALIGN_PARAGRAPH.CENTER
        p=self.doc.add_paragraph(AUTHOR);p.alignment=WD_ALIGN_PARAGRAPH.CENTER
        p.paragraph_format.space_after=Pt(12)
        if main:
            sec=self.doc.add_section(WD_SECTION_START.CONTINUOUS);cols=sec._sectPr.find(qn('w:cols'));cols.set(qn('w:num'),'2');cols.set(qn('w:space'),'360')
        foot=self.doc.sections[0].footer.paragraphs[0];foot.alignment=WD_ALIGN_PARAGRAPH.CENTER
        fld=OxmlElement('w:fldSimple');fld.set(qn('w:instr'),'PAGE');foot._p.append(fld)
    def p(self,s,style=None):
        p=self.doc.add_paragraph(style=style);p.alignment=WD_ALIGN_PARAGRAPH.JUSTIFY if style is None else WD_ALIGN_PARAGRAPH.LEFT
        for i,t in enumerate(re.split(r'\*\*(.*?)\*\*',s)):
            r=p.add_run(t);r.bold=i%2==1
        if style is None:p.paragraph_format.first_line_indent=Inches(.12 if self.main else 0)
        return p
    def eq(self,s):
        self.eqn+=1;p=self.doc.add_paragraph();p.alignment=WD_ALIGN_PARAGRAPH.CENTER
        p.paragraph_format.space_before=Pt(3);p.paragraph_format.space_after=Pt(6)
        result=XSL(etree.fromstring(mathml(s).encode('utf-8')))
        om=copy.deepcopy(result.xpath('//*[local-name()="oMath"]')[0]);p._p.append(om)
        for run in om.iter(qn('m:r')):
            wr=run.find(qn('w:rPr'))
            if wr is None:
                wr=OxmlElement('w:rPr');run.insert(1 if run.find(qn('m:rPr')) is not None else 0,wr)
            sz=OxmlElement('w:sz');sz.set(qn('w:val'),'18' if self.main else '22');wr.append(sz)
        p.add_run('  ('+('' if self.main else 'S')+str(self.eqn)+')').font.size=Pt(8 if self.main else 10)
    def table(self,lines):
        data=[[c.strip() for c in l.strip().strip('|').split('|')] for l in lines]
        data=[r for r in data if not all(re.fullmatch(r'[:\- ]+',c or '-') for c in r)]
        n=len(data[0]);width=3.4 if self.main else 7.1
        weights=[max(5,min(27,max(len(r[j]) for r in data))) for j in range(n)]
        widths=[width*w/sum(weights) for w in weights]
        if self.main and self.pending_caption in TABLE_WIDTHS:widths=TABLE_WIDTHS[self.pending_caption]
        self.tabn+=1;p=self.p('TABLE '+str(self.tabn)+'  '+self.pending_caption,'Caption');p.alignment=WD_ALIGN_PARAGRAPH.CENTER;p.paragraph_format.keep_with_next=True
        t=self.doc.add_table(rows=0,cols=n);t.autofit=False
        for c,w in zip(t.columns,widths):c.width=Inches(w)
        for i,row in enumerate(data):
            cells=t.add_row().cells
            for j,(cell,txt) in enumerate(zip(cells,row)):
                cell.width=Inches(widths[j]);cell.vertical_alignment=1;p=cell.paragraphs[0];p.paragraph_format.space_after=Pt(1);p.paragraph_format.line_spacing=1
                p.alignment=WD_ALIGN_PARAGRAPH.LEFT if j==0 else WD_ALIGN_PARAGRAPH.CENTER
                if self.pending_caption in KEEP_TABLES and i<len(data)-1:p.paragraph_format.keep_with_next=True
                r=p.add_run(txt);r.font.size=Pt(7.5 if self.main else 9);r.bold=i==0
                tc=cell._tc.get_or_add_tcPr();bd=OxmlElement('w:tcBorders')
                for edge in ['top','bottom','left','right']:
                    e=OxmlElement('w:'+edge);e.set(qn('w:val'),'single');e.set(qn('w:sz'),'4');e.set(qn('w:color'),'D9D9D9');bd.append(e)
                tc.append(bd);pad=OxmlElement('w:tcMar')
                for edge in ['top','bottom','left','right']:
                    e=OxmlElement('w:'+edge);e.set(qn('w:w'),'55');e.set(qn('w:type'),'dxa');pad.append(e)
                tc.append(pad)
                if i==0 or i%2==0:
                    sh=OxmlElement('w:shd');sh.set(qn('w:fill'),'E7EDF1' if i==0 else 'F6F7F8');tc.append(sh)
            tr=t.rows[-1]._tr.get_or_add_trPr();tr.append(OxmlElement('w:cantSplit'))
            if i==0:tr.append(OxmlElement('w:tblHeader'))
        self.doc.add_paragraph().paragraph_format.space_after=Pt(1)
    def fig(self,s):
        name,caption=s.split('|',1);self.fign+=1;p=self.doc.add_paragraph();p.alignment=WD_ALIGN_PARAGRAPH.CENTER;p.paragraph_format.keep_with_next=True
        p.add_run().add_picture(str(FIG/(name+'.png')),width=Inches(3.4 if self.main else 6.3))
        p=self.p('Fig. '+str(self.fign)+'. '+caption,'Caption');p.paragraph_format.keep_together=True
    def build(self,text,stem):
        lines=text.strip().splitlines();i=0
        while i<len(lines):
            s=lines[i].strip();i+=1
            if not s:continue
            if s.startswith('@@'):self.pending_caption=s[2:].strip();continue
            if s.startswith('|'):
                tab=[s]
                while i<len(lines) and lines[i].strip().startswith('|'):tab.append(lines[i]);i+=1
                self.table(tab);continue
            if s.startswith('!FIG '):self.fig(s[5:]);continue
            if s.startswith('$$'):self.eq(s.strip('$ ').strip());continue
            if s.startswith('### '):self.p(s[4:],'Heading 2');continue
            if s.startswith('## '):
                p=self.p(s[3:],'Heading 1');p.alignment=WD_ALIGN_PARAGRAPH.CENTER if self.main else WD_ALIGN_PARAGRAPH.LEFT;continue
            if s.startswith('[') and re.match(r'\[\d+\]',s):
                p=self.p(s);p.alignment=WD_ALIGN_PARAGRAPH.LEFT;p.paragraph_format.first_line_indent=Inches(-.15);p.paragraph_format.left_indent=Inches(.15)
                for r in p.runs:r.font.size=Pt(8.5 if self.main else 9)
            else:self.p(s)
        for tree in [self.doc._element,self.doc.styles._element]:
            for e in list(tree.iter(qn('w:pBdr'))):e.getparent().remove(e)
            for e in tree.iter(qn('w:rFonts')):
                for a in ['asciiTheme','hAnsiTheme','eastAsiaTheme','cstheme','csTheme']:e.attrib.pop(qn('w:'+a),None)
        self.doc.save(OUT/(stem+'.docx'));(OUT/(stem+'.md')).write_text('# '+self.doc.core_properties.title+'\n\n'+text,encoding='utf-8')
        print(stem,'equations',self.eqn,'tables',self.tabn,'figures',self.fign,flush=True)

if __name__=='__main__':
    prepare()
    Builder(True).build((ROOT/'source/manuscript.md').read_text(encoding='utf-8'),'ieee_manuscript')
    Builder(False).build((ROOT/'source/supplement.md').read_text(encoding='utf-8'),'detailed_derivations')
