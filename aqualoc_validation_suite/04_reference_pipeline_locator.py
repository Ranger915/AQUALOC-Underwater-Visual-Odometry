#!/usr/bin/env python3
"""Locate reference-geometry and mutual-matching artifacts; inspect schemas.
Does not mislabel RANSAC consistency as independently referenced precision/F1/AP.
"""
import argparse,csv,json
from pathlib import Path

KEYWORDS=('calibrat','mutual','sampson','colmap','reference','groundtruth','bootstrap','feature','match')
def main():
    p=argparse.ArgumentParser();p.add_argument('--root',required=True);p.add_argument('--output',required=True);args=p.parse_args()
    root=Path(args.root);out=Path(args.output);out.mkdir(parents=True,exist_ok=True)
    catalog=[]
    for file in root.rglob('*'):
        if not file.is_file() or not any(k in str(file).lower() for k in KEYWORDS):continue
        if file.suffix.lower() not in ('.py','.csv','.json','.yaml','.yml','.gz','.txt'):continue
        r={'path':str(file),'bytes':file.stat().st_size,'extension':file.suffix.lower()}
        if file.suffix.lower()=='.csv':
            try:
                with file.open(newline='',encoding='utf-8-sig') as f:
                    rd=csv.DictReader(f);r['columns']=rd.fieldnames;r['first_row']=next(rd,None)
            except Exception as e:r['preview_error']=str(e)
        catalog.append(r)
    dest=out/'reference_pipeline_inventory.json';dest.write_text(json.dumps(catalog,indent=2),encoding='utf-8')
    print('[SAVED]',dest)
    print('Possible calibrated/mutual/reference script candidates:')
    for r in catalog:
        if r['extension']=='.py' and any(x in r['path'].lower() for x in ('calibrat','mutual','sampson','colmap','reference')):print(r['path'])
    print('Python scripts seen:',sum(r['extension']=='.py' for r in catalog),'CSV files:',sum(r['extension']=='.csv' for r in catalog))
    print('DO NOT calculate reference F1/AP/Sampson from frontend CSV alone: no match coordinates or reference pair labels.')

if __name__=='__main__':main()
