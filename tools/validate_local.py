"""Local structural precheck. NOT a substitute for organizer validator."""
import argparse,csv,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from src.core import PATTERN
def main():
 p=argparse.ArgumentParser();p.add_argument('--dataset',default='dataset');args=p.parse_args()
 root=Path(args.dataset);issues=[];types={};count=0;path=root/'meta.csv'
 if not path.exists():raise SystemExit(f'No {path}')
 expected={'image','plate_num','plate_type','bbox','quad','is_vehicle','is_synthetic','source','license','conditions'}
 with path.open(encoding='utf-8',newline='') as f:
  r=csv.DictReader(f,delimiter=';')
  if not expected.issubset(set(r.fieldnames or [])):issues.append('Missing columns: '+str(expected-set(r.fieldnames or [])))
  for n,row in enumerate(r,2):
   count+=1;typ=row['plate_type'];types[typ]=types.get(typ,0)+1
   if typ not in ('type1','type1a','type1b','other'):issues.append(f'row {n}: type')
   if typ!='other' and not PATTERN.fullmatch(row['plate_num']):issues.append(f'row {n}: number')
   if not (root/row['image']).is_file():issues.append(f'row {n}: missing image')
   for field,sz in [('bbox',4),('quad',8)]:
    try:
     values=list(map(float,row[field].split(',')))
     if len(values)!=sz:issues.append(f'row {n}: {field} size')
    except (TypeError,ValueError):issues.append(f'row {n}: {field} invalid')
   if row['is_vehicle'] not in ('0','1') or row['is_synthetic'] not in ('0','1'):issues.append(f'row {n}: flags')
   if not row['source'] or not row['license']:issues.append(f'row {n}: provenance')
   if len(issues)>100:break
 print('Entries:',count,'Types:',types,'Errors:',len(issues));print('\n'.join(issues[:20]));sys.exit(bool(issues))
if __name__=='__main__':main()
