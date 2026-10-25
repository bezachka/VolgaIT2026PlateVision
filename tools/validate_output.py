import argparse,csv,re,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from src.core import PATTERN
p=argparse.ArgumentParser();p.add_argument('file');a=p.parse_args()
with open(a.file,encoding='utf-8',newline='') as f:
 r=csv.DictReader(f,delimiter=';');required=['image','plate_num','plate_type','confidence']
 assert r.fieldnames==required, f'Columns must be {required}'
 rows=list(r)
 for i,row in enumerate(rows,2):
  assert row['image']==Path(row['image']).name,f'row {i}: filename'
  assert row['plate_type'] in ['type1','type1a','type1b','other'],f'row {i}: class'
  assert PATTERN.fullmatch(row['plate_num']) or re.fullmatch(r'[A-Z0-9#]{8,9}',row['plate_num']),f'row {i}: plate'
  assert 0<=float(row['confidence'])<=1,f'row {i}: confidence'
print('CSV valid; rows:',len(rows))
