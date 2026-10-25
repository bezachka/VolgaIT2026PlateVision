"""Train YOLO detector; requires pip install ultralytics and pretrained local checkpoint.
Never downloads weights at runtime unless user explicitly provides local base checkpoint.
"""
import argparse,csv,hashlib,shutil
from pathlib import Path
import yaml

def main():
    p=argparse.ArgumentParser();p.add_argument('--dataset',default='dataset');p.add_argument('--base-weights',required=True,help='Existing locally available YOLO .pt checkpoint');p.add_argument('--epochs',type=int,default=35);p.add_argument('--batch',type=int,default=8);p.add_argument('--device',default='cpu');p.add_argument('--project',default='runs/plates');args=p.parse_args()
    from ultralytics import YOLO
    root=Path(args.dataset).resolve();out=root/'yolo';out.mkdir(exist_ok=True)
    counts={'train':0,'val':0,'test':0}
    with (root/'meta.csv').open(encoding='utf-8',newline='') as f:
        for row in csv.DictReader(f,delimiter=';'):
            source=root/row['image'];lab=root/'labels'/(source.stem+'.txt')
            if not source.exists() or not lab.exists():continue
            # split by plate identity to reduce exact string leakage
            key=row['plate_num'] if row['plate_type']!='other' else source.stem
            hv=int(hashlib.sha256(key.encode()).hexdigest()[:8],16)%10
            split='train' if hv<8 else ('val' if hv==8 else 'test')
            for sub,src in [('images',source),('labels',lab)]:
                d=out/sub/split;d.mkdir(parents=True,exist_ok=True)
                target=d/src.name
                if not target.exists():shutil.copy2(src,target)
            counts[split]+=1
    cfg={'path':str(out.resolve()),'train':'images/train','val':'images/val','test':'images/test','names':{0:'type1',1:'type1a',2:'type1b',3:'other'}}
    cfgpath=out/'data.yaml';cfgpath.write_text(yaml.safe_dump(cfg,allow_unicode=True),encoding='utf-8')
    print('Split counts:',counts)
    YOLO(args.base_weights).train(data=str(cfgpath),epochs=args.epochs,imgsz=640,batch=args.batch,device=args.device,project=args.project,workers=2)
if __name__=='__main__':main()