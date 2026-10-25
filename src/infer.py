"""CSV inference for zero/one/multiple plates per image; standalone/offline."""
import argparse,csv,os,time
from pathlib import Path
import cv2
import numpy as np
from .core import detect_candidates,ocr_plate,plate_type,PATTERN

def load_yolo(weights):
    if not weights:return None
    if not Path(weights).is_file():raise FileNotFoundError(f'Weights absent: {weights}')
    try:from ultralytics import YOLO
    except ImportError as e:raise RuntimeError('For --weights install ultralytics locally') from e
    return YOLO(weights)

def predict_image(img,model=None,min_conf=.20):
    if model is None:
        candidates=detect_candidates(img)
    else:
        predictions=model.predict(source=img,conf=min_conf,imgsz=640,verbose=False,device='cpu')[0]
        candidates=[]
        for b in predictions.boxes:
            cls=int(b.cls.item()); label=str(model.names[cls]).lower()
            if label=='other':continue
            x,y,x2,y2=[int(v) for v in b.xyxy[0].tolist()]
            candidates.append((x,y,x2,y2,float(b.conf.item()),label))
    rows=[]; seen=[]
    for c in candidates:
        x,y,x2,y2=c[:4]
        roi=img[max(0,y):min(img.shape[0],y2),max(0,x):min(img.shape[1],x2)]
        if roi.size==0:continue
        typ=c[5] if len(c)>5 and c[5] in ('type1','type1a','type1b') else plate_type(roi)
        number,ocr_conf=ocr_plate(roi,typ=='type1a')
        if not PATTERN.fullmatch(number):continue # avoid arbitrary hallucinations
        conf=round(float(min(1,max(0,ocr_conf*(c[4] if model is not None else .75)))),4)
        if conf<min_conf:continue
        if any(n==number and abs(x-u)<40 and abs(y-v)<30 for n,u,v in seen):continue
        seen.append((number,x,y))
        rows.append((number,typ,conf))
    return rows

def main():
    p=argparse.ArgumentParser(description='Offline Russian plate OCR')
    p.add_argument('--input',required=True,help='directory of JPG/PNG frames')
    p.add_argument('--output',default='predictions.csv')
    p.add_argument('--weights',default=None,help='optional trained YOLO weights')
    p.add_argument('--min-conf',type=float,default=.20)
    args=p.parse_args()
    src=Path(args.input)
    if not src.is_dir():p.error(f'Input directory not found: {src}')
    files=sorted(x for x in src.iterdir() if x.is_file() and x.suffix.lower() in ('.jpg','.jpeg','.png'))
    model=load_yolo(args.weights)
    t=time.perf_counter();n=0
    out=Path(args.output);out.parent.mkdir(parents=True,exist_ok=True)
    with out.open('w',encoding='utf-8',newline='') as f:
        wr=csv.writer(f,delimiter=';');wr.writerow(['image','plate_num','plate_type','confidence'])
        for file in files:
            image=cv2.imdecode(np.fromfile(str(file),dtype=np.uint8),cv2.IMREAD_COLOR)
            if image is None:print(f'WARNING: unreadable image: {file}');continue
            for number,typ,confidence in predict_image(image,model,args.min_conf):
                wr.writerow([file.name,number,typ,f'{confidence:.4f}']);n+=1
    secs=time.perf_counter()-t
    print(f'Images={len(files)} detections={n} CSV={out} mean_ms={secs/max(len(files),1)*1000:.1f} (hardware-dependent)')
if __name__=='__main__':main()