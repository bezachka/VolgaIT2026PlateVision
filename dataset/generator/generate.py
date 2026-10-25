"""Reproducible self-generated synthetic data and YOLO labels; no external assets."""
import argparse,csv,random,math,sys,hashlib
from pathlib import Path
from PIL import Image,ImageDraw,ImageFont,ImageFilter,ImageEnhance
import numpy as np
import cv2
LETTERS='ABEKMHOPCTYX'
CLASSES=['type1','type1a','type1b','other']
HEADER=['image','plate_num','plate_type','bbox','quad','is_vehicle','is_synthetic','source','license','conditions']
def font(size):
    candidates=['/usr/share/fonts/truetype/dejavu/DejaVuSansCondensed-Bold.ttf','/usr/share/fonts/truetype/liberation2/LiberationSans-Bold.ttf','C:/Windows/Fonts/arialbd.ttf']
    for f in candidates:
        if Path(f).exists():return ImageFont.truetype(f,size)
    return ImageFont.load_default()
def gen_number(rng):
    return rng.choice(LETTERS)+''.join(rng.choices('0123456789',k=3))+''.join(rng.choices(LETTERS,k=2))+ rng.choice(['77','50','99','25','716','777','197','178','116','02','74'])
def plate(rng,typ,number):
    w,h=(290,170) if typ=='type1a' else (520,112)
    col=(249,209,40) if typ=='type1b' else (242,242,240)
    img=Image.new('RGB',(w,h),col);d=ImageDraw.Draw(img)
    d.rounded_rectangle([2,2,w-3,h-3],radius=8,outline=(25,25,25),width=4)
    if typ=='type1a':
        d.text((12,5),number[:4],font=font(79),fill=(15,15,15),stroke_width=0)
        d.text((10,91),number[4:6],font=font(62),fill=(15,15,15))
        d.text((130,105),number[6:],font=font(48),fill=(15,15,15))
        d.text((235,143),'RUS',font=font(16),fill=(15,15,15))
    elif typ=='other':
        d.text((30,15),'MOTO 123',font=font(59),fill=(15,15,15))
    else:
        d.text((18,12),number[:6],font=font(82),fill=(15,15,15))
        d.line((387,8,387,104),fill=(50,50,50),width=2)
        d.text((398,11),number[6:],font=font(48 if len(number)==9 else 58),fill=(15,15,15))
        d.text((433,76),'RUS',font=font(16),fill=(15,15,15))
    return img

def sample(rng,width,height,typ,number):
    arr=np.zeros((height,width,3),np.uint8)
    # synthetic vehicle-like scene, never mislabeled as a real photograph
    sky=rng.randint(90,180); arr[:height//2]=[sky,sky+10 if sky<=240 else sky,sky]
    arr[height//2:]=[rng.randint(50,130)]*3
    body_col=(rng.randint(25,180),rng.randint(25,180),rng.randint(25,180))
    cv2.rectangle(arr,(int(width*.1),int(height*.25)),(int(width*.9),int(height*.83)),body_col,-1)
    cv2.rectangle(arr,(int(width*.26),int(height*.3)),(int(width*.75),int(height*.52)),(45,56,62),-1)
    for q in (.22,.78):cv2.circle(arr,(int(width*q),int(height*.83)),int(height*.075),(18,18,18),-1)
    base=Image.fromarray(arr)
    pl=plate(rng,typ,number)
    scale=rng.uniform(.35,.82)
    pw=int(pl.width*scale);ph=int(pl.height*scale)
    pl=pl.resize((pw,ph),Image.Resampling.BICUBIC)
    x=int((width-pw)*.5+rng.uniform(-.07,.07)*width)
    y=int(height*.58+rng.uniform(-.045,.055)*height)
    x=max(0,min(width-pw,x)); y=max(0,min(height-ph,y))
    if pw>=width or ph>=height: raise ValueError('Canvas too small')
    base.paste(pl,(x,y))
    if rng.random()<.25:base=base.filter(ImageFilter.GaussianBlur(radius=rng.uniform(.2,.8)))
    if rng.random()<.40:base=ImageEnhance.Brightness(base).enhance(rng.uniform(.65,1.2))
    return base,(x,y,pw,ph),[x,y,x+pw,y,x+pw,y+ph,x,y+ph]

def main():
    a=argparse.ArgumentParser();a.add_argument('--output',default='dataset');a.add_argument('--count',type=int,default=5000);a.add_argument('--seed',type=int,default=2026);a.add_argument('--width',type=int,default=640);a.add_argument('--height',type=int,default=360);args=a.parse_args()
    if args.count<1:a.error('--count must be positive')
    root=Path(args.output);(root/'images/synthetic').mkdir(parents=True,exist_ok=True);(root/'labels').mkdir(exist_ok=True)
    rng=random.Random(args.seed)
    metapath=root/'meta.csv'
    # deterministic overwrite; do not append duplicate metadata
    with metapath.open('w',encoding='utf-8',newline='') as f:
        wr=csv.writer(f,delimiter=';');wr.writerow(HEADER)
        for i in range(args.count):
            typ=CLASSES[i%4] if i<4 else rng.choices(CLASSES,[.30,.31,.29,.10])[0]
            num=gen_number(rng)
            im,bbox,quad=sample(rng,args.width,args.height,typ,num)
            relative=f'images/synthetic/syn_{i:06}.jpg'
            im.save(root/relative,quality=79,optimize=True)
            x,y,w,h=bbox
            yolo=[(x+w/2)/args.width,(y+h/2)/args.height,w/args.width,h/args.height]
            (root/'labels'/f'syn_{i:06}.txt').write_text(f'{CLASSES.index(typ)} '+ ' '.join(f'{v:.6f}' for v in yolo)+'\n',encoding='utf-8')
            wr.writerow([relative, num if typ!='other' else '########',typ,','.join(map(str,bbox)),','.join(map(str,quad)),1,1,'self_generated','CC BY 4.0','synthetic'])
            if (i+1)%1000==0:print(f'Generated {i+1}/{args.count}',flush=True)
    print('Saved',metapath)
if __name__=='__main__':main()