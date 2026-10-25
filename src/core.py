"""Shared offline detection and OCR utilities."""
from __future__ import annotations
import re
import cv2
import numpy as np

ALPHABET = "ABEKMHOPCTYX"
VALID_TYPES = {"type1", "type1a", "type1b", "other"}
MAPPING = str.maketrans("АВЕКМНОРСТУХ", "ABEKMHOPCTYX")
PATTERN = re.compile(r"^[ABEKMHOPCTYX][0-9]{3}[ABEKMHOPCTYX]{2}(?:[0-9]{2}|[127][0-9]{2})$")

def normalize(s):
    return re.sub(r"[^A-Z0-9#]", "", s.upper().translate(MAPPING))

def correct_by_mask(s):
    s = normalize(s)
    to_letter = str.maketrans({"0":"O", "8":"B", "4":"A", "3":"E", "1":"T", "7":"Y"})
    to_digit = str.maketrans({"O":"0", "B":"8", "I":"1", "Z":"2", "S":"5", "T":"7"})
    if len(s) not in (8, 9):
        return s
    chars = list(s)
    for i in (0, 4, 5):
        chars[i] = chars[i].translate(to_letter)
    for i in [1, 2, 3] + list(range(6, len(s))):
        chars[i] = chars[i].translate(to_digit)
    return "".join(chars)

def plate_type(crop):
    h, w = crop.shape[:2]
    if h == 0 or w == 0:
        return "other"
    if w/h < 2.8:
        return "type1a"
    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    yellow = cv2.inRange(hsv, np.array([13,55,65]), np.array([42,255,255]))
    return "type1b" if np.mean(yellow > 0) > .22 else "type1"

def detect_candidates(img):
    """Classical image-processing fallback. Use trained detector for production."""
    h, w = img.shape[:2]
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(cv2.GaussianBlur(gray,(5,5),0),40,135)
    contours, _ = cv2.findContours(cv2.dilate(edges, cv2.getStructuringElement(cv2.MORPH_RECT,(5,3))),cv2.RETR_LIST,cv2.CHAIN_APPROX_SIMPLE)
    found = []
    for c in contours:
        x,y,bw,bh = cv2.boundingRect(c)
        ratio = bw/max(bh,1)
        if bw < max(65,w*.075) or bh < 18 or bw*bh < .0008*w*h or bw*bh > .35*w*h:
            continue
        if not (1.35 <= ratio <= 6.6):
            continue
        pad = max(2,int(.02*bw))
        found.append((max(0,x-pad),max(0,y-pad),min(w,x+bw+pad),min(h,y+bh+pad),.5))
    found.sort(key=lambda q:(q[2]-q[0])*(q[3]-q[1]),reverse=True)
    kept=[]
    for c in found:
        x,y,x2,y2,_=c
        if any((max(0,min(x2,k[2])-max(x,k[0]))*max(0,min(y2,k[3])-max(y,k[1])))/max(1,min((x2-x)*(y2-y),(k[2]-k[0])*(k[3]-k[1])))>.72 for k in kept):
            continue
        kept.append(c)
        if len(kept)>=20:
            break
    return kept

def ocr_plate(crop, is_square=False):
    try:
        import pytesseract
    except ImportError:
        return "", 0.
    if crop.size==0:
        return "", 0.
    gray=cv2.cvtColor(crop,cv2.COLOR_BGR2GRAY)
    outputs=[]
    for scale in [2,3]:
        resized=cv2.resize(gray,None,fx=scale,fy=scale,interpolation=cv2.INTER_CUBIC)
        for threshold in [False, True]:
            view=cv2.adaptiveThreshold(resized,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY,31,9) if threshold else resized
            for psm in ([6,11] if is_square else [7]):
                try:
                    data=pytesseract.image_to_data(view,config=f"--oem 3 --psm {psm} -c tessedit_char_whitelist=ABEKMHOPCTYX0123456789",output_type=pytesseract.Output.DICT)
                except (pytesseract.TesseractNotFoundError,FileNotFoundError):
                    return "",0.
                items=[]; conf=[]
                for t,q in zip(data["text"],data["conf"]):
                    value=normalize(t)
                    if value:
                        items.append(value)
                    try:
                        if float(q)>=0:
                            conf.append(float(q))
                    except (ValueError,TypeError):
                        pass
                candidate=correct_by_mask("".join(items))
                if candidate:
                    valid=bool(PATTERN.fullmatch(candidate))
                    score=min(.99,max(0.,np.mean(conf)/100 if conf else 0.))
                    outputs.append((candidate,score,valid))
    if not outputs:
        return "",0.
    outputs.sort(key=lambda x:(x[2],x[1],-abs(len(x[0])-9)),reverse=True)
    return outputs[0][:2]
