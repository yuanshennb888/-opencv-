"""形态学环提取法 — 顶帽变换提取环形亮结构"""
import cv2, numpy as np, time, os
from collections import deque, Counter

_BASE = os.path.dirname(os.path.abspath(__file__))
_PARENT = os.path.dirname(_BASE)
CLS_MODEL = os.path.join(_PARENT, "ring_detect", "runs_cls", "cifar12_cls", "weights", "best.pt")
CLS_SIZE, SMOOTH_N = 96, 15

from ultralytics import YOLO
print("加载模型...")
cls_model = YOLO(CLS_MODEL)
cap = cv2.VideoCapture(1)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

def find_ring(gray):
    h, w = gray.shape[:2]
    ks = max(5, int(h*0.04))
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ks, ks))
    tophat = cv2.morphologyEx(gray, cv2.MORPH_TOPHAT, kernel)
    _, binary = cv2.threshold(tophat, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
    cnts, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts: return None
    best, bs = None, 0
    for cnt in cnts:
        area = cv2.contourArea(cnt)
        if area < h*w*0.003 or area > h*w*0.3: continue
        (cx, cy), r = cv2.minEnclosingCircle(cnt); r = int(r)
        if r < h*0.06 or r > h*0.35: continue
        d = np.hypot(cx-w/2, cy-h/2)
        if d > min(w,h)*0.3: continue
        hull = cv2.convexHull(cnt)
        ha = cv2.contourArea(hull)
        circ = area/(ha+1)
        score = circ * (1-d/(min(w,h)*0.5))
        if score > bs: bs = score; best = (int(cx), int(cy), r)
    return best

def classify(crop):
    if crop is None or crop.size == 0: return "?", 0, []
    r = cls_model(cv2.resize(crop, (CLS_SIZE, CLS_SIZE)), verbose=False)
    for x in r:
        if x.probs is not None:
            t3 = [(cls_model.names.get(int(x.probs.top5[i]), "?"), float(x.probs.top5conf[i])) for i in range(min(3, len(x.probs.top5)))]
            return (t3[0][0], t3[0][1], t3) if t3 else ("?", 0, [])
    return "?", 0, []

print("="*50); print("  形态学环提取法"); print("  按键: q=退出 空格=截图"); print("="*50)
fps_s, hist = 0, deque(maxlen=SMOOTH_N)
stable = ("", 0)

while True:
    t0 = time.perf_counter()
    ok, frame = cap.read()
    if not ok: break
    h, w = frame.shape[:2]
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    rr = find_ring(gray)
    cls_n, cls_c, top3 = "?", 0, []
    if rr:
        cx, cy, r = rr
        m = int(r*1.5); x1,y1=max(0,cx-m),max(0,cy-m); x2,y2=min(w,cx+m),min(h,cy+m)
        rect = cv2.resize(frame[y1:y2,x1:x2], (480,480))
        inner = rect[96:384, 96:384]
        cls_n, cls_c, top3 = classify(inner)
        cv2.circle(frame, (cx, cy), r, (0,255,0), 2)
        cv2.putText(frame, "Morph", (cx-20, cy-r-5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0,255,255), 1)
        if cls_n != "?":
            hist.append((cls_n, cls_c))
            if len(hist) >= 5:
                cnt = Counter([c for c,_ in hist])
                best_n, best_cnt = cnt.most_common(1)[0]
                if best_cnt/len(hist) >= 0.4:
                    stable = (best_n, np.mean([conf for n,conf in hist if n==best_n]))
            else: stable = (cls_n, cls_c)
    fps_s = 0.9*fps_s + 0.1*(1/(time.perf_counter()-t0)) if fps_s else 1/(time.perf_counter()-t0)
    if stable[0]:
        cv2.putText(frame, f"{stable[0]} {stable[1]:.1%}", (15,35), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0,255,0), 2)
        print(f"\r[{stable[0]}] {stable[1]:.1%} [Morph] FPS:{fps_s:.0f}  ", end="")
    cv2.putText(frame, f"FPS:{fps_s:.0f}", (w-100,25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 1)
    cv2.imshow("Morph Ring", frame)
    k = cv2.waitKey(1) & 0xFF
    if k == ord('q'): break
    elif k == ord(' '):
        os.makedirs(os.path.join(_BASE,"shots"), exist_ok=True)
        cv2.imwrite(os.path.join(_BASE,"shots",f"morph_{time.strftime('%H%M%S')}.jpg"), frame)

cap.release(); cv2.destroyAllWindows(); print("\nDone")
