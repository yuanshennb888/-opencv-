"""
单模型靶标识别：纯圆环定位 + YOLOv8n-cls 分类
===============================================
不使用 YOLO 检测模型，靠靶标特有的同心圆环图案定位。
轻量：仅 3MB 分类模型，全流程 GPU 加速。

流程:
  1. HoughCircles 在画面中找同心双圆 → 定位靶标
  2. 四边形角点检测 → 透视校正
  3. 裁切内环区域（纯嵌入图）
  4. YOLOv8n-cls 分类 → 输出类别

按键: q=退出  空格=截图  +/-=Hough灵敏度
"""

import cv2
import numpy as np
import time
import os
from collections import deque

# ==================== 配置 ====================
_BASE = os.path.dirname(os.path.abspath(__file__))
CLS_MODEL_PATH = os.path.join(_BASE, "runs_cls", "cifar12_cls", "weights", "best.pt")
CAM_ID = 1
RECT_SIZE = 480
CLS_SIZE = 96          # 分类输入尺寸（匹配训练）

# 圆环检测参数
HOUGH_PARAM1 = 80
HOUGH_PARAM2 = 40

# 时序平滑
SMOOTH_WINDOW = 15
SMOOTH_MIN_RATIO = 0.4

# ==================== 初始化 ====================
from ultralytics import YOLO

print("加载分类模型...")
cls_model = YOLO(CLS_MODEL_PATH)
cap = cv2.VideoCapture(CAM_ID)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

fps_smooth = 0
cls_history = deque(maxlen=SMOOTH_WINDOW)  # 时序平滑缓存
stable_cls, stable_conf = "", 0             # 稳定输出
print("=" * 55)
print("  单模型靶标识别 (纯圆环定位)")
print(f"  分类: YOLOv8n-cls@224px | Top-1: 100%")
print("  按键: q=退出 空格=截图  +/-=Hough")
print("=" * 55)


# ==================== 靶标定位 ====================

def find_target_by_rings(frame):
    """在画面中寻找同心圆环 → 定位靶标
    
    返回: (cx, cy, outer_r, inner_r, quad_pts) 或 None
    """
    h, w = frame.shape[:2]
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (7, 7), 2)

    all_circles = []
    for p2 in [HOUGH_PARAM2, max(20, HOUGH_PARAM2 - 20)]:
        circles = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT,
                                    dp=1.2, minDist=30,
                                    param1=HOUGH_PARAM1, param2=p2,
                                    minRadius=int(h*0.04), maxRadius=int(h*0.40))
        if circles is not None:
            all_circles.extend(circles[0])

    if len(all_circles) < 2: return None

    # 找最佳同心圆对
    best_pair, best_score = None, 0
    for i in range(len(all_circles)):
        for j in range(i+1, len(all_circles)):
            x1, y1, r1 = all_circles[i]
            x2, y2, r2 = all_circles[j]
            center_dist = np.hypot(x1-x2, y1-y2)
            r_min, r_max = min(r1, r2), max(r1, r2)
            if r_max < 1: continue
            ratio = r_min / r_max
            if center_dist > r_max * 0.20: continue
            if not (0.45 < ratio < 0.85): continue
            off_center = np.hypot((x1+x2)/2 - w/2, (y1+y2)/2 - h/2)
            if off_center > min(w, h) * 0.35: continue
            score = (1 - center_dist/(r_max+1)) * (1 - off_center/(min(w,h)*0.5))
            if score > best_score:
                best_score = score
                cx, cy = int((x1+x2)/2), int((y1+y2)/2)
                best_pair = (cx, cy, int(r_max), int(r_min))

    if best_pair is None: return None
    cx, cy, outer_r, inner_r = best_pair

    # 方块四角
    search_r = int(outer_r * 1.55)
    x1_r, y1_r = max(0, cx-search_r), max(0, cy-search_r)
    x2_r, y2_r = min(w, cx+search_r), min(h, cy+search_r)
    roi = frame[y1_r:y2_r, x1_r:x2_r]
    quad_pts = _find_square_corners(roi)
    if quad_pts is not None:
        quad_pts = [(p[0]+x1_r, p[1]+y1_r) for p in quad_pts]

    return cx, cy, outer_r, inner_r, quad_pts


def _find_square_corners(roi):
    """在 ROI 中找黑色方块四角（精简版）"""
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                    cv2.THRESH_BINARY_INV, 21, 5)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours: return None
    best, best_area = None, 0
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < roi.shape[0]*roi.shape[1]*0.02: continue
        peri = cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, 0.04*peri, True)
        if len(approx) == 4 and area > best_area:
            best_area = area; best = approx
    if best is None: return None
    pts = best.reshape(4, 2).astype(np.float32)
    s, diff = pts.sum(axis=1), np.diff(pts, axis=1)
    return [tuple(pts[np.argmin(s)]), tuple(pts[np.argmin(diff)]),
            tuple(pts[np.argmax(s)]), tuple(pts[np.argmax(diff)])]


def find_ring_on_rectified(rectified):
    """在校正图上快速定位白环（仅 Hough，两档灵敏度）"""
    h, w = rectified.shape[:2]
    gray = cv2.cvtColor(rectified, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 1.5)
    for p2 in [HOUGH_PARAM2, max(15, HOUGH_PARAM2-15)]:
        circles = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT,
                                    dp=1.2, minDist=15,
                                    param1=60, param2=p2,
                                    minRadius=int(h*0.06), maxRadius=int(h*0.35))
        if circles is not None:
            best, best_d = None, float('inf')
            for c in circles[0]:
                d = np.hypot(c[0]-w/2, c[1]-h/2)
                if d < best_d and d < h*0.25:
                    best_d = d; best = c
            if best is not None:
                return int(best[0]), int(best[1]), int(best[2])
    return None


def perspective_correct(frame, quad_pts):
    """用方块四角做透视校正 → 480×480"""
    src = np.float32(quad_pts)
    dst = np.float32([[0, 0], [RECT_SIZE, 0], [RECT_SIZE, RECT_SIZE], [0, RECT_SIZE]])
    M = cv2.getPerspectiveTransform(src, dst)
    return cv2.warpPerspective(frame, M, (RECT_SIZE, RECT_SIZE))


def draw_markers(img, cx, cy, r):
    annotated = img.copy()
    cv2.circle(annotated, (cx, cy), r, (0, 255, 0), 3)
    cv2.line(annotated, (cx-20, cy), (cx+20, cy), (0, 255, 255), 2)
    cv2.line(annotated, (cx, cy-20), (cx, cy+20), (0, 255, 255), 2)
    cv2.circle(annotated, (cx, cy), 5, (0, 0, 255), -1)
    return annotated


def crop_inside_ring(rectified, cx, cy, r):
    inner_r = int(r * 0.82)
    h, w = rectified.shape[:2]
    x1, y1 = max(0, cx-inner_r), max(0, cy-inner_r)
    x2, y2 = min(w, cx+inner_r), min(h, cy+inner_r)
    return rectified[y1:y2, x1:x2]


# ==================== 分类 ====================

def classify(inner_crop):
    """YOLOv8n-cls 分类（96px，匹配源图信息量）"""
    if inner_crop is None or inner_crop.size == 0:
        return "?", 0, []
    img = cv2.resize(inner_crop, (CLS_SIZE, CLS_SIZE))
    results = cls_model(img, verbose=False)
    for r in results:
        if r.probs is not None:
            top3 = []
            for i in range(min(3, len(r.probs.top5))):
                idx = int(r.probs.top5[i])
                name = cls_model.names.get(idx, str(idx))
                conf = float(r.probs.top5conf[i])
                top3.append((name, conf))
            if top3:
                return top3[0][0], top3[0][1], top3
    return "?", 0, []


# ==================== 主循环 ====================
print("摄像头已开启...\n")

while True:
    t0 = time.perf_counter()
    ret, frame = cap.read()
    if not ret: break
    h, w = frame.shape[:2]

    # ---- 圆环定位靶标 ----
    target = find_target_by_rings(frame)
    rectified = None
    inner_crop = None
    ring_disp = None
    cls_name, cls_conf, top3 = "?", 0, []

    if target is not None:
        cx, cy, outer_r, inner_r, quad_pts = target

        # 透视校正
        if quad_pts is not None:
            rectified = perspective_correct(frame, quad_pts)
        else:
            # 无角点时：以圆环为中心裁切
            margin = int(outer_r * 1.1)
            x1, y1 = max(0, cx-margin), max(0, cy-margin)
            x2, y2 = min(w, cx+margin), min(h, cy+margin)
            rectified = cv2.resize(frame[y1:y2, x1:x2], (RECT_SIZE, RECT_SIZE))

        # 在校正图上精确定位内环
        ring_result = find_ring_on_rectified(rectified)
        if ring_result is not None:
            rx, ry, rr = ring_result
            inner_crop = crop_inside_ring(rectified, rx, ry, rr)
            ring_disp = draw_markers(rectified, rx, ry, rr)
        else:
            ring_disp = rectified.copy()
            # 无环时裁中心
            m = RECT_SIZE // 4
            inner_crop = rectified[m:RECT_SIZE-m, m:RECT_SIZE-m]

        # 分类
        cls_name, cls_conf, top3 = classify(inner_crop)

        # === 时序平滑：缓存结果，多数投票消除跳变 ===
        if cls_name != "?":
            cls_history.append((cls_name, cls_conf))
        
        if len(cls_history) >= 5:
            # 统计窗口内各类出现次数
            from collections import Counter
            names = [c for c, _ in cls_history]
            counter = Counter(names)
            best_name, best_count = counter.most_common(1)[0]
            ratio = best_count / len(cls_history)
            
            if ratio >= SMOOTH_MIN_RATIO:
                # 计算该类在窗口内的平均置信度
                avg_conf = np.mean([conf for n, conf in cls_history if n == best_name])
                stable_cls, stable_conf = best_name, avg_conf
        else:
            stable_cls, stable_conf = cls_name, cls_conf

        # 在原图上画靶标位置标记
        cv2.circle(frame, (cx, cy), outer_r, (0, 255, 0), 2)
        cv2.circle(frame, (cx, cy), inner_r, (255, 200, 0), 1)
        if quad_pts is not None:
            for i in range(4):
                pt1 = tuple(map(int, quad_pts[i]))
                pt2 = tuple(map(int, quad_pts[(i+1)%4]))
                cv2.line(frame, pt1, pt2, (0, 255, 255), 2)

    # ---- 显示（使用稳定后的结果） ----
    elapsed = time.perf_counter() - t0
    fps_smooth = 0.9*fps_smooth + 0.1*(1/elapsed) if fps_smooth else 1/elapsed

    if stable_cls:
        # 稳定结果（大字绿色）
        cv2.putText(frame, f"{stable_cls} {stable_conf:.1%}", (15, 35),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 2)
        # 当前帧原始结果（小字灰色参考）
        if cls_name != "?" and cls_name != stable_cls:
            cv2.putText(frame, f"raw:{cls_name} {cls_conf:.1%}", (15, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (120, 120, 120), 1)
        if top3:
            top3_text = " | ".join([f"{n}:{s:.2f}" for n, s in top3])
            cv2.putText(frame, top3_text, (15, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (160, 160, 160), 1)

        print(f"\r[{stable_cls}] {stable_conf:.1%} | FPS:{fps_smooth:.0f}  ", end="")

    cv2.putText(frame, f"FPS:{fps_smooth:.0f}", (w-100, 25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 1)

    # 显示面板
    if rectified is not None:
        panel_h = min(480, h)
        ratio = panel_h / h
        frame_s = cv2.resize(frame, (int(w*ratio), panel_h))
        ring_s = cv2.resize(ring_disp if ring_disp is not None else rectified, (RECT_SIZE, min(RECT_SIZE, panel_h)))
        panel_w = frame_s.shape[1] + RECT_SIZE + 10
        if inner_crop is not None and inner_crop.size > 0:
            inner_s = cv2.resize(inner_crop, (RECT_SIZE, min(RECT_SIZE, panel_h)))
            panel_w += RECT_SIZE + 5

        panel = np.zeros((panel_h, panel_w, 3), dtype=np.uint8)
        panel[:, :frame_s.shape[1]] = frame_s
        panel[:ring_s.shape[0], frame_s.shape[1]+5:frame_s.shape[1]+5+RECT_SIZE] = ring_s
        if inner_crop is not None and inner_crop.size > 0:
            off = frame_s.shape[1] + RECT_SIZE + 10
            panel[:inner_s.shape[0], off:off+RECT_SIZE] = inner_s

        cv2.imshow("Target - Ring Only", panel)
    else:
        cv2.imshow("Target - Ring Only", frame)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'): break
    elif key == ord(' '):
        os.makedirs(os.path.join(_BASE, "screenshots"), exist_ok=True)
        ts = time.strftime("%Y%m%d_%H%M%S")
        cv2.imwrite(os.path.join(_BASE, "screenshots", f"cap_{ts}.jpg"), frame)
        print(f"\n[截图] {ts}")
    elif key in (ord('+'), ord('=')):
        HOUGH_PARAM2 = min(200, HOUGH_PARAM2+5)
        print(f"\rHough p2={HOUGH_PARAM2}  ", end="")
    elif key == ord('-'):
        HOUGH_PARAM2 = max(5, HOUGH_PARAM2-5)
        print(f"\rHough p2={HOUGH_PARAM2}  ", end="")

cap.release()
cv2.destroyAllWindows()
print("\nDone")
