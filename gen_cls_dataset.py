"""
生成 12 类 CIFAR 图片分类数据集
================================
从 1/ 2/ 3/ 文件夹中读取 12 张源图，
通过大量增强模拟真实摄像头拍摄效果，
生成 YOLO 分类训练数据。

输出: dataset_cls/
  train/  apple/ baby/ bear/ ... (每类 500 张)
  val/    apple/ baby/ bear/ ... (每类 100 张)
"""

import os
import random
import numpy as np
import cv2
from tqdm import tqdm

# ==================== 配置 ====================
_BASE = os.path.dirname(os.path.abspath(__file__))
_PARENT = os.path.dirname(_BASE)
OUTPUT = os.path.join(_BASE, "dataset_cls")

CLASSES = ['apple','baby','bear','carassius',
           'africanized','beaver','bed','beetle',
           'beer','bicycle','bowl','boy']

SRC_MAP = {
    'apple':       os.path.join(_PARENT, '1', 'apple.png'),
    'baby':        os.path.join(_PARENT, '1', 'baby.png'),
    'bear':        os.path.join(_PARENT, '1', 'bear.png'),
    'carassius':   os.path.join(_PARENT, '1', 'carassius.png'),
    'africanized': os.path.join(_PARENT, '2', 'africanized.png'),
    'beaver':      os.path.join(_PARENT, '2', 'beaver.png'),
    'bed':         os.path.join(_PARENT, '2', 'bed.png'),
    'beetle':      os.path.join(_PARENT, '2', 'beetle.png'),
    'beer':        os.path.join(_PARENT, '3', 'beer.png'),
    'bicycle':     os.path.join(_PARENT, '3', 'bicycle.png'),
    'bowl':        os.path.join(_PARENT, '3', 'bowl.png'),
    'boy':         os.path.join(_PARENT, '3', 'boy.png'),
}

TRAIN_PER_CLASS = 500
VAL_PER_CLASS = 100
IMG_SIZE = 96          # 匹配 32×32 源图实际信息量，不过度放大
# 对难分类的类加倍训练量
OVER_SAMPLE = {'africanized': 2, 'baby': 2}  # 类名: 倍数


def augment_image(img, strong=False):
    """模拟摄像头实拍效果的数据增强"""
    h, w = img.shape[:2]
    aug = img.copy()

    # 随机缩放 (80%~120%)
    sc = random.uniform(0.8, 1.2)
    new_w, new_h = int(w * sc), int(h * sc)
    aug = cv2.resize(aug, (new_w, new_h))

    # 随机旋转 (±90° 全角度)
    angle = random.uniform(-90, 90)
    M = cv2.getRotationMatrix2D((new_w/2, new_h/2), angle, 1.0)
    aug = cv2.warpAffine(aug, M, (new_w, new_h), borderValue=(128, 128, 128))

    # 随机翻转（水平+垂直，覆盖倒置情况）
    if random.random() < 0.5:
        aug = cv2.flip(aug, 1)  # 水平
    if random.random() < 0.5:
        aug = cv2.flip(aug, 0)  # 垂直（倒置）

    # 裁剪/填充到目标尺寸
    if aug.shape[0] > IMG_SIZE:
        y1 = random.randint(0, aug.shape[0] - IMG_SIZE)
        x1 = random.randint(0, aug.shape[1] - IMG_SIZE)
        aug = aug[y1:y1+IMG_SIZE, x1:x1+IMG_SIZE]
    else:
        pad_h = max(0, IMG_SIZE - aug.shape[0])
        pad_w = max(0, IMG_SIZE - aug.shape[1])
        aug = cv2.copyMakeBorder(aug, pad_h//2, pad_h-pad_h//2,
                                  pad_w//2, pad_w-pad_w//2,
                                  cv2.BORDER_CONSTANT, value=(128, 128, 128))
    aug = cv2.resize(aug, (IMG_SIZE, IMG_SIZE))

    # 颜色抖动
    aug = aug.astype(np.float32)
    aug *= random.uniform(0.7, 1.3)
    aug += random.uniform(-20, 20)
    aug = np.clip(aug, 0, 255).astype(np.uint8)

    # 高斯模糊 (30% 概率)
    if random.random() < 0.3:
        k = random.choice([3, 5])
        aug = cv2.GaussianBlur(aug, (k, k), 0)

    # 噪声 (20% 概率)
    if random.random() < 0.2:
        noise = np.random.randint(-12, 12, aug.shape, dtype=np.int16)
        aug = np.clip(aug.astype(np.int16) + noise, 0, 255).astype(np.uint8)

    # JPEG 压缩伪影 (20% 概率)
    if random.random() < 0.2:
        quality = random.randint(30, 80)
        _, buf = cv2.imencode('.jpg', aug, [cv2.IMWRITE_JPEG_QUALITY, quality])
        aug = cv2.imdecode(buf, cv2.IMREAD_COLOR)

    # 透视变换 (15% 概率)
    if random.random() < 0.15:
        s = IMG_SIZE
        src = np.float32([[0,0],[s,0],[0,s],[s,s]])
        jitter = s * 0.08
        dst = np.float32([
            [random.uniform(-jitter,jitter), random.uniform(-jitter,jitter)],
            [s+random.uniform(-jitter,jitter), random.uniform(-jitter,jitter)],
            [random.uniform(-jitter,jitter), s+random.uniform(-jitter,jitter)],
            [s+random.uniform(-jitter,jitter), s+random.uniform(-jitter,jitter)],
        ])
        M = cv2.getPerspectiveTransform(src, dst)
        aug = cv2.warpPerspective(aug, M, (s, s), borderValue=(128,128,128))

    return aug


def generate():
    print("=" * 55)
    print("  12 类 CIFAR 分类数据集生成")
    print(f"  训练: {TRAIN_PER_CLASS}/类  验证: {VAL_PER_CLASS}/类")
    print(f"  尺寸: {IMG_SIZE}×{IMG_SIZE}")
    print("=" * 55)

    for split, num in [('train', TRAIN_PER_CLASS), ('val', VAL_PER_CLASS)]:
        for cls in CLASSES:
            os.makedirs(os.path.join(OUTPUT, split, cls), exist_ok=True)

        for cls in tqdm(CLASSES, desc=f"  {split}"):
            src = cv2.imread(SRC_MAP[cls])
            if src is None:
                print(f"  [MISS] {cls} <- {SRC_MAP[cls]}")
                continue
            src = cv2.resize(src, (IMG_SIZE, IMG_SIZE))

            # 过采样：对难分类的类生成更多样本
            multiplier = OVER_SAMPLE.get(cls, 1) if split == 'train' else 1
            actual_num = num * multiplier

            for i in range(actual_num):
                aug = augment_image(src, strong=(split == 'train'))
                path = os.path.join(OUTPUT, split, cls, f"{cls}_{i:04d}.jpg")
                cv2.imwrite(path, aug)

    total_train = sum(TRAIN_PER_CLASS * OVER_SAMPLE.get(c, 1) for c in CLASSES)
    print(f"\n[OK] 数据集: {OUTPUT}")
    print(f"  训练: {total_train} 张 (含过采样)")
    print(f"  验证: {VAL_PER_CLASS * len(CLASSES)} 张")


if __name__ == "__main__":
    generate()
