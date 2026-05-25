"""
训练 YOLOv8 12 类 CIFAR 分类模型
===================================
轻量分类模型，专用于环内裁切图的 12 类识别，
不受圆环/方块结构干扰。

部署目标: Jetson Orin Nano TensorRT FP16
"""

import os
import sys
from ultralytics import YOLO

_BASE = os.path.dirname(os.path.abspath(__file__))
DATASET = os.path.join(_BASE, "dataset_cls")
OUTPUT = os.path.join(_BASE, "runs_cls")

MODEL = "yolov8n-cls.pt"
EPOCHS = 80
IMGSZ = 96             # 匹配 32×32 源图，不过度放大
BATCH = 128            # 小尺寸可加大批次
DEVICE = 0


def train():
    print("=" * 55)
    print("  YOLOv8n-cls 12 类 CIFAR 分类训练")
    print(f"  数据集: {DATASET}")
    print(f"  {EPOCHS} epochs | {IMGSZ}px | batch={BATCH}")
    print("=" * 55)

    model = YOLO(MODEL)
    model.train(
        data=DATASET,
        epochs=EPOCHS,
        imgsz=IMGSZ,
        batch=BATCH,
        device=DEVICE,
        workers=2,
        lr0=0.01,
        lrf=0.01,
        cos_lr=True,
        optimizer='AdamW',
        amp=True,
        seed=42,
        project=OUTPUT,
        name='cifar12_cls',
        exist_ok=True,
        # 强旋转增强：物理靶标可任意角度
        hsv_h=0.02,
        hsv_s=0.8,
        hsv_v=0.5,
        degrees=90,        # ±90° 全角度覆盖
        translate=0.15,
        scale=0.6,
        shear=10,          # 剪切模拟透视变形
        fliplr=0.5,
        flipud=0.3,        # 上下翻转
        erasing=0.25,
    )

    best = os.path.join(OUTPUT, "cifar12_cls", "weights", "best.pt")
    print(f"\n[OK] {best}")

    # 验证
    model = YOLO(best)
    results = model.val(data=DATASET, imgsz=IMGSZ)
    print(f"  Top-1: {results.top1:.4f}  Top-5: {results.top5:.4f}")

    print(f"\n  部署到 Orin Nano:")
    print(f"    model.export(format='engine', imgsz=224, half=True)")


if __name__ == "__main__":
    train()
