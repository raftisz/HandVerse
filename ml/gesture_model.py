"""
โมเดลจำท่ามือ (ใช้แค่ numpy -> ไม่ต้องลงไลบรารีเพิ่มตอนรันเกม)
- hand_features(): แปลงจุดข้อนิ้ว 21 จุด -> ฟีเจอร์ที่ไม่สนตำแหน่ง/ขนาด/การหมุน/มือซ้ายขวา
- GestureModel: โหลดน้ำหนักจาก gesture_model.npz แล้วทาย (MLP เขียนเองด้วย numpy)
"""
import os

import numpy as np

CLASSES = ["one", "peace", "fist", "palm", "middle", "call", "none"]
TIPS = [4, 8, 12, 16, 20]


def hand_features(xy, aspect=640 / 480):
    """xy: (21, 2) พิกัด 0-1 จาก MediaPipe -> เวกเตอร์ฟีเจอร์ 1 มิติ
    1) ปรับสเกลแกน x ให้เท่าแกน y (ภาพไม่ใช่สี่เหลี่ยมจัตุรัส)
    2) ย้ายข้อมือไปที่ (0,0)
    3) หมุนให้ ข้อมือ->โคนนิ้วกลาง ชี้ขึ้นเสมอ  (ชี้ไปทางไหนก็เป็นท่าเดิม)
    4) ย่อ/ขยายให้ ข้อมือ->โคนนิ้วกลาง ยาว 1  (มือใกล้/ไกลกล้อง ก็เป็นท่าเดิม)
    5) กลับซ้าย-ขวา ให้นิ้วโป้งอยู่ฝั่งขวาเสมอ  (มือซ้าย/ขวา หน้ามือ/หลังมือ ก็เป็นท่าเดิม)"""
    p = np.asarray(xy, dtype=float)[:, :2].copy()
    p[:, 0] *= aspect
    p -= p[0]
    v = p[9]
    ang = np.arctan2(v[0], -v[1])                     # มุมจากแนวตั้ง
    c, s = np.cos(ang), np.sin(ang)
    p = p @ np.array([[c, -s], [s, c]])                # หมุนกลับ -> ข้อมือ->โคนนิ้วกลาง ชี้ขึ้น
    p /= (np.linalg.norm(p[9]) + 1e-6)
    if p[2, 0] < 0:                                   # โคนนิ้วโป้งต้องอยู่ฝั่งขวา
        p[:, 0] *= -1
    tip_d = np.linalg.norm(p[TIPS], axis=1)           # ระยะปลายนิ้วถึงข้อมือ
    # มุมงอที่ข้อกลางของแต่ละนิ้ว (cos)
    bends = []
    for mcp, pip, tip in [(5, 6, 8), (9, 10, 12), (13, 14, 16), (17, 18, 20), (2, 3, 4)]:
        a, b = p[pip] - p[mcp], p[tip] - p[pip]
        bends.append(a @ b / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-6))
    return np.concatenate([p.ravel(), tip_d, bends])


class GestureModel:
    def __init__(self, path):
        d = np.load(path, allow_pickle=False)
        self.mean, self.std = d["mean"], d["std"]
        n = int(d["n_layers"])
        self.W = [d[f"W{i}"] for i in range(n)]
        self.b = [d[f"b{i}"] for i in range(n)]
        self.classes = [str(c) for c in d["classes"]]

    def proba(self, feat):
        h = (np.asarray(feat) - self.mean) / self.std
        for i, (W, b) in enumerate(zip(self.W, self.b)):
            h = h @ W + b
            if i < len(self.W) - 1:
                h = np.maximum(h, 0)                   # ReLU
        h = np.exp(h - h.max())
        return h / h.sum()

    def predict(self, xy, aspect=640 / 480):
        """คืน (ชื่อท่า, ความมั่นใจ 0-1)"""
        pr = self.proba(hand_features(xy, aspect))
        i = int(np.argmax(pr))
        return self.classes[i], float(pr[i])


def load_default():
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "gesture_model.npz")
    return GestureModel(path) if os.path.exists(path) else None
