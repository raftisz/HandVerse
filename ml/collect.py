"""
อัดท่ามือของตัวเอง -> เก็บเป็นจุดข้อนิ้ว (ไม่ได้เก็บภาพ)
-----------------------------------------------------
วิธีใช้:
  python ml/collect.py train    อัดชุดเทรน   (ท่าละ ~200)
  python ml/collect.py test     อัดชุดทดสอบ (ท่าละ ~100) -> อัดคนละรอบกับชุดเทรน จะได้วัดผลแบบยุติธรรม

ปุ่ม:
  1-7     เลือกท่าที่จะอัด
  SPACE   กดค้าง = อัด (ทำท่าค้างไว้ แล้วขยับมือไปมา: ใกล้/ไกล/เอียง/ชี้เข้ากล้อง/ทั้งสองมือ)
  BACKSPACE  ลบที่เพิ่งอัดของท่านี้ในรอบนี้
  ESC     บันทึกและออก
ไฟล์: data/own_train.csv, data/own_test.csv
"""
import csv
import os
import sys
import time

import cv2
import mediapipe as mp
import numpy as np
import pygame

GESTURES = [            # (ชื่อ, คำอธิบายบนจอ)
    ("one",    "1  index finger only (pen / planet)"),
    ("peace",  "2  two fingers (lift pen / heart)"),
    ("fist",   "3  fist"),
    ("palm",   "4  open hand"),
    ("middle", "5  middle finger only"),
    ("call",   "6  pinky only (thumb optional)"),
    ("none",   "7  anything else / relaxed hand / half gestures"),
]
TARGET = {"train": 200, "test": 100}

split = sys.argv[1] if len(sys.argv) > 1 else "train"
assert split in ("train", "test"), "ใช้: python ml/collect.py train  หรือ  python collect.py test"
HERE = os.path.dirname(os.path.abspath(__file__))
os.makedirs(os.path.join(HERE, "data"), exist_ok=True)
OUT = os.path.join(HERE, "data", f"own_{split}.csv")
SESSION = time.strftime("%Y%m%d_%H%M%S")

# นับของเดิมที่อัดไว้แล้ว (อัดต่อจากรอบก่อนได้)
counts = {g: 0 for g, _ in GESTURES}
if os.path.exists(OUT):
    with open(OUT) as f:
        for row in csv.reader(f):
            if row and row[0] in counts:
                counts[row[0]] += 1


def scancode_digit(e):
    """อ่านปุ่มเลข 1-7 จากตำแหน่งปุ่ม (ใช้ได้แม้แป้นเป็นภาษาไทย)"""
    for i in range(1, 8):
        if e.scancode == getattr(pygame, f"KSCAN_{i}") or e.key == getattr(pygame, f"K_{i}"):
            return i
    return None


def main():
    cap = cv2.VideoCapture(0, cv2.CAP_AVFOUNDATION) if sys.platform == "darwin" else cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    hands = mp.solutions.hands.Hands(max_num_hands=2, model_complexity=1,
                                     min_detection_confidence=0.6, min_tracking_confidence=0.6)
    drawer = mp.solutions.drawing_utils

    pygame.init()
    screen = pygame.display.set_mode((960, 600))
    pygame.display.set_caption(f"collect ({split})")
    font = pygame.font.SysFont(None, 28)
    big = pygame.font.SysFont(None, 44)
    cur = 0
    rows, added = [], {g: 0 for g, _ in GESTURES}
    running = True
    while running:
        for e in pygame.event.get():
            if e.type == pygame.QUIT or (e.type == pygame.KEYDOWN and e.key == pygame.K_ESCAPE):
                running = False
            elif e.type == pygame.KEYDOWN:
                d = scancode_digit(e)
                if d:
                    cur = d - 1
                if e.key == pygame.K_BACKSPACE:          # ลบที่อัดของท่านี้ในรอบนี้
                    g = GESTURES[cur][0]
                    rows = [r for r in rows if r[0] != g]
                    counts[g] -= added[g]
                    added[g] = 0
        recording = pygame.key.get_pressed()[pygame.K_SPACE]

        ok, frame = cap.read()
        if not ok:
            continue
        frame = cv2.flip(frame, 1)
        res = hands.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        g = GESTURES[cur][0]
        for h, hd in zip(res.multi_hand_landmarks or [], res.multi_handedness or []):
            drawer.draw_landmarks(frame, h, mp.solutions.hands.HAND_CONNECTIONS)
            if recording:
                vals = [v for p in h.landmark for v in (p.x, p.y, p.z)]
                rows.append([g, hd.classification[0].label, SESSION] + [f"{v:.5f}" for v in vals])
                counts[g] += 1
                added[g] += 1

        img = cv2.cvtColor(cv2.resize(frame, (640, 480)), cv2.COLOR_BGR2RGB)
        screen.fill((15, 15, 20))
        screen.blit(pygame.surfarray.make_surface(img.swapaxes(0, 1)), (0, 60))
        title = f"[{split}]  {GESTURES[cur][1]}"
        screen.blit(big.render(title, True, (255, 210, 60)), (12, 12))
        if recording:
            pygame.draw.circle(screen, (255, 40, 40), (620, 90), 12)
        for i, (name, desc) in enumerate(GESTURES):
            n, t = counts[name], TARGET[split]
            col = (120, 255, 140) if n >= t else (220, 220, 220)
            if i == cur:
                col = (255, 210, 60)
            screen.blit(font.render(f"{i + 1}. {name:7s} {n:4d}/{t}", True, col), (660, 80 + i * 34))
        screen.blit(font.render("hold SPACE = record   BACKSPACE = undo   ESC = save", True, (150, 150, 150)), (12, 560))
        pygame.display.flip()

    if rows:
        with open(OUT, "a", newline="") as f:
            csv.writer(f).writerows(rows)
    print(f"บันทึก {len(rows)} ตัวอย่าง -> {OUT}")
    print({k: v for k, v in counts.items()})
    cap.release()
    pygame.quit()


if __name__ == "__main__":
    main()
