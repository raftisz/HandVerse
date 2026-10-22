"""อัดวิดีโอหน้าจอ (.mp4) และแคปหน้าจอ (.png)"""

import os
import queue
import threading
import time

import cv2
import pygame

from .config import H, REC_DIR, REC_FPS, W


class Recorder:
    """อัดหน้าจอเป็น .mp4 โดยเขียนไฟล์ใน thread แยก (เกมไม่กระตุกตอนอัด)"""

    def __init__(self):
        self.q = None
        self.path = None
        self.next_t = 0.0
        self.start_t = 0.0

    @property
    def on(self):
        return self.q is not None

    def start(self):
        os.makedirs(REC_DIR, exist_ok=True)
        self.path = os.path.join(REC_DIR, time.strftime("particles_%Y%m%d_%H%M%S.mp4"))
        writer = cv2.VideoWriter(self.path, cv2.VideoWriter_fourcc(*"mp4v"), REC_FPS, (W, H))
        self.q = queue.Queue(maxsize=120)
        self.start_t = self.next_t = time.time()

        def work(q=self.q):
            while True:
                f = q.get()
                if f is None:
                    break
                writer.write(f)
            writer.release()
        threading.Thread(target=work, daemon=True).start()

    def add(self, screen):
        """เก็บเฟรมตามจังหวะ 30 fps (ถ้าเกมช้ากว่า -> ใส่เฟรมเดิมซ้ำ ให้ความเร็ววิดีโอตรงกับจริง)"""
        now = time.time()
        if now < self.next_t:
            return
        img = cv2.cvtColor(pygame.surfarray.array3d(screen).swapaxes(0, 1), cv2.COLOR_RGB2BGR)
        while self.next_t <= now:
            try:
                self.q.put_nowait(img)
            except queue.Full:                         # เครื่องเขียนไฟล์ไม่ทัน -> ข้ามเฟรม
                pass
            self.next_t += 1 / REC_FPS

    def stop(self):
        self.q.put(None)
        self.q = None
        return self.path


def save_screenshot(screen):
    os.makedirs(REC_DIR, exist_ok=True)
    path = os.path.join(REC_DIR, time.strftime("particles_%Y%m%d_%H%M%S.png"))
    pygame.image.save(screen, path)
    return path
