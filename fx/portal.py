"""โหมดพอร์ทัล: ฟิลเตอร์ภาพ, รูปทรงกรอบ (สี่เหลี่ยม/ตัว X), วาดกรอบ, แคปรูป"""

import math
import os
import time

import cv2
import numpy as np

from .config import CAPTURE_HOLD, H, REC_DIR, RING_AFTER, W


# ================= โหมดพอร์ทัล (กรอบจากสองมือ + ฟิลเตอร์) =================
def f_sketch(img, t):
    """ภาพสเก็ตช์ดินสอ: เทาๆ -> กลับสี -> เบลอ -> หาร (เทคนิค color dodge)"""
    g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(255 - g, (21, 21), 0)
    s = cv2.divide(g, 255 - blur, scale=256)
    return cv2.cvtColor(s, cv2.COLOR_GRAY2BGR)


def f_glitch(img, t):
    """แยกช่องสีแดง/น้ำเงินให้เหลื่อมกัน + แถบภาพกระตุกเป็นช่วงๆ"""
    b, g, r = cv2.split(img)
    sh = 6 + int(5 * math.sin(t * 9))
    out = cv2.merge([np.roll(b, -sh, 1), g, np.roll(r, sh, 1)])
    h = out.shape[0]
    rs = np.random.default_rng(int(t * 12))            # สุ่มแถบใหม่ ~12 ครั้ง/วินาที
    for _ in range(4):
        y, hh = int(rs.integers(0, max(h - 1, 1))), int(rs.integers(3, 18))
        out[y:y + hh] = np.roll(out[y:y + hh], int(rs.integers(-25, 25)), 1)
    return out


def f_neon(img, t):
    """เส้นขอบเรืองแสงสีฟ้า + ชมพู บนพื้นดำ"""
    g = cv2.GaussianBlur(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), (5, 5), 0)
    e = cv2.dilate(cv2.Canny(g, 50, 130), np.ones((2, 2), np.uint8))
    glow = cv2.GaussianBlur(e, (0, 0), 5).astype(float)
    ef = e.astype(float)
    out = ef[..., None] * np.array([1.0, 1.0, 0.1]) + glow[..., None] * np.array([1.6, 0.2, 1.6])
    return np.clip(out, 0, 255).astype(np.uint8)


def f_pixel(img, t):
    h, w = img.shape[:2]
    small = cv2.resize(img, (max(1, w // 14), max(1, h // 14)), interpolation=cv2.INTER_LINEAR)
    return cv2.resize(small, (w, h), interpolation=cv2.INTER_NEAREST)


def f_thermal(img, t):
    return cv2.applyColorMap(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), cv2.COLORMAP_INFERNO)


def f_darkmode(img, t):
    """ขาวดำแบบฟิล์ม (คล้าย Gritty ในไอจี):
    ไม่มีสี + ดำไม่สนิท/ขาวไม่จ้า (โทนหม่นแบบฟิล์ม) + เม็ดเกรน + จุดฝุ่นขาวเล็กๆ"""
    g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255
    g = np.clip((g - 0.5) * 1.15 + 0.5, 0, 1)                 # คอนทราสต์เพิ่มนิดเดียว
    g = 0.15 + g * 0.76                                       # ยกเงาขึ้น (ดำ -> เทาเข้ม), กดไฮไลต์ (ขาว -> ขาวนวล)
    h, w = g.shape
    rs = np.random.default_rng(int(t * 24))                   # เกรนเปลี่ยน ~24 ครั้ง/วิ เหมือนฟิล์มเดิน
    g += rs.normal(0, 0.045, (h, w)).astype(np.float32)       # เม็ดเกรน
    out = np.clip(g * 255, 0, 255).astype(np.uint8)
    for _ in range(int(rs.integers(2, 7))):                   # จุดฝุ่น/รอยขีดขาวเล็กๆ
        x, y = int(rs.integers(0, w)), int(rs.integers(0, h))
        if rs.random() < 0.7:
            cv2.circle(out, (x, y), int(rs.integers(1, 3)), 235, -1, cv2.LINE_AA)
        else:
            cv2.line(out, (x, y), (x + int(rs.integers(-6, 6)), y + int(rs.integers(4, 14))), 220, 1, cv2.LINE_AA)
    return cv2.cvtColor(out, cv2.COLOR_GRAY2BGR)


PORTAL_FILTERS = [("SKETCH", f_sketch), ("GLITCH", f_glitch), ("NEON", f_neon),
                  ("PIXEL", f_pixel), ("THERMAL", f_thermal), ("DARKMODE", f_darkmode)]


def order_quad(pts):
    """เรียง 4 จุดเป็นรูปสี่เหลี่ยม (วนรอบจุดศูนย์กลาง) ไม่ให้เส้นไขว้กัน"""
    c = pts.mean(0)
    ang = np.arctan2(pts[:, 1] - c[1], pts[:, 0] - c[0])
    return pts[np.argsort(ang)]


def _cross(o, a, b):
    return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])


def _segments_cross(p1, p2, p3, p4):
    """เส้น p1-p2 กับ p3-p4 ตัดกันไหม"""
    d1, d2 = _cross(p3, p4, p1), _cross(p3, p4, p2)
    d3, d4 = _cross(p1, p2, p3), _cross(p1, p2, p4)
    return (d1 * d2 < 0) and (d3 * d4 < 0)


def frame_shape(pts, dirs):
    """pts = 4 มุม (ปลายโป้ง/ชี้ ของสองมือ), dirs = ทิศนิ้วชี้ของมือ 1 และมือ 2
    - นิ้วชี้สองมือชี้ทางเดียวกัน (เช่น ขึ้นทั้งคู่)   -> สี่เหลี่ยม
    - นิ้วชี้ชี้สวนทางกัน (ข้างหนึ่งกลับหัว)          -> นาฬิกาทราย (ตัว X)
    คืน (จุดเรียงเป็นรูปแล้ว, ชื่อรูปทรง)"""
    p = np.asarray(pts, dtype=float)
    c = p.mean(0)
    p = p[np.argsort(np.arctan2(p[:, 1] - c[1], p[:, 0] - c[0]))]   # วนรอบ (ไม่ไขว้)
    s_, d_ = p.sum(1), p[:, 0] - p[:, 1]
    tl, br = p[np.argmin(s_)], p[np.argmax(s_)]
    tr, bl = p[np.argmax(d_)], p[np.argmin(d_)]
    da, db = dirs
    if float(np.dot(da, db)) < -0.3 and len({tuple(tl), tuple(tr), tuple(br), tuple(bl)}) == 4:
        return np.array([tl, tr, bl, br]), "hourglass"   # ขอบบน + ขอบล่าง แล้วไขว้ตรงกลาง ⧗
    return p, "quad"


def frame_area(frame):
    """พื้นที่ในกรอบ เป็นสัดส่วนของจอ (0-1) — วาดหน้ากากขนาดเล็ก 1/4 จอ ให้คำนวณเร็ว"""
    q = (frame[0] / 4).astype(np.int32)
    m = frame_mask(q, frame[1], (H // 4, W // 4))
    return float((m > 0).mean())


def frame_mask(q, shape, size):
    """หน้ากากพื้นที่ในกรอบ (ใช้ได้ทั้งสี่เหลี่ยมและนาฬิกาทราย)"""
    mask = np.zeros(size, np.uint8)
    cv2.fillPoly(mask, [q], 255)                       # ใช้ได้ทั้งสี่เหลี่ยม (รวมรูปเว้า) และรูปไขว้ (เติมสามเหลี่ยมทั้ง 2 ชิ้น)
    return mask


def render_portal(clean, frame, fidx, t, hud=True, extra=None):
    """ภาพกล้องเต็มจอ + ฟิลเตอร์เฉพาะในกรอบ -> RGB (H, W, 3)
    frame = (จุด 4 มุม, "quad"/"hourglass") หรือ None / extra = ข้อมูลวงแหวนนับถอยหลัง"""
    base = cv2.resize(clean, (W, H))
    out = (base * 0.8).astype(np.uint8)                # นอกกรอบมืดลงนิด ให้กรอบเด่น
    name, fn = PORTAL_FILTERS[fidx % len(PORTAL_FILTERS)]
    content = None                                     # (ภาพในกรอบ, หน้ากาก, x0, y0) ไว้ใช้ตอนแคป
    if frame is not None:
        q = frame[0].astype(np.int32)
        shape = frame[1]
        x0, y0 = np.clip(q.min(0), 0, [W - 1, H - 1])
        x1, y1 = np.clip(q.max(0) + 1, 1, [W, H])
        if x1 - x0 > 8 and y1 - y0 > 8:
            mask = frame_mask(q - [x0, y0], shape, (y1 - y0, x1 - x0))
            if (mask > 0).sum() > 1500:                # กรอบเล็กเกินไป (มือชิดกัน) -> ไม่วาด
                roi = base[y0:y1, x0:x1]
                filt = fn(roi, t)
                m = mask.astype(bool)
                out[y0:y1, x0:x1][m] = filt[m]
                content = (filt, mask, x0, y0)
                cv2.polylines(out, [q], True, (255, 255, 255), 2, cv2.LINE_AA)
                for p in q:
                    cv2.circle(out, tuple(int(v) for v in p), 6, (255, 255, 255), -1, cv2.LINE_AA)
                if hud:
                    top = q[np.argmin(q[:, 1])]
                    label = f"PORTAL: {name}" + ("  X" if shape == "hourglass" else "")
                    cv2.putText(out, label, (int(top[0]) - 60, max(20, int(top[1]) - 14)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)
    if hud:
        if extra and extra.get("hold", 0) > RING_AFTER and frame is not None:   # วงแหวนนับถอยหลังแคปรูป
            c = tuple(int(v) for v in frame[0].mean(0))
            prog = min((extra["hold"] - RING_AFTER) / (CAPTURE_HOLD - RING_AFTER), 1.0)
            cv2.circle(out, c, 38, (40, 40, 40), 6, cv2.LINE_AA)
            cv2.ellipse(out, c, (38, 38), -90, 0, 360 * prog, (60, 210, 255), 6, cv2.LINE_AA)
            n = str(max(1, math.ceil(CAPTURE_HOLD - extra["hold"])))
            cv2.putText(out, n, (c[0] - 12, c[1] + 13), cv2.FONT_HERSHEY_SIMPLEX, 1.2,
                        (255, 255, 255), 3, cv2.LINE_AA)
        if extra and extra.get("locked"):
            cv2.putText(out, "frame locked", (14, 58), cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                        (200, 200, 200), 1, cv2.LINE_AA)
        cv2.putText(out, f"PORTAL MODE  filter: {name}", (14, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                    (60, 220, 255), 2, cv2.LINE_AA)
        cv2.putText(out, "L-shapes = frame (flip one = X)   pinky+thumb = filter   big frame + hold still 2.5s = photo   P = back",
                    (14, H - 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1, cv2.LINE_AA)
    return cv2.cvtColor(out, cv2.COLOR_BGR2RGB), content


def save_capture(full_rgb, content, frame):
    """บันทึก: ภาพเต็มจอ + ภาพเฉพาะในกรอบ (สี่เหลี่ยม -> ดึงให้ตรง / นาฬิกาทราย -> พื้นโปร่ง)"""
    os.makedirs(REC_DIR, exist_ok=True)
    stamp = time.strftime("%Y%m%d_%H%M%S")
    full_path = os.path.join(REC_DIR, f"portal_{stamp}_full.png")
    cv2.imwrite(full_path, cv2.cvtColor(full_rgb, cv2.COLOR_RGB2BGR))
    if content is None:
        return full_path, None
    filt, mask, x0, y0 = content
    crop_path = os.path.join(REC_DIR, f"portal_{stamp}_frame.png")
    pts, shape = frame
    if shape == "quad":
        # เรียงมุม: บนซ้าย, บนขวา, ล่างขวา, ล่างซ้าย แล้วดึงภาพให้เป็นสี่เหลี่ยมตรง
        hull = cv2.convexHull(pts.astype(np.float32))[:, 0]
        if len(hull) == 4:
            s, d = hull.sum(1), np.diff(hull, axis=1)[:, 0]
            src = np.float32([hull[np.argmin(s)], hull[np.argmin(d)], hull[np.argmax(s)], hull[np.argmax(d)]])
            w = int(max(np.linalg.norm(src[0] - src[1]), np.linalg.norm(src[3] - src[2])))
            h = int(max(np.linalg.norm(src[0] - src[3]), np.linalg.norm(src[1] - src[2])))
            if w > 10 and h > 10:
                full_filtered = np.zeros((H, W, 3), np.uint8)
                full_filtered[y0:y0 + filt.shape[0], x0:x0 + filt.shape[1]] = filt
                M = cv2.getPerspectiveTransform(src.astype(np.float32), np.float32([[0, 0], [w, 0], [w, h], [0, h]]))
                cv2.imwrite(crop_path, cv2.warpPerspective(full_filtered, M, (w, h)))
                return full_path, crop_path
    bgra = cv2.cvtColor(filt, cv2.COLOR_BGR2BGRA)      # นาฬิกาทราย: นอกกรอบโปร่งใส
    bgra[..., 3] = mask
    cv2.imwrite(crop_path, bgra)
    return full_path, crop_path
