"""thread กล้อง: อ่านภาพ -> หามือ -> ส่งข้อมูลให้ thread วาด (ผ่าน shared_data)"""

import math
import os
import sys
import time

import cv2
import mediapipe as mp
import numpy as np

from .config import (
    BG_DIM,
    CAPTURE_HOLD,
    FIST_FRAMES,
    GRACE,
    H,
    LOCK_BACK,
    LOCK_DROP,
    LOCK_MAX,
    MIN_FRAME,
    PEN_COLORS,
    REARM,
    STEADY_C,
    STEADY_S,
    SWAP_HANDS,
    W,
    shared_data,
)
from .hands import HandSystem, OneEuro, classify, palm_center, pinch_dist, pinch_edge
from .portal import frame_area, frame_shape, render_portal, save_capture


def open_camera():
    """เปิดกล้อง: บน Mac ใช้ AVFoundation (เสถียรกว่า) และขอ 30 fps"""
    if sys.platform == "darwin":
        cap = cv2.VideoCapture(0, cv2.CAP_AVFOUNDATION)
    else:
        cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv2.CAP_PROP_FPS, 30)
    return cap


def camera_loop():
    """thread กล้อง: อ่านภาพ -> หามือ 2 ข้าง -> แยกบทบาท -> กรองให้นิ่ง -> ส่งให้ thread วาด"""
    cap = open_camera()
    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(max_num_hands=2,
                           model_complexity=1,          # 1 = แม่นกว่า (0 = เร็วกว่า)
                           min_detection_confidence=0.6,
                           min_tracking_confidence=0.6)
    drawer = mp.solutions.drawing_utils
    style_c = drawer.DrawingSpec(color=(255, 200, 80), thickness=2, circle_radius=2)   # มือคุม = ฟ้า
    style_p = drawer.DrawingSpec(color=(80, 120, 255), thickness=2, circle_radius=2)   # มือเขียน = ส้ม

    system = HandSystem()
    swap_seen = shared_data.get("swap", SWAP_HANDS)
    ctrl_stable, ctrl_last = 0, None
    ctrl_fired = False
    fist = 0
    portal_quad = None                                 # ((มุม 4 จุด, รูปทรง), เวลาที่เห็นล่าสุด)
    hold, cap_prev, cap_t = 0.0, None, 0.0             # แคปรูป: ถือนิ่งมานานเท่าไหร่ / มุมเฟรมก่อน
    cap_armed, cap_anchor = True, None                 # พร้อมแคปไหม / ตำแหน่งกรอบตอนแคปล่าสุด
    fps_t, fps_n = time.time(), 0
    while shared_data["running"]:
        ret, frame = cap.read()
        if not ret:
            time.sleep(0.01)
            continue
        now = time.time()
        fps_n += 1
        if now - fps_t >= 1.0:
            shared_data["cam_fps"] = fps_n / (now - fps_t)
            fps_t, fps_n = now, 0
        frame = cv2.flip(frame, 1)                     # กลับภาพเหมือนกระจก
        clean = frame.copy()                           # ภาพไม่มีเส้นโครงมือ (ใช้เป็นพื้นหลังตอนเขียน)
        res = hands.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

        swap = shared_data.get("swap", SWAP_HANDS)
        if swap != swap_seen:                          # กด H -> สลับบทบาทมือที่กำลังติดตามอยู่ทันที
            system.swap_roles()
            swap_seen = swap
        # ภาพถูกกลับแบบกระจกแล้ว -> ป้าย Left/Right ของ MediaPipe ตรงกับมือจริงของเรา
        found = [(h.landmark, hd.classification[0].label)
                 for h, hd in zip(res.multi_hand_landmarks or [], res.multi_handedness or [])]
        got = system.assign(found, now, swap)
        tc = system.tracks["ctrl"]                    # มือคุมอนุภาค "ยืนยันแล้ว" = ป้ายส่วนใหญ่ (หลายเฟรม) ตรงบทบาท
        shared_data["ctrl_ok"] = (tc.alive(now) and len(tc.votes) >= 6
                                  and sum(v == "ctrl" for v in tc.votes) / len(tc.votes) >= 0.7)
        for h in res.multi_hand_landmarks or []:      # วาดเส้นโครงมือ (สีตามบทบาท)
            role = "ctrl" if got["ctrl"] is h.landmark else "pen" if got["pen"] is h.landmark else None
            if role:
                st = style_c if role == "ctrl" else style_p
                drawer.draw_landmarks(frame, h, mp_hands.HAND_CONNECTIONS, st, st)

        # ================= โหมดพอร์ทัล =================
        if shared_data.get("mode") == "portal":
            locked = False
            for role in ("ctrl", "pen"):
                lm, tr = got[role], system.tracks[role]
                if lm is None:
                    continue
                if pinch_edge(tr, lm, 4, 20, "filter"):   # ก้อยแตะโป้ง (มือไหนก็ได้) = ฟิลเตอร์ถัดไป
                    shared_data["filter"] = shared_data.get("filter", 0) + 1
                # ล็อกกรอบ: ตอนโป้งเคลื่อนเข้าหาก้อย มุมกรอบของมือนี้ค้างไว้ที่เดิม (กรอบไม่บิด)
                d = pinch_dist(lm, 4, 20)
                base_d = float(np.median([hd for hd, _ in tr.corner_hist])) if len(tr.corner_hist) >= 8 else None
                if not tr.locked and base_d and d < base_d * LOCK_DROP:
                    # โป้งหดเข้าหาก้อยชัดเจน (เทียบกับท่าตัว L ปกติของมือนี้) -> ล็อก + ย้อนมุมกรอบไปก่อนโป้งขยับ
                    tr.locked, tr.lock_far, tr.lock_t = True, base_d, now
                    for hd, corners in reversed(tr.corner_hist):
                        if hd >= base_d * 0.9:
                            tr.corner_last = dict(corners)
                            break
                elif tr.locked and (d > tr.lock_far * LOCK_BACK or now - tr.lock_t > LOCK_MAX):
                    tr.locked = False                  # ปล่อยนิ้วแล้ว (หรือล็อกนานเกิน) -> ปลดล็อก
                    tr.corner_hist = []
                if not tr.locked:
                    tr.corner_hist = (tr.corner_hist + [(d, dict(tr.corner_last))])[-30:]   # ~1 วินาทีล่าสุด
                locked |= tr.locked
            if got["ctrl"] is not None and got["pen"] is not None:
                pts = []
                for role in ("ctrl", "pen"):            # มุมกรอบ = ปลายโป้ง + ปลายชี้ ของสองมือ
                    lm, tr = got[role], system.tracks[role]
                    order = (4, 8) if role == "ctrl" else (8, 4)   # โป้ง1, ชี้1, ชี้2, โป้ง2
                    for idx in order:
                        if tr.locked and idx in tr.corner_last:
                            pts.append(tr.corner_last[idx])       # ล็อกอยู่ -> ใช้ตำแหน่งเดิม
                        else:
                            f = tr.f_corner.setdefault(idx, OneEuro())
                            tr.corner_last[idx] = f((lm[idx].x * W, lm[idx].y * H), now)
                            pts.append(tr.corner_last[idx])
                dirs = []
                for role in ("ctrl", "pen"):            # ทิศนิ้วชี้ (โคน -> ปลาย) ของแต่ละมือ บนจอ
                    lm = got[role]
                    v = np.array([(lm[8].x - lm[5].x) * W, (lm[8].y - lm[5].y) * H])
                    dirs.append(v / (np.linalg.norm(v) + 1e-6))
                portal_quad = (frame_shape(pts, dirs), now)
            if portal_quad and now - portal_quad[1] > GRACE:
                portal_quad = None                     # มือหายไปนาน -> ไม่มีกรอบ
            frame_now = portal_quad[0] if portal_quad else None

            # ---- แคปรูป: ถือกรอบนิ่ง CAPTURE_HOLD วินาที ----
            if frame_now is not None and not locked:
                q = frame_now[0]
                c = q.mean(0)                          # จุดกลางกรอบ
                dg = float(np.linalg.norm(q.max(0) - q.min(0)))   # ขนาดกรอบ (เส้นทแยงมุม)
                if not cap_armed:
                    if np.linalg.norm(c - cap_anchor) > REARM * dg:
                        cap_armed, cap_prev = True, None   # ขยับมือแล้ว -> พร้อมแคปรอบใหม่
                elif frame_area(frame_now) < MIN_FRAME:
                    cap_prev, hold = None, 0.0         # กรอบเล็กเกินไป -> ไม่นับ
                elif (cap_prev is None or np.linalg.norm(c - cap_prev[0]) > STEADY_C * cap_prev[1]
                      or abs(dg - cap_prev[1]) > STEADY_S * cap_prev[1]):
                    cap_prev, hold = (c, dg), 0.0      # กรอบเลื่อน/ย่อขยาย -> เริ่มนับใหม่จากตรงนี้
                else:
                    hold += now - cap_t                # ปลายนิ้วสั่นนิดหน่อยไม่เป็นไร -> นับต่อ
                cap_t = now
            else:
                hold, cap_prev, cap_t = 0.0, None, now
            extra = {"hold": hold if cap_armed else 0.0, "locked": locked}
            fidx = shared_data.get("filter", 0)
            try:
                img, content = render_portal(clean, frame_now, fidx, now, extra=extra)
                if cap_armed and hold >= CAPTURE_HOLD:            # แชะ!
                    cap_armed, hold, cap_anchor = False, 0.0, frame_now[0].mean(0)   # แคปครั้งเดียว ไม่ว่าสำเร็จไหม
                    try:
                        full, _ = render_portal(clean, frame_now, fidx, now, hud=False)
                        paths = save_capture(full, content, frame_now)
                        shared_data["flash_t"] = time.time()
                        shared_data["toast_msg"] = "photo saved: " + os.path.basename(paths[1] or paths[0])
                    except Exception as ex:
                        print("capture error:", repr(ex))
                        shared_data["toast_msg"] = "photo failed: " + type(ex).__name__
                shared_data["portal_frame"] = img
            except Exception as ex:                     # ฟิลเตอร์/บันทึกพัง -> ข้าม แต่กล้องต้องไม่ดับ
                print("portal error:", repr(ex))
                shared_data["filter"] = fidx + 1
            small = cv2.resize(frame, (320, 240))
            shared_data["frame"] = cv2.cvtColor(small, cv2.COLOR_BGR2RGB)
            continue

        # ================= มือคุมอนุภาค =================
        tr = system.tracks["ctrl"]
        lm = got["ctrl"]
        if lm is not None:
            g = classify(tr, lm)
            hx, hy = tr.f_palm(palm_center(lm), now)
            # ขนาดมือ = ระยะจากข้อมือถึงโคนนิ้วกลาง (มือใกล้กล้อง = ค่ามาก)
            hsize = float(tr.f_size([math.hypot(lm[0].x - lm[9].x, lm[0].y - lm[9].y)], now)[0])
            shared_data["hand"] = (hx, hy, hsize)
            # มุมเอียงมือ: เส้นจากข้อมือ (0) ไปโคนนิ้วกลาง (9)  (คูณ 640/480 ให้สเกล x,y เท่ากัน)
            ddx = (lm[9].x - lm[0].x) * 640 / 480
            ddy = lm[9].y - lm[0].y
            shared_data["roll"] = float(tr.f_roll([math.atan2(ddx, -ddy)], now)[0])
            # 🤙 ก้อยอย่างเดียว = แรงผลัก/ดูด
            if g == "pinky":
                shared_data["tip"] = tuple(tr.f_pinky((lm[20].x, lm[20].y), now))
            else:
                shared_data["tip"] = None
            shape = {"draw": "planet", "pinky": None}.get(g, g)   # ☝️ = ดาวเคราะห์, 🤙 = ไม่เปลี่ยนรูป
            ctrl_stable = ctrl_stable + 1 if shape == ctrl_last else 0
            ctrl_last = shape
            if ctrl_stable == 0:
                ctrl_fired = False                     # ทำท่าใหม่ -> เปลี่ยนรูปได้อีก 1 ครั้ง
            if shape and ctrl_stable >= 5 and shared_data["ctrl_ok"] and not ctrl_fired:
                ctrl_fired = True                      # เปลี่ยนรูปครั้งเดียวต่อท่า (มือที่ยืนยันแล้วเท่านั้น)
                shared_data["shape"] = shape
        elif not tr.alive(now):                        # หายไปนานเกิน GRACE -> ถือว่าเอามือลงแล้ว
            ctrl_stable, ctrl_last = 0, None
            tr.fingers, tr.prob, tr.last_g = None, None, None
            shared_data["hand"] = None
            shared_data["tip"] = None
            shared_data["roll"] = 0.0
        # (หายไปแป๊บเดียว -> ไม่ทำอะไร ใช้ค่าเดิมไปก่อน)

        # ================= มือเขียน =================
        tr = system.tracks["pen"]
        lm = got["pen"]
        if lm is not None:
            g = classify(tr, lm)
            if g in ("middle", "pinky"):
                g = None
            # มือเขียน: ยางลบ = ชู ชี้+กลาง+นาง แต่พับก้อย (เช็กจากนิ้วโดยตรง ไม่ต้องรอโมเดลมั่นใจ)
            if tr.fingers == [True, True, True, False]:
                g = "erase"
            elif g == "explode":
                g = "lift"                             # ✋ แบมือ = ยกปากกา (ขยับไปที่อื่นได้ ไม่มีเส้น)
            elif g == "heart":
                g = None                               # ✌️ ไม่ใช้กับมือเขียนแล้ว
            # 👌 โป้งแตะชี้ (นิ้วกลาง นาง ก้อย ชูอยู่) = เปลี่ยนสีปากกา
            if tr.fingers[1:] == [True, True, True] and pinch_edge(tr, lm, 4, 8, "color"):
                shared_data["pen_color"] = (shared_data.get("pen_color", 0) + 1) % len(PEN_COLORS)
            shared_data["lraw"] = tr.gesture(g, 3)     # ☝️ เขียน / ✋ ยก / 3 นิ้ว ลบ / ✊ แปลง
            shared_data["lpen"] = tuple(tr.f_tip((lm[8].x, lm[8].y), now))   # ปลายนิ้วชี้ = ปากกา
            shared_data["lpalm"] = tuple(tr.f_palm(palm_center(lm), now))
            fist = fist + 1 if tr.cur == "gather" else 0
            shared_data["lfist"] = min(fist / FIST_FRAMES, 1.0)   # ความคืบหน้า (โชว์วงแหวน)
            if fist == FIST_FRAMES:                    # กำมือค้าง ~1 วิ -> หมึกกลายเป็นอนุภาค
                shared_data["convert"] = True
        elif not tr.alive(now):
            fist = 0
            tr.fingers, tr.cur, tr.cnt = None, None, 0
            tr.prob, tr.last_g = None, None
            shared_data["lfist"] = 0.0
            shared_data["lraw"] = None
            shared_data["lpen"] = None
            shared_data["lpalm"] = None
        if tr.alive(now):                              # ภาพกล้องเต็มจอ (มืดลง) เป็นพื้นหลังตอนเขียน
            big = cv2.resize(clean, (W, H)) * BG_DIM
            shared_data["frame_big"] = cv2.cvtColor(big.astype(np.uint8), cv2.COLOR_BGR2RGB)

        cv2.putText(frame, f"shape: {shared_data['shape']}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        # macOS ห้ามเปิดหน้าต่างจาก thread รอง -> ส่งภาพไปให้ thread หลักวาดแทน
        small = cv2.resize(frame, (320, 240))
        shared_data["frame"] = cv2.cvtColor(small, cv2.COLOR_BGR2RGB)

    cap.release()
