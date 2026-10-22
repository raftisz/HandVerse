"""
HandVerse — ควบคุมอนุภาค เขียนกลางอากาศ และเปิดพอร์ทัลฟิลเตอร์ ด้วยมือผ่านกล้อง
รัน:  python main.py
-----------------------------------------------------
มือซ้าย = คุมอนุภาค
  ✋ แบมือ               -> อนุภาคแตกกระจาย (สีรุ้ง)
  ✊ กำมือ               -> อนุภาครวมกันเป็นลูกพลังงาน
  ☝️ ชูนิ้วชี้             -> ดาวเคราะห์มีวงแหวน
  ✌️ ชูสองนิ้ว           -> หัวใจ
  🖕 ชูนิ้วกลางอย่างเดียว -> ตัวหนังสือ FUCK YOU (สีไฟ)
  🤙 ชูนิ้วก้อยอย่างเดียว -> ปลายก้อยผลัก/ดูดอนุภาค (Space สลับ ผลัก/ดูด)
  ขยับมือ = รูปลอยตาม, มือใกล้กล้อง = ซูม, เอียงมือ = รูปเอียง
มือขวา = เขียน (ภาพกล้องจะขึ้นเป็นพื้นหลังตอนกำลังเขียน)
  ☝️ เขียน, ✋ แบมือ = ยกปากกา, 3 นิ้ว (ชี้+กลาง+นาง ก้อยพับ) = ยางลบ, ✊ กำค้าง 3 วิ = ตัวหนังสือกลายเป็นอนุภาค
👌 (มือเขียน) โป้งแตะชี้ = เปลี่ยนสีปากกา
P = โหมดพอร์ทัล: สองมือทำตัว L เป็นกรอบ ภาพในกรอบติดฟิลเตอร์, ก้อยแตะโป้ง = เปลี่ยนฟิลเตอร์
      กลับด้านตัว L ข้างหนึ่ง = กรอบตัว X (นาฬิกาทราย), ถือกรอบนิ่ง 2.5 วิ (กรอบต้องใหญ่ ≥12% ของจอ) = แคปรูป (เก็บใน recordings)
ปุ่ม: R = อัดวิดีโอ (กดอีกครั้งหยุด), S = แคปหน้าจอ  -> เก็บในโฟลเดอร์ recordings
      C = ลบกระดาน, H = สลับหน้าที่มือซ้าย/ขวา, M = สลับ โมเดล/กฎเดิม, ESC = ออก
(ในโค้ด ตัวแปรชื่อ right/left หมายถึง "มือคุมอนุภาค"/"มือเขียน" ส่วนข้างจริงกำหนดด้วย SWAP_HANDS)
"""

import math
import os
import threading
import time

import numpy as np
import pygame

from fx.camera import camera_loop
from fx.config import ERASER_R, H, N, PEN_COLORS, PEN_W, SWAP_HANDS, W, shared_data
from fx.hands import MODEL
from fx.portal import PORTAL_FILTERS
from fx.recorder import Recorder, save_screenshot
from fx.shapes import SHAPES, blackhole_points, blackhole_update, hsv_to_rgb, ink_to_particles


# ---------------- ส่วนวาดอนุภาค ----------------
def pressed(e, letter):
    """เช็กปุ่มจาก "ตำแหน่งปุ่ม" บนคีย์บอร์ด (scancode) -> ใช้ได้ทั้งแป้นไทยและอังกฤษ
    (ถ้าแป้นเป็นภาษาไทย ปุ่ม R จะพิมพ์ "พ" ถ้าเช็กแค่ตัวอักษร โปรแกรมจะไม่รู้ว่ากด R)"""
    if e.type != pygame.KEYDOWN:
        return False
    return (e.key == getattr(pygame, "K_" + letter)
            or getattr(e, "scancode", None) == getattr(pygame, "KSCAN_" + letter.upper()))


def main():
    pygame.init()
    screen = pygame.display.set_mode((W, H))
    pygame.display.set_caption("HandVerse")
    clock = pygame.time.Clock()

    cam = threading.Thread(target=camera_loop, daemon=True)
    cam.start()

    pos = np.random.normal(0, 200, (N, 3))      # ตำแหน่งปัจจุบันของอนุภาค
    color = np.full((N, 3), 150.0)
    current = None
    target, target_col = pos.copy(), color.copy()
    buf = np.zeros((W, H, 3))                   # บัฟเฟอร์ภาพ (ทำให้มีหางเรืองแสง)
    angle = 0.0
    # ตำแหน่ง/ขนาดที่ตามมือ (ค่อยๆ เลื่อนตาม ไม่กระตุก)
    off_x, off_y, scale = 0.0, 0.0, 1.0
    roll = 0.0                                   # มุมเอียงรูป (ตามมือ)
    # แรงจากปลายนิ้ว: อนุภาคถูกผลักเบี่ยงออกจากตำแหน่งเดิม (disp) แล้วสปริงดึงกลับ
    disp = np.zeros((N, 2))
    vel = np.zeros((N, 2))
    repel = True                                 # True = ผลัก, False = ดูด (หลุมดำ)
    font = pygame.font.SysFont(None, 26)

    # ---- กระดานเขียน (มือซ้าย) ----
    ink = pygame.Surface((W, H))                 # หมึก ตัวเส้น (พื้นดำ)
    glow_ink = pygame.Surface((W, H))            # หมึก แสงเรืองรอบเส้น (แยกชั้น ไม่ให้ทับตัวเส้น)
    pen_s = None                                 # ตำแหน่งปากกาแบบนุ่ม
    last_pc = shared_data.get("pen_color", 0)    # สีปากกาล่าสุด (ไว้แจ้งตอนเปลี่ยน)
    last_filter = shared_data.get("filter", 0)

    def finish_frame():
        """จบเฟรม: อัดวิดีโอ + ป้าย REC + ข้อความแจ้ง + fps -> แสดงผล (ใช้ทั้งโหมดอนุภาคและพอร์ทัล)"""
        if rec.on:
            rec.add(screen)                              # เก็บเฟรม (ก่อนวาดป้าย REC จะได้ไม่ติดในวิดีโอ)
            secs = int(time.time() - rec.start_t)
            if int(time.time() * 2) % 2 == 0:            # จุดแดงกระพริบ
                pygame.draw.circle(screen, (255, 40, 40), (W // 2 - 50, 22), 8)
            screen.blit(font.render(f"REC {secs // 60}:{secs % 60:02d}", True, (255, 80, 80)), (W // 2 - 36, 13))
        if toast and time.time() - toast_t < 2.5:
            screen.blit(font.render(toast, True, (120, 255, 140)), (W // 2 - 150, 40))
        mode_g = "model" if MODEL is not None and shared_data.get("use_model", True) else "rules"
        fps_txt = f"fps {clock.get_fps():.0f}  cam {shared_data.get('cam_fps', 0):.0f}  {mode_g}"
        screen.blit(font.render(fps_txt, True, (120, 120, 120)), (W - 210, 10))
        pygame.display.flip()
        clock.tick(60)

    last_pt = None                               # จุดก่อนหน้าของเส้น (None = เพิ่งวางปากกา)
    ink_len = 0.0                                # ความยาวหมึกสะสม (ใช้ไล่สีรุ้ง)
    bg_alpha = 0.0                               # ความชัดของกระดาน (ภาพกล้อง + หมึก) 0-1
    last_pen_t = -1e9                            # เวลาที่เห็นมือเขียนล่าสุด
    p_alpha = 1.0                                # ความชัดของอนุภาค 0-1
    just_converted = False                       # เพิ่งกำมือแปลงหมึกเป็นอนุภาค (ให้โชว์อนุภาคต่อ)
    rec = Recorder()                             # R = อัดวิดีโอ
    toast, toast_t = "", 0.0                     # ข้อความแจ้งเตือนสั้นๆ (เช่น บันทึกไฟล์แล้ว)
    has_ink = False                              # บนกระดานมีหมึกอยู่ไหม (ไม่มี = ไม่ต้องโชว์ยางลบ)

    while shared_data["running"]:
        for e in pygame.event.get():
            if e.type == pygame.QUIT or (e.type == pygame.KEYDOWN and e.key == pygame.K_ESCAPE):
                shared_data["running"] = False
            if e.type == pygame.KEYDOWN and e.key == pygame.K_SPACE:
                repel = not repel                # กด Space สลับ ผลัก/ดูด
            if pressed(e, "c"):
                ink.fill((0, 0, 0)); glow_ink.fill((0, 0, 0))   # กด C ลบกระดาน
                has_ink = False
            if pressed(e, "r"):     # R = เริ่ม/หยุด อัดวิดีโอ
                if rec.on:
                    toast, toast_t = "saved: " + os.path.basename(rec.stop()), time.time()
                else:
                    rec.start()
            if pressed(e, "s"):     # S = แคปหน้าจอ
                toast, toast_t = "saved: " + os.path.basename(save_screenshot(screen)), time.time()
            if pressed(e, "m"):                                   # M = สลับ โมเดล / กฎเดิม
                shared_data["use_model"] = not shared_data.get("use_model", True)
                toast, toast_t = ("gesture: MODEL" if shared_data["use_model"] else "gesture: RULES"), time.time()
            if pressed(e, "h"):
                shared_data["swap"] = not shared_data.get("swap", SWAP_HANDS)   # กด H สลับซ้าย/ขวา
            if pressed(e, "p"):                                   # P = สลับ โหมดพอร์ทัล / อนุภาค
                portal = shared_data.get("mode") != "portal"
                shared_data["mode"] = "portal" if portal else "particles"
                shared_data["portal_frame"] = None
                toast, toast_t = ("PORTAL MODE" if portal else "PARTICLE MODE"), time.time()

        msg = shared_data.pop("toast_msg", None)          # ข้อความจาก thread กล้อง (เช่น แคปรูปแล้ว)
        if msg:
            toast, toast_t = msg, time.time()
        pc = shared_data.get("pen_color", 0)
        if pc != last_pc:                                # 👌 เปลี่ยนสีปากกา -> แจ้งบนจอ
            toast, toast_t, last_pc = "pen color: " + PEN_COLORS[pc][0], time.time(), pc
        fi = shared_data.get("filter", 0)
        if fi != last_filter:
            toast, toast_t, last_filter = "filter: " + PORTAL_FILTERS[fi % len(PORTAL_FILTERS)][0], time.time(), fi

        # ================= โหมดพอร์ทัล: แค่แสดงภาพที่ thread กล้องทำไว้ =================
        if shared_data.get("mode") == "portal":
            pf = shared_data.get("portal_frame")
            if pf is not None and pf.shape[:2] == (H, W):
                pygame.surfarray.blit_array(screen, pf.swapaxes(0, 1))
                ft = time.time() - shared_data.get("flash_t", -9)
                if ft < 0.35:                                    # แฟลชขาวตอนแคป ค่อยๆ จางหาย
                    flash = pygame.Surface((W, H))
                    flash.fill((255, 255, 255))
                    flash.set_alpha(int(230 * (1 - ft / 0.35)))
                    screen.blit(flash, (0, 0))
            else:
                screen.fill((0, 0, 0))
                screen.blit(font.render("starting portal... show both hands", True, (200, 200, 200)), (14, 14))
            finish_frame()
            continue

        # ================= มือซ้าย: เขียน =================
        lraw = shared_data.get("lraw")
        lpen = shared_data.get("lpen")
        lpalm = shared_data.get("lpalm")
        if lpen:
            p = (lpen[0] * W, lpen[1] * H)               # ปากกาตรงนิ้ว 1:1 กับภาพกล้องเต็มจอ
            pen_s = p                                    # กรองให้นิ่งแล้วในกล้อง (One Euro)
        else:
            pen_s = None
        if lraw == "draw" and pen_s is not None:         # ☝️ เขียน
            if last_pt is not None:
                ink_len += math.hypot(pen_s[0] - last_pt[0], pen_s[1] - last_pt[1])
            fixed = PEN_COLORS[shared_data.get("pen_color", 0)][1]
            if fixed is None:                            # สีรุ้ง ไล่ตามความยาวเส้น
                c = hsv_to_rgb(np.array([(ink_len / 1500) % 1.0]), 1.0, 1.0)[0] * 255
            else:
                c = np.array(fixed, dtype=float)
            a = last_pt if last_pt else pen_s
            ip, iq = (int(a[0]), int(a[1])), (int(pen_s[0]), int(pen_s[1]))
            pygame.draw.line(glow_ink, c * 0.3, ip, iq, PEN_W * 3)   # เรืองแสงรอบเส้น
            pygame.draw.circle(glow_ink, c * 0.3, iq, PEN_W * 3 // 2)
            pygame.draw.line(ink, c, ip, iq, PEN_W)                  # ตัวเส้น
            pygame.draw.circle(ink, c, iq, PEN_W // 2)
            last_pt = pen_s
            has_ink = True
        else:
            last_pt = None                               # ยกปากกา -> เส้นใหม่ไม่ต่อกัน
        eraser = None
        if lraw == "erase" and pen_s is not None and has_ink:     # 3 นิ้ว (ก้อยพับ) = ยางลบ ที่ปลายนิ้วชี้
            eraser = (int(pen_s[0]), int(pen_s[1]))
            pygame.draw.circle(ink, (0, 0, 0), eraser, ERASER_R)
            pygame.draw.circle(glow_ink, (0, 0, 0), eraser, ERASER_R + PEN_W)

        if shared_data.pop("convert", False) and has_ink:   # ✊ มือเขียน: หมึก -> อนุภาค (ต้องมีหมึก)
            shared_data["shape"] = "ink"
            current = "ink"
            target, target_col = ink_to_particles(ink, N)
            pos, color = target.copy(), target_col.copy()   # วางอนุภาคตรงหมึกพอดี ไม่กระโดด
            off_x, off_y, scale, roll = 0.0, 0.0, 1.0, 0.0
            angle = round(angle / (2 * math.pi)) * 2 * math.pi
            disp[:] = 0; vel[:] = 0
            buf = pygame.surfarray.array3d(ink).astype(float)
            ink.fill((0, 0, 0)); glow_ink.fill((0, 0, 0))   # หมึกกลายเป็นอนุภาคแล้ว -> ล้างกระดาน
            has_ink = False
            just_converted = True

        # ================= มือขวา: เปลี่ยนรูปทรง =================
        if shared_data["shape"] != current:
            current = shared_data["shape"]
            target, target_col = SHAPES[current](N)
        if current == "blackhole":                 # หลุมดำขยับตลอด -> อัปเดตเป้าหมายทุกเฟรม
            blackhole_update()
            target, target_col = blackhole_points()

        # ค่อยๆ เคลื่อนไปหาเป้าหมาย + สั่นเล็กน้อยให้ดูมีชีวิต
        pos += (target - pos) * 0.06 + np.random.normal(0, 0.6, pos.shape)
        color += (target_col - color) * 0.05

        # หมุนรอบแกน Y แล้วฉายภาพแบบมีมุมมอง (perspective)
        if current in ("middle", "ink"):
            # ท่านี้หันหน้าเข้าจอตลอด (ค่อยๆ หมุนกลับมาด้านหน้า แทนการหมุนเอง)
            front = round(angle / (2 * math.pi)) * 2 * math.pi
            angle += (front - angle) * 0.08
        else:
            angle += 0.008
        ca, sa = math.cos(angle), math.sin(angle)
        x = pos[:, 0] * ca + pos[:, 2] * sa
        z = -pos[:, 0] * sa + pos[:, 2] * ca
        y = pos[:, 1]

        # ---- เอียงตามมือขวา (หมุนรอบแกนที่พุ่งออกจากจอ) ----
        want_roll = shared_data.get("roll", 0.0) * 1.2   # 1.2 = เอียงมือนิดเดียวรูปเอียงมากขึ้น
        roll += (want_roll - roll) * 0.15
        cr, sr = math.cos(-roll), math.sin(-roll)        # ลบ เพราะแกน y ของจอกลับหัว
        x, y = x * cr - y * sr, x * sr + y * cr
        # ---- ตามมือขวา ----
        hand = shared_data.get("hand")
        if hand:
            hx, hy, hsize = hand
            want_x = (hx - 0.5) * W * 1.3              # 1.3 = ขยับมือนิดเดียวรูปไปได้ไกล
            want_y = (hy - 0.5) * H * 1.3
            # ซูม: มือใกล้กล้อง = ขยาย ยกกำลัง 1.8 ทำให้ยิ่งใกล้ยิ่งขยายเร็ว
            want_s = np.clip((hsize / 0.18) ** 1.8, 0.3, 4.0)   # 0.18 = ขนาดมือปกติ, ใหญ่สุด 4 เท่า
        else:
            want_x, want_y, want_s = 0.0, 0.0, 1.0      # ไม่เห็นมือขวา -> กลับกลางจอ
        k = 0.15                                        # ความไวในการตาม (มาก = ตามเร็ว)
        off_x += (want_x - off_x) * k
        off_y += (want_y - off_y) * k
        scale += (want_s - scale) * k

        f = 600 / (600 + z * min(scale, 1.5))          # จำกัดความลึกตอนขยายมากๆ ไม่ให้ภาพบิดเบี้ยว
        px = W / 2 + off_x + x * f * scale
        py = H / 2 + off_y - y * f * scale

        # ---- แรงจากปลายนิ้วก้อยมือขวา (เฉพาะตอนชูก้อยอย่างเดียว) ----
        tip = shared_data.get("tip")
        tx = ty = None
        if tip:
            # แปลงพิกัดนิ้วให้ตรงกับระบบเดียวกับที่รูปทรงตามมือ
            tx = W / 2 + (tip[0] - 0.5) * W * 1.3
            ty = H / 2 + (tip[1] - 0.5) * H * 1.3
            dx = px + disp[:, 0] - tx
            dy = py + disp[:, 1] - ty
            d = np.sqrt(dx * dx + dy * dy) + 1e-3
            R = 140                                      # รัศมีแรง (พิกเซล)
            near = d < R
            power = (1 - d[near] / R) * 6.0              # ใกล้นิ้ว = แรงมาก
            sign = 1 if repel else -1
            vel[near, 0] += sign * dx[near] / d[near] * power
            vel[near, 1] += sign * dy[near] / d[near] * power
        vel -= disp * 0.04                               # สปริงดึงกลับรูปเดิม
        vel *= 0.86                                      # แรงเสียดทาน
        disp += vel

        sx = (px + disp[:, 0]).astype(int)
        sy = (py + disp[:, 1]).astype(int)

        ok = (sx >= 0) & (sx < W - 1) & (sy >= 0) & (sy < H - 1) & (f > 0)
        glow = {"explode": 1.8, "ink": 1.1}.get(current, 0.55)   # รูปที่อนุภาคห่างกัน -> เพิ่มความสว่าง
        sx, sy, c = sx[ok], sy[ok], color[ok] * glow

        buf *= 0.80                                 # จางภาพเก่า = หางเรืองแสง
        np.add.at(buf, (sx, sy), c)
        np.add.at(buf, (sx + 1, sy), c * 0.5)
        np.add.at(buf, (sx, sy + 1), c * 0.5)

        # ---- เลือกว่าจะโชว์อะไร ตามมือที่ยกขึ้นมา ----
        pen_up = lpen is not None                        # เห็นมือเขียน
        ctrl_up = shared_data.get("hand") is not None    # เห็นมือคุมอนุภาค
        if just_converted and (not pen_up or lraw == "draw"):
            just_converted = False                       # เอามือเขียนลง หรือเริ่มเขียนใหม่ -> กลับโหมดปกติ
        now_t = time.time()
        if pen_up:
            last_pen_t = now_t
        pen_recent = now_t - last_pen_t < 0.7            # เพิ่งเอามือเขียนลง (กันกล้องหามือหายแป๊บเดียว)
        ctrl_ok = ctrl_up and shared_data.get("ctrl_ok", False)
        show_board = pen_up and not just_converted       # ยกมือเขียน = เปิดกระดาน
        # อนุภาคโชว์เมื่อ: ยกมือคุม (ยืนยันแล้ว) / ไม่มีมือเขียนมาสักพัก / เพิ่งแปลงตัวหนังสือ
        # (มีมือที่ยังไม่ยืนยันว่าข้างไหน -> ซ่อนไว้ก่อน กันมือขวาที่ถูกบอกข้างผิดแว๊บเป็นอนุภาค)
        show_parts = ctrl_ok or (not pen_up and not ctrl_up and not pen_recent) or just_converted
        if not show_parts and (pen_up or pen_recent):
            p_alpha = 0.0                                # ยกมือเขียน -> ปิดอนุภาคทันที ไม่ต้องรอจาง
        writing = show_board
        bg_alpha += ((1.0 if show_board else 0.0) - bg_alpha) * 0.12   # ค่อยๆ จางเข้า/ออก
        p_alpha += ((1.0 if show_parts else 0.0) - p_alpha) * 0.12

        # ---- ประกอบภาพ: ภาพกล้อง (กระดาน) + อนุภาค + หมึก ----
        out = buf * p_alpha
        bg = shared_data.get("frame_big")
        if bg_alpha > 0.02 and bg is not None and bg.shape[:2] == (H, W):
            out = out + bg.swapaxes(0, 1) * bg_alpha
        pygame.surfarray.blit_array(screen, np.clip(out, 0, 255).astype(np.uint8))
        if bg_alpha > 0.02:                              # หมึกโชว์เฉพาะตอนเปิดกระดาน
            for layer in (glow_ink, ink):
                if bg_alpha < 0.98:                      # ระหว่างจางเข้า/ออก -> หรี่หมึกตาม
                    layer = layer.copy()
                    v = int(255 * bg_alpha)
                    layer.fill((v, v, v), special_flags=pygame.BLEND_MULT)
                screen.blit(layer, (0, 0), special_flags=pygame.BLEND_ADD)

        # ภาพกล้องเล็กมุมซ้ายบน (ซ่อนตอนเขียน เพราะมีภาพเต็มจออยู่แล้ว)
        cam_img = shared_data.get("frame")
        if cam_img is not None and bg_alpha < 0.5:
            surf = pygame.surfarray.make_surface(cam_img.swapaxes(0, 1))
            screen.blit(surf, (10, 10))
            pygame.draw.rect(screen, (80, 80, 80), (10, 10, 320, 240), 1)

        # วงกลมบอกตำแหน่งปลายนิ้วก้อย (ขึ้นเฉพาะตอนแรงทำงาน)
        if tx is not None and show_parts:
            ring = (120, 200, 255) if repel else (200, 120, 255)
            pygame.draw.circle(screen, ring, (int(tx), int(ty)), 10, 2)
        # ปากกา/ยางลบ มือซ้าย
        if eraser and show_board:
            pygame.draw.circle(screen, (255, 60, 60), eraser, ERASER_R, 2)
        elif pen_s is not None and lraw in ("draw", "lift") and show_board:
            pygame.draw.circle(screen, (255, 255, 255), (int(pen_s[0]), int(pen_s[1])),
                               6, 0 if lraw == "draw" else 2)    # ทึบ = เขียน, กลวง = ยกปากกา

        prog = shared_data.get("lfist", 0.0)
        if prog > 0.05 and pen_s is not None and has_ink:     # วงแหวนนับถอยหลัง ก่อนแปลงเป็นอนุภาค
            cx, cy = int(pen_s[0]), int(pen_s[1])
            pygame.draw.circle(screen, (90, 90, 90), (cx, cy), 34, 3)
            pygame.draw.arc(screen, (255, 210, 60), (cx - 34, cy - 34, 68, 68),
                            math.pi / 2, math.pi / 2 + prog * 2 * math.pi, 5)
            screen.blit(font.render("hold fist = to particles", True, (255, 210, 60)), (cx - 95, cy + 42))
        if writing:
            tip_txt = "PEN: 1 finger = write  open = lift  3 fingers = erase  OK sign = color  hold fist = particles"
            screen.blit(font.render(tip_txt, True, (200, 200, 200)), (14, H - 55))
        if show_parts:
            mode = "REPEL (Space = switch)" if repel else "BLACK HOLE (Space = switch)"
            screen.blit(font.render(mode + "   H = swap hands", True, (180, 180, 180)), (W - 410, H - 30))

        finish_frame()

    if rec.on:                                       # ปิดโปรแกรมระหว่างอัด -> บันทึกไฟล์ให้ก่อน
        rec.stop()
        time.sleep(0.5)
    pygame.quit()
    cam.join(timeout=1)




if __name__ == "__main__":
    main()
