"""จับมือ: นับนิ้ว, ตัดสินท่ามือ (กฎ/โมเดล), ตัวกรองมือสั่น, แยกมือซ้าย-ขวา, การแตะนิ้ว"""

import math

import numpy as np

from .config import GRACE, MATCH_DIST, MODEL_CONF, PINCH_RATIO, shared_data


# ---------------- ตรวจจับท่ามือ ----------------
def _v(lm, a, b):
    """เวกเตอร์ 3 มิติจากจุด a ไปจุด b (ปรับสเกล x,y,z ให้เท่ากัน ตามขนาดภาพ 640x480)"""
    return np.array([(lm[b].x - lm[a].x) * 640, (lm[b].y - lm[a].y) * 480, (lm[b].z - lm[a].z) * 640])


def finger_states(lm):
    """นิ้วชี้ กลาง นาง ก้อย: เหยียดอยู่ไหม (ดูจากความตรงของนิ้ว ไม่สนว่าชี้ไปทางไหน)
    นิ้วเหยียด = ท่อน โคน->ข้อกลาง กับ ข้อกลาง->ปลาย ชี้ไปทางเดียวกัน (มุมงอน้อย)
    + ปลายนิ้วอยู่ไกลข้อมือกว่าข้อกลางนิ้ว"""
    out = []
    for mcp, pip, dip, tip in [(5, 6, 7, 8), (9, 10, 11, 12), (13, 14, 15, 16), (17, 18, 19, 20)]:
        a, b = _v(lm, mcp, pip), _v(lm, pip, tip)
        cos = a @ b / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-6)
        farther = np.linalg.norm(_v(lm, 0, tip)) > np.linalg.norm(_v(lm, 0, pip))
        out.append(bool(cos > 0.6 and farther))   # cos > 0.6 = งอไม่เกิน ~53°
    return out


def detect_gesture(lm):
    """แปลงสถานะนิ้วเป็นท่ามือ"""
    up = finger_states(lm)      # [ชี้, กลาง, นาง, ก้อย]
    count = sum(up)
    if count >= 3:
        return "explode"        # แบมือ = แตกกระจาย
    if up == [True, False, False, False]:
        return "draw"           # ชูนิ้วชี้ (มือเขียน = ปากกา, มือคุม = ดาวเคราะห์)
    if up == [True, True, False, False]:
        return "heart"
    if up == [False, True, False, False]:
        return "middle"
    if count == 0:
        return "gather"         # กำมือ = รวมกัน
    return None


def palm_center(lm):
    """จุดกลางฝ่ามือ = ค่าเฉลี่ยของข้อมือ + โคนนิ้วทั้ง 4 (ค่า 0-1)"""
    ids = [0, 5, 9, 13, 17]
    return (sum(lm[i].x for i in ids) / 5, sum(lm[i].y for i in ids) / 5)


# ================= ระบบทำให้การจับมือนิ่ง =================
class OneEuro:
    """ตัวกรอง One Euro: ขยับช้า = กรองแรง (นิ่ง ไม่สั่น), ขยับเร็ว = กรองน้อย (ตามทัน ไม่หน่วง)
    min_cutoff ยิ่งน้อยยิ่งนิ่ง / beta ยิ่งมากยิ่งตามไวตอนขยับเร็ว"""

    def __init__(self, min_cutoff=1.2, beta=8.0, d_cutoff=1.0):
        self.min_cutoff, self.beta, self.d_cutoff = min_cutoff, beta, d_cutoff
        self.x = self.dx = self.t = None

    @staticmethod
    def _a(cutoff, dt):
        r = 2 * math.pi * cutoff * dt
        return r / (r + 1)

    def __call__(self, x, t):
        x = np.asarray(x, dtype=float)
        if self.x is None or t - self.t > 0.5:       # เริ่มใหม่ (หรือหายไปนาน) -> ไม่กรอง
            self.x, self.dx, self.t = x, np.zeros_like(x), t
            return x
        dt = max(t - self.t, 1e-3)
        self.t = t
        dx = (x - self.x) / dt
        a_d = self._a(self.d_cutoff, dt)
        self.dx = self.dx + a_d * (dx - self.dx)
        cutoff = self.min_cutoff + self.beta * np.linalg.norm(self.dx)
        a = self._a(cutoff, dt)
        self.x = self.x + a * (x - self.x)
        return self.x


def finger_states_h(lm, prev=None):
    """เหมือน finger_states แต่มี 2 เกณฑ์ (hysteresis):
    จะเปลี่ยนเป็น "ชู" ต้องตรงชัด (cos > 0.7), จะเปลี่ยนเป็น "พับ" ต้องงอชัด (cos < 0.35)
    ช่วงกลางๆ = ใช้สถานะเดิม -> นิ้วที่งอครึ่งๆ ไม่สลับไปมา"""
    out = []
    for i, (mcp, pip, tip) in enumerate([(5, 6, 8), (9, 10, 12), (13, 14, 16), (17, 18, 20)]):
        a, b = _v(lm, mcp, pip), _v(lm, pip, tip)
        cos = a @ b / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-6)
        farther = np.linalg.norm(_v(lm, 0, tip)) > np.linalg.norm(_v(lm, 0, pip))
        was = prev[i] if prev else None
        if not farther or cos < 0.35:
            up = False
        elif cos > 0.7:
            up = True
        else:
            up = was if was is not None else cos > 0.6
        out.append(bool(up))
    return out


def gesture_from(up):
    """[ชี้, กลาง, นาง, ก้อย] -> ชื่อท่า"""
    count = sum(up)
    if count >= 3:
        return "explode"
    if up == [True, False, False, False]:
        return "draw"
    if up == [True, True, False, False]:
        return "heart"
    if up == [False, True, False, False]:
        return "middle"
    if up == [False, False, False, True]:
        return "pinky"
    if count == 0:
        return "gather"
    return None


# ---------------- โมเดลจำท่ามือ (ถ้ามีไฟล์ gesture_model.npz) ----------------
try:
    from ml.gesture_model import hand_features, load_default
    MODEL = load_default()
except Exception as ex:                              # ไม่มีไฟล์/โหลดไม่ได้ -> ใช้กฎเดิม
    print("ไม่ได้ใช้โมเดล:", ex)
    MODEL = None


MODEL_TO_GESTURE = {"one": "draw", "peace": "heart", "fist": "gather", "palm": "explode",
                    "middle": "middle", "call": "pinky", "none": None}


def classify(tr, lm):
    """ตัดสินท่ามือ: ใช้โมเดล (ถ้ามีและเปิดอยู่) ไม่งั้นใช้กฎนับนิ้ว"""
    tr.fingers = finger_states_h(lm, tr.fingers)       # กฎ (ใช้เป็นตัวสำรองเสมอ)
    if MODEL is None or not shared_data.get("use_model", True):
        return gesture_from(tr.fingers)
    p = MODEL.proba(hand_features([(q.x, q.y) for q in lm]))
    tr.prob = p if tr.prob is None else tr.prob * 0.5 + p * 0.5   # เฉลี่ยข้ามเฟรม กันกระพริบ
    i = int(np.argmax(tr.prob))
    if tr.prob[i] >= MODEL_CONF:
        tr.last_g = MODEL_TO_GESTURE[MODEL.classes[i]]
    return tr.last_g


class Track:
    """ข้อมูลของมือ 1 ข้าง ที่ติดตามข้ามเฟรม"""

    def __init__(self, role):
        self.role = role                 # "ctrl" = มือคุมอนุภาค, "pen" = มือเขียน
        self.palm = None                 # ตำแหน่งกลางฝ่ามือล่าสุด (ยังไม่กรอง ใช้จับคู่มือ)
        self.seen = -1e9                 # เวลาที่เห็นล่าสุด
        self.lm = None                   # จุดมือล่าสุด
        self.fingers = None              # สถานะนิ้วล่าสุด (ใช้กับ hysteresis)
        self.f_tip = OneEuro()           # ปลายนิ้วชี้
        self.f_palm = OneEuro()          # กลางฝ่ามือ
        self.f_size = OneEuro(0.8, 2.0)  # ขนาดมือ (ซูม)
        self.f_roll = OneEuro(0.8, 1.0)  # มุมเอียง
        self.f_pinky = OneEuro()         # ปลายก้อย
        self.cur = self.new = None       # กันกระพริบ: ท่าที่ใช้อยู่ / ท่าใหม่ที่กำลังนับ
        self.cnt = 0
        self.prob = None                 # ความน่าจะเป็นของโมเดล (เฉลี่ยข้ามเฟรม)
        self.votes = []                  # ป้ายซ้าย/ขวา ย้อนหลัง (บอกว่ามือนี้ควรเป็นบทบาทไหน)
        self.pinch = {}                  # สถานะการแตะนิ้ว (เปลี่ยนฟิลเตอร์ / เปลี่ยนสีปากกา)
        self.f_corner = {}               # ตัวกรองมุมกรอบพอร์ทัล (ปลายโป้ง / ปลายชี้)
        self.corner_last = {}            # มุมกรอบล่าสุด (ใช้ตอนล็อกกรอบ)
        self.corner_hist = []            # มุมกรอบย้อนหลัง + ระยะโป้ง-ก้อย (ใช้ย้อนกลับตอนเริ่มล็อก)
        self.locked = False              # โป้งกำลังเข้าหาก้อย -> ล็อกมุมกรอบของมือนี้
        self.lock_far = 0.0              # ระยะโป้ง-ก้อยปกติตอนเริ่มล็อก (ใช้ตัดสินตอนปลดล็อก)
        self.lock_t = 0.0                # เวลาที่เริ่มล็อก
        self.last_g = None               # ท่าล่าสุดที่มั่นใจ

    def alive(self, now):
        return now - self.seen <= GRACE

    def gesture(self, g, need=3):
        """ท่าใหม่ต้องเห็นติดกัน need เฟรมถึงจะเปลี่ยน"""
        if g == self.cur:
            self.cnt = 0
        else:
            self.cnt = self.cnt + 1 if g == self.new else 1
            self.new = g
            if self.cnt >= need:
                self.cur, self.cnt = g, 0
        return self.cur


class HandSystem:
    """จับคู่มือที่เจอแต่ละเฟรม -> บทบาท (คุมอนุภาค / เขียน) แบบไม่สลับไปมา"""

    def __init__(self):
        self.tracks = {"ctrl": Track("ctrl"), "pen": Track("pen")}

    def role_from_label(self, label, swap):
        # ไม่ swap: มือขวาจริง = คุมอนุภาค, มือซ้าย = เขียน / swap: กลับกัน
        is_right = label == "Right"
        return ("pen" if is_right else "ctrl") if swap else ("ctrl" if is_right else "pen")

    def assign(self, found, now, swap):
        """found = [(landmarks, label), ...] -> {"ctrl": lm หรือ None, "pen": lm หรือ None}
        1) มือที่อยู่ใกล้ตำแหน่งเดิมของบทบาทไหน = บทบาทนั้น (ไม่สนป้ายซ้าย/ขวาที่อาจกระโดด)
        2) มือใหม่ที่ไม่ใกล้ใคร ค่อยใช้ป้ายซ้าย/ขวาจาก MediaPipe"""
        out = {"ctrl": None, "pen": None}
        items = [(lm, label, palm_center(lm)) for lm, label in found]
        free = list(range(len(items)))
        pairs = []
        for role, tr in self.tracks.items():
            if tr.alive(now) and tr.palm is not None:
                for i in free:
                    d = math.hypot(items[i][2][0] - tr.palm[0], items[i][2][1] - tr.palm[1])
                    pairs.append((d, role, i))
        for d, role, i in sorted(pairs):              # จับคู่ระยะใกล้สุดก่อน
            if d < MATCH_DIST and out[role] is None and i in free:
                out[role] = items[i][0]
                free.remove(i)
        for i in list(free):                          # มือใหม่ -> ใช้ป้าย
            role = self.role_from_label(items[i][1], swap)
            if out[role] is not None:                 # ป้ายชนกัน -> ให้บทบาทที่ยังว่าง
                role = "pen" if role == "ctrl" else "ctrl"
            if out[role] is None:
                out[role] = items[i][0]
                free.remove(i)
        label_of = {id(lm): label for lm, label in found}
        for role, lm in out.items():
            if lm is not None:
                tr = self.tracks[role]
                if not tr.alive(now):
                    tr.votes = []                     # มือใหม่ -> เริ่มนับโหวตใหม่
                tr.palm, tr.seen, tr.lm = palm_center(lm), now, lm
                tr.votes = (tr.votes + [self.role_from_label(label_of[id(lm)], swap)])[-20:]
        # ---- แก้บทบาทผิด: ถ้าป้ายส่วนใหญ่ (หลายเฟรม) บอกว่าเป็นอีกข้าง -> ย้ายบทบาท ----
        c, p = self.tracks["ctrl"], self.tracks["pen"]
        def wrong(tr):
            return len(tr.votes) >= 6 and sum(v != tr.role for v in tr.votes) / len(tr.votes) > 0.7
        if (wrong(c) and (wrong(p) or not p.alive(now))) or (wrong(p) and not c.alive(now)):
            self.swap_roles()
            for tr in self.tracks.values():
                tr.votes = []
            out = {"ctrl": out["pen"], "pen": out["ctrl"]}
        return out

    def swap_roles(self):
        t = self.tracks
        t["ctrl"], t["pen"] = t["pen"], t["ctrl"]
        t["ctrl"].role, t["pen"].role = "ctrl", "pen"


def pinch_dist(lm, a, b):
    """ระยะปลายนิ้ว a-b เทียบกับขนาดฝ่ามือ (ข้อมือ->โคนนิ้วกลาง) -> ไม่ขึ้นกับว่ามือใกล้/ไกลกล้อง"""
    return np.linalg.norm(_v(lm, a, b)) / (np.linalg.norm(_v(lm, 0, 9)) + 1e-6)


def pinch_edge(tr, lm, a, b, key):
    """คืน True ครั้งเดียวตอน "เพิ่งแตะ" (แตะค้าง 3 เฟรม) -> ต้องปล่อยก่อนถึงจะนับครั้งใหม่"""
    st = tr.pinch.setdefault(key, [0, False])          # [นับเฟรมที่แตะ, นับไปแล้วหรือยัง]
    if pinch_dist(lm, a, b) < PINCH_RATIO:
        st[0] += 1
        if st[0] >= 3 and not st[1]:
            st[1] = True
            return True
    elif pinch_dist(lm, a, b) > PINCH_RATIO * 1.4:    # ต้องแยกออกชัดๆ ถึงนับว่าปล่อย (กันสั่น)
        st[0], st[1] = 0, False
    return False
