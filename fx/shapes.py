"""รูปทรงของอนุภาค (กาแล็กซี่ หัวใจ ตัวหนังสือ ฯลฯ) + เปลี่ยนหมึกเป็นอนุภาค"""

import math

import numpy as np
import pygame

from .config import H, TEXT_LINES, W


# ---------------- สร้างรูปทรงเป้าหมาย ----------------
def rot_x(points, a):
    """หมุนจุดรอบแกน X (ใช้เอียงรูปให้มองเห็นเป็น 3D)"""
    c, s = math.cos(a), math.sin(a)
    m = np.array([[1, 0, 0], [0, c, -s], [0, s, c]])
    return points @ m.T


def make_galaxy(n):
    """กาแล็กซี่ 3 แขนหมุนวน"""
    r = np.random.exponential(90, n).clip(5, 330)
    arm = np.random.randint(0, 3, n)
    ang = arm * 2 * np.pi / 3 + r * 0.018 + np.random.normal(0, 0.35, n)
    pts = np.stack([r * np.cos(ang), np.random.normal(0, 8, n), r * np.sin(ang)], 1)
    pts = rot_x(pts, 1.1)
    # สี: ตรงกลางขาวอมฟ้า ขอบเป็นฟ้าน้ำเงิน
    t = (r / 330)[:, None]
    col = (1 - t) * np.array([220, 230, 255]) + t * np.array([70, 110, 255])
    return pts, col


def make_planet(n):
    """ดาวเคราะห์ทรงกลม + วงแหวน"""
    k = int(n * 0.65)
    v = np.random.normal(size=(k, 3))
    v /= np.linalg.norm(v, axis=1, keepdims=True)       # จุดบนผิวทรงกลม
    v *= 110 * np.random.uniform(0.85, 1.0, (k, 1))
    m = n - k
    a = np.random.uniform(0, 2 * np.pi, m)
    r = np.random.uniform(160, 240, m)
    ring = np.stack([r * np.cos(a), np.random.normal(0, 2, m), r * np.sin(a)], 1)
    ring = rot_x(ring, 0.45)
    pts = np.vstack([v, ring])
    col = np.vstack([
        np.tile([255, 140, 40], (k, 1)) * np.random.uniform(0.6, 1.0, (k, 1)),
        np.tile([255, 210, 130], (m, 1)) * np.random.uniform(0.4, 0.9, (m, 1)),
    ])
    return pts, col


def make_heart(n):
    """หัวใจ 3D จากสมการหัวใจ 2D แล้วเพิ่มความหนา"""
    t = np.random.uniform(0, 2 * np.pi, n)
    s = np.sqrt(np.random.uniform(0, 1, n))             # กระจายเต็มพื้นที่ด้านใน
    x = 16 * np.sin(t) ** 3
    y = 13 * np.cos(t) - 5 * np.cos(2 * t) - 2 * np.cos(3 * t) - np.cos(4 * t)
    z = np.random.uniform(-1, 1, n) * 6 * np.sqrt(1 - s ** 2 + 0.05)
    pts = np.stack([x * s, y * s + 2, z], 1) * 12
    col = np.tile([255, 50, 80], (n, 1)) * np.random.uniform(0.6, 1.0, (n, 1))
    return pts, col


def ink_to_particles(ink, n):
    """เปลี่ยนหมึกบนกระดานเป็นอนุภาค: สุ่มพิกเซลที่มีหมึก -> ตำแหน่ง + สีเดิม"""
    arr = pygame.surfarray.array3d(ink)               # (W, H, 3)
    xs, ys = np.nonzero(arr.max(axis=2) > 60)
    if len(xs) == 0:                                  # ไม่ได้เขียนอะไร -> ใช้ลูกพลังงานแทน
        return make_gather(n)
    pick = np.random.randint(0, len(xs), n)
    x, y = xs[pick].astype(float), ys[pick].astype(float)
    pts = np.stack([x - W / 2, H / 2 - y, np.zeros(n)], 1)   # พิกัดจอ -> พิกัดรูป (กลางจอ = 0)
    col = arr[xs[pick], ys[pick]].astype(float)
    return pts, col


_DIRS = {}   # ทิศของอนุภาคแต่ละตัว ใช้ร่วมกันระหว่าง "แตก" กับ "รวม" -> อนุภาคพุ่งออก/หดกลับตามแนวเดิม


def _dirs(n):
    if n not in _DIRS:
        v = np.random.normal(size=(n, 3))
        _DIRS[n] = v / np.linalg.norm(v, axis=1, keepdims=True)
    return _DIRS[n]


def _rainbow(v):
    """สีรุ้งสด (HSV เต็ม 100%) ตามทิศของอนุภาค"""
    hue = (np.arctan2(v[:, 1], v[:, 0]) / (2 * np.pi) + 0.5) % 1.0
    return hsv_to_rgb(hue, 1.0, 1.0) * 255


def make_explode(n):
    """แตกกระจายไปทั่วจอ เป็นวงรุ้งเหมือนพลุ"""
    v = _dirs(n)
    pts = v * np.random.uniform(250, 500, (n, 1))
    return pts, _rainbow(v)


def make_gather(n):
    """รวมกันเป็นลูกพลังงานแน่นๆ ตรงกลาง: แกนกลางขาวร้อน ขอบเป็นสีรุ้ง"""
    v = _dirs(n)
    r = np.random.exponential(38, n).clip(0, 130)
    pts = v * r[:, None]
    t = (r / 130)[:, None]
    col = _rainbow(v) * t + 255 * (1 - t)      # ใกล้กลาง = ขาว, ขอบ = สีรุ้ง
    return pts, col


def _capsule(x, y, ax, ay, bx, by, r):
    """เช็กว่าจุด (x,y) อยู่ในแคปซูล (เส้นตรงหนา r ปลายมน) จาก A ไป B ไหม"""
    abx, aby = bx - ax, by - ay
    t = np.clip(((x - ax) * abx + (y - ay) * aby) / (abx * abx + aby * aby), 0, 1)
    return (x - ax - t * abx) ** 2 + (y - ay - t * aby) ** 2 <= r * r


def make_middle_finger(n):
    """มือกำ + ชูนิ้วกลาง: สร้างจากรูปทรงง่ายๆ (แคปซูล) มาต่อกัน"""
    def inside(x, y):
        return (
            _capsule(x, y, -40, -110, 40, -110, 45)      # กำปั้น ส่วนล่าง
            | _capsule(x, y, -40, -60, 40, -60, 45)      # กำปั้น ส่วนบน
            | _capsule(x, y, -14, -30, -14, 200, 18)     # นิ้วกลางชูขึ้น
            | _capsule(x, y, -50, -5, -50, 12, 16)       # ข้อนิ้วชี้ที่พับ
            | _capsule(x, y, 22, -5, 22, 12, 16)         # ข้อนิ้วนางที่พับ
            | _capsule(x, y, 55, -18, 55, -2, 14)        # ข้อนิ้วก้อยที่พับ
            | _capsule(x, y, -88, -45, -20, -78, 15)     # นิ้วโป้งพาดด้านหน้า
            | _capsule(x, y, -5, -185, 5, -185, 36)      # ข้อมือ
        )

    # สุ่มจุดในกรอบเยอะๆ แล้วเก็บเฉพาะจุดที่อยู่ในรูปมือ
    x = np.random.uniform(-110, 100, n * 8)
    y = np.random.uniform(-225, 225, n * 8)
    keep = inside(x, y)
    x, y = x[keep], y[keep]
    # จุดที่อยู่ใกล้ขอบ = ขยับไป 7 พิกเซลแล้วหลุดออกนอกรูป -> เก็บไว้ทั้งหมดให้ขอบชัด
    e = 7
    edge = ~(inside(x + e, y) & inside(x - e, y) & inside(x, y + e) & inside(x, y - e))
    inner = ~edge & (np.random.rand(len(x)) < 0.35)     # ข้างในเก็บแค่ 35% ให้โปร่งๆ
    x, y = x[edge | inner], y[edge | inner]
    idx = np.random.permutation(len(x))[:n]
    if len(idx) < n:                                     # เผื่อจุดไม่พอ
        idx = np.concatenate([idx, np.random.randint(0, len(x), n - len(idx))])
    x, y = x[idx], y[idx]

    z = np.random.uniform(-1, 1, n) * 18                 # ความหนาให้ดูเป็น 3D
    pts = np.stack([x, y - 10, z], 1) * 1.35
    # สีไฟ: ล่างแดง -> บนเหลือง
    t = ((y + 225) / 450)[:, None]
    col = (1 - t) * np.array([255, 30, 30]) + t * np.array([255, 220, 40])
    return pts, col


def make_text(n):
    """ตัวหนังสืออนุภาค: วาดข้อความลงภาพขาวดำ แล้วสุ่มจุดจากพิกเซลที่เป็นตัวอักษร"""
    font = pygame.font.SysFont("arialblack,impact,arial", 150, bold=True)
    lines = [font.render(t, True, (255, 255, 255)) for t in TEXT_LINES]
    w = max(l.get_width() for l in lines)
    h = sum(l.get_height() for l in lines)
    canvas = pygame.Surface((w, h))
    yy = 0
    for l in lines:                                   # จัดกึ่งกลางทีละบรรทัด
        canvas.blit(l, ((w - l.get_width()) // 2, yy))
        yy += l.get_height()
    mask = pygame.surfarray.array3d(canvas)[:, :, 0] > 128
    xs, ys = np.nonzero(mask)                         # พิกัดพิกเซลที่เป็นตัวอักษร
    pick = np.random.randint(0, len(xs), n)
    x = xs[pick] + np.random.uniform(-0.5, 0.5, n) - w / 2
    y = -(ys[pick] + np.random.uniform(-0.5, 0.5, n) - h / 2)   # กลับหัว (จอ y ชี้ลง)
    s = min(700 / w, 420 / h)                         # ย่อ/ขยายให้พอดีจอ
    z = np.random.uniform(-1, 1, n) * 12
    pts = np.stack([x * s, y * s, z], 1)
    # สีไฟ: ล่างแดง -> บนเหลือง
    t = ((y * s + 210) / 420).clip(0, 1)[:, None]
    col = (1 - t) * np.array([255, 30, 30]) + t * np.array([255, 220, 40])
    return pts, col


def make_dna(n):
    """DNA เกลียวคู่: 2 เกลียว (ชมพู/ฟ้า) พันกัน + ขั้นบันได (คู่เบส) เชื่อมระหว่างเกลียว"""
    R, Hh, turns = 80, 250, 1.4          # รัศมีเกลียว / ครึ่งความสูง / จำนวนรอบ
    pink, cyan = np.array([255, 50, 190]), np.array([40, 210, 255])

    # --- 2 เกลียว ---
    ns = int(n * 0.6) // 2
    t = np.random.uniform(-1, 1, ns)
    ph = t * turns * 2 * np.pi
    def strand(phase):
        rr = R + np.random.normal(0, 2.5, ns)          # ความหนาของเส้นเกลียว
        return np.stack([rr * np.cos(phase), t * Hh + np.random.normal(0, 3, ns),
                         rr * np.sin(phase)], 1)
    a, b = strand(ph), strand(ph + np.pi)               # เกลียวที่ 2 อยู่ตรงข้าม (ห่าง 180°)

    # --- ขั้นบันได ---
    nr = n - 2 * ns
    rungs = 16
    k = np.random.randint(0, rungs, nr)
    tk = -1 + (k + 0.5) * 2 / rungs
    pk = tk * turns * 2 * np.pi
    u = np.random.uniform(0, 1, nr)                    # ตำแหน่งตามแนวขั้น (0 = ฝั่งชมพู, 1 = ฝั่งฟ้า)
    pa = np.stack([R * np.cos(pk), tk * Hh, R * np.sin(pk)], 1)
    pb = np.stack([R * np.cos(pk + np.pi), tk * Hh, R * np.sin(pk + np.pi)], 1)
    rung = pa + (pb - pa) * u[:, None] + np.random.normal(0, 2, (nr, 3))
    rung_col = np.where((u < 0.5)[:, None], pink, cyan) * 0.6 + 255 * 0.4   # ผสมขาว ให้ขั้นเด่น

    pts = rot_x(np.vstack([a, b, rung]), 0.3)          # เอียงเข้าหาจอนิดๆ ให้เห็นความลึก
    col = np.vstack([np.tile(pink, (ns, 1)), np.tile(cyan, (ns, 1)), rung_col])
    return pts, col


# ---------------- หลุมดำ (ไม่ได้ใช้แล้ว เก็บไว้เผื่อเอากลับมา) ----------------
BH = {}            # เก็บสถานะของอนุภาคหลุมดำ (รัศมี มุม) เพื่ออัปเดตทุกเฟรม


BH_IN, BH_OUT = 62, 300   # ขอบในสุด (ตกลงหลุม) / ขอบนอกของจาน


def make_blackhole(n):
    """จาน (accretion disk) หมุนรอบหลุม + วงแสงโค้งรอบหลุม (เหมือนแสงถูกแรงโน้มถ่วงดัด)"""
    nd = int(n * 0.72)                       # 72% เป็นจาน ที่เหลือเป็นวงแสง
    nh = n - nd
    BH["nd"] = nd
    BH["r"] = BH_IN + np.random.exponential(55, nd).clip(0, BH_OUT - BH_IN)
    BH["th"] = np.random.uniform(0, 2 * np.pi, nd)
    BH["dy"] = np.random.normal(0, 3, nd)    # จานหนานิดๆ
    BH["hr"] = 82 + np.random.exponential(7, nh).clip(0, 40)
    BH["hth"] = np.random.uniform(0, 2 * np.pi, nh)
    BH["hz"] = np.random.normal(0, 2, nh)
    return blackhole_points()


def blackhole_update():
    """ทุกเฟรม: หมุนจาน (ใกล้หลุม = หมุนเร็วกว่า) + ดูดเข้าทีละนิด"""
    r = BH["r"]
    BH["th"] += 0.035 * (BH_IN / r) ** 1.5   # กฎของเคปเลอร์: วงใน หมุนเร็วกว่าวงนอก
    BH["r"] -= 0.25 + 10 / r                 # ค่อยๆ ไหลเข้า ใกล้หลุมยิ่งเร็ว
    fell = BH["r"] < BH_IN                   # ตกลงหลุมแล้ว -> เกิดใหม่ที่ขอบนอก
    BH["r"][fell] = np.random.uniform(BH_OUT * 0.6, BH_OUT, fell.sum())
    BH["hth"] += 0.004                       # วงแสงหมุนช้าๆ ให้ดูระยิบ


def blackhole_points():
    r, th = BH["r"], BH["th"]
    disk = np.stack([r * np.cos(th), BH["dy"], r * np.sin(th)], 1)
    disk = rot_x(disk, 0.22)                 # เอียงจานนิดเดียว มองเกือบด้านข้าง
    hr, hth = BH["hr"], BH["hth"]
    halo = np.stack([hr * np.cos(hth), hr * np.sin(hth), BH["hz"]], 1)
    pts = np.vstack([disk, halo])

    # สี: วงในขาวร้อน -> ส้มทอง -> แดงเข้มที่ขอบนอก
    t = ((r - BH_IN) / (BH_OUT - BH_IN)).clip(0, 1)[:, None]
    hot, mid, cold = np.array([255, 245, 220]), np.array([255, 150, 30]), np.array([170, 25, 5])
    disk_col = np.where(t < 0.3, hot + (mid - hot) * (t / 0.3),
                        mid + (cold - mid) * ((t - 0.3) / 0.7))
    th_ = ((hr - 82) / 40)[:, None]
    halo_col = np.array([255, 235, 190]) * (1 - th_) + np.array([255, 120, 20]) * th_
    return pts, np.vstack([disk_col, halo_col])


def hsv_to_rgb(h, s, v):
    """แปลง HSV -> RGB (ค่า 0-1) แบบทำทีละหลายพันจุดพร้อมกัน"""
    i = np.floor(h * 6).astype(int) % 6
    f = h * 6 - np.floor(h * 6)
    p, q, t = v * (1 - s), v * (1 - f * s), v * (1 - (1 - f) * s)
    v = np.full_like(h, v)
    p, q, t = np.broadcast_to(p, h.shape), q, t
    table = [(v, t, p), (q, v, p), (p, v, t), (p, q, v), (t, p, v), (v, p, q)]
    rgb = np.zeros((len(h), 3))
    for k, (r, g, b) in enumerate(table):
        m = i == k
        rgb[m] = np.stack([r[m], g[m], b[m]], 1)
    return rgb


SHAPES = {
    "galaxy": make_galaxy,
    "planet": make_planet,
    "heart": make_heart,
    "explode": make_explode,
    "gather": make_gather,
    "middle": make_text,   # ข้อความ (อยากได้รูปมือ/DNA/หลุมดำ เปลี่ยนเป็น make_middle_finger / make_dna / make_blackhole)
}
