"""
เทรนโมเดลจำท่ามือ + เทียบกับกฎเดิม (วัดบนชุดทดสอบที่เป็นมือคุณเอง)
ใช้:  python ml/train.py [โฟลเดอร์ annotations ของ HaGRID (ไม่ต้องใส่ ถ้ามี ml/data/hagrid_sample.npz แล้ว)]
ผลลัพธ์: gesture_model.npz (โมเดลที่ดีที่สุด) + report.txt
"""
import csv
import glob
import json
import math
import os
import sys
import time

import numpy as np
from sklearn.metrics import confusion_matrix
from sklearn.neural_network import MLPClassifier

from gesture_model import CLASSES, hand_features

HERE = os.path.dirname(os.path.abspath(__file__))
rng = np.random.default_rng(0)

# HaGRID -> ท่าของเรา (ท่าที่ไม่อยู่ในนี้ = none)
HAGRID_MAP = {
    "one": "one",
    "peace": "peace", "peace_inverted": "peace", "two_up": "peace", "two_up_inverted": "peace",
    "fist": "fist",
    "palm": "palm", "stop": "palm", "stop_inverted": "palm", "four": "palm",
    "three": "palm", "three2": "palm", "three3": "palm",       # กฎเดิมนับ >=3 นิ้ว = แบมือ
    "middle_finger": "middle",
    "call": "call", "little_finger": "call",          # ชูก้อย (มี/ไม่มีนิ้วโป้ง)
    "point": "one",
}
# ท่ากำกวม (ชี้นิ้ว + นิ้วโป้งกาง / แตะปาก) -> ไม่ใช้เลย กันโมเดลสับสนกับท่า one
EXCLUDE = {"mute", "thumb_index", "thumb_index2", "three_gun"}
PER_CLASS = 6000          # ใช้ HaGRID ท่าละไม่เกินเท่านี้ (พอแล้ว + เทรนเร็ว)


def log(*a):
    msg = " ".join(str(x) for x in a)
    print(msg, flush=True)
    REPORT.append(msg)


REPORT = []


# ---------------- โหลดข้อมูล ----------------
def load_hagrid(root):
    """อ่าน json ของ HaGRID: แต่ละภาพมีหลายมือ (labels/hand_landmarks เรียงตรงกัน)
    ข้อมูลมี ~1.3 ล้านมือ -> สุ่มเก็บท่าละไม่เกิน PER_CLASS แบบ reservoir (ไม่กินแรม)"""
    keep = {c: [] for c in CLASSES}
    seen = {c: 0 for c in CLASSES}
    files = sorted(glob.glob(os.path.join(root, "**", "*.json"), recursive=True))
    log(f"HaGRID: พบไฟล์ annotation {len(files)} ไฟล์")
    for k, fp in enumerate(files):
        with open(fp) as f:
            data = json.load(f)
        for ann in data.values():
            for lab, lm in zip(ann.get("labels", []), ann.get("hand_landmarks", [])):
                if lab in EXCLUDE or not lm or len(lm) != 21:
                    continue
                c = HAGRID_MAP.get(lab, "none")
                seen[c] += 1
                if len(keep[c]) < PER_CLASS:
                    keep[c].append(np.asarray(lm, dtype=float)[:, :2])
                else:
                    j = rng.integers(0, seen[c])
                    if j < PER_CLASS:
                        keep[c][j] = np.asarray(lm, dtype=float)[:, :2]
        del data
        if (k + 1) % 20 == 0:
            print(f"  อ่านแล้ว {k + 1}/{len(files)} ไฟล์", flush=True)
    X, y = [], []
    for c in CLASSES:
        X += keep[c]; y += [c] * len(keep[c])
        log(f"  {c:7s} มี {seen[c]:8d} มือ -> ใช้ {len(keep[c])}")
    return X, y


def load_own(name):
    fp = os.path.join(HERE, "data", f"own_{name}.csv")
    X, y, X3 = [], [], []
    if not os.path.exists(fp):
        return X, y, X3
    with open(fp) as f:
        for row in csv.reader(f):
            if not row or row[0] not in CLASSES:
                continue
            v = np.array(row[3:], dtype=float).reshape(21, 3)
            X.append(v[:, :2]); X3.append(v); y.append(row[0])
    return X, y, X3


def featurize(xys, aspect=None, augment=False):
    """aspect=None -> HaGRID (ไม่รู้สัดส่วนภาพจริง -> สุ่มสัดส่วนเพื่อให้โมเดลไม่ยึดติด)"""
    out = []
    for xy in xys:
        a = aspect if aspect is not None else math.exp(rng.uniform(math.log(0.5), math.log(2.0)))
        p = xy.copy()
        if augment:
            p = p + rng.normal(0, 0.004, p.shape)       # จำลองมือสั่นนิดๆ
        out.append(hand_features(p, a))
    return np.array(out)


# ---------------- กฎเดิม (ก๊อปจาก index.py) ----------------
def _v(lm, a, b):
    return np.array([(lm[b][0] - lm[a][0]) * 640, (lm[b][1] - lm[a][1]) * 480, (lm[b][2] - lm[a][2]) * 640])


def rule_predict(lm):
    up = []
    for mcp, pip, tip in [(5, 6, 8), (9, 10, 12), (13, 14, 16), (17, 18, 20)]:
        a, b = _v(lm, mcp, pip), _v(lm, pip, tip)
        cos = a @ b / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-6)
        farther = np.linalg.norm(_v(lm, 0, tip)) > np.linalg.norm(_v(lm, 0, pip))
        up.append(bool(farther and cos > 0.6))
    n = sum(up)
    if n >= 3:
        return "palm"
    return {(True, False, False, False): "one", (True, True, False, False): "peace",
            (False, True, False, False): "middle", (False, False, False, True): "call"}.get(
        tuple(up), "fist" if n == 0 else "none")


# ---------------- เทรน / วัดผล ----------------
class Named:
    """ห่อ MLP ให้เทรน/ทายด้วยเลข แต่คืนค่าเป็นชื่อท่า (sklearn บางเวอร์ชันมีบั๊กกับชื่อที่เป็นตัวหนังสือ)"""

    def __init__(self, clf):
        self.clf = clf
        self.coefs_, self.intercepts_ = clf.coefs_, clf.intercepts_
        self.classes_ = [CLASSES[i] for i in clf.classes_]

    def predict(self, X):
        return np.array([CLASSES[i] for i in self.clf.predict(X)])


def train_mlp(X, y):
    mean, std = X.mean(0), X.std(0) + 1e-6
    yi = np.array([CLASSES.index(c) for c in y])
    clf = MLPClassifier(hidden_layer_sizes=(64, 32), alpha=1e-3, max_iter=300,
                        early_stopping=True, random_state=0)
    clf.fit((X - mean) / std, yi)
    return Named(clf), mean, std


def acc_report(name, y_true, y_pred):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    acc = (y_true == y_pred).mean() * 100
    per = "  ".join(f"{c}={((y_pred[y_true == c] == c).mean() * 100):.0f}%"
                    for c in CLASSES if (y_true == c).any())
    log(f"{name:28s} ความแม่นยำรวม {acc:5.1f}%   |  {per}")
    return acc


def save(clf, mean, std, path):
    d = {"mean": mean, "std": std, "n_layers": len(clf.coefs_), "classes": np.array(clf.classes_)}
    for i, (W, b) in enumerate(zip(clf.coefs_, clf.intercepts_)):
        d[f"W{i}"], d[f"b{i}"] = W, b
    np.savez(path, **d)


def main():
    t0 = time.time()
    log("=== เทรนโมเดลจำท่ามือ ===", time.strftime("%Y-%m-%d %H:%M"))
    own_tr, y_tr, _ = load_own("train")
    own_te, y_te, own_te3 = load_own("test")
    log(f"มือคุณ: ชุดเทรน {len(own_tr)} ตัวอย่าง, ชุดทดสอบ {len(own_te)} ตัวอย่าง")
    hag_X, hag_y = ([], [])
    cache = os.path.join(HERE, "data", "hagrid_sample.npz")   # อ่าน HaGRID ครั้งเดียว แล้วเก็บไว้
    if os.path.exists(cache):
        d = np.load(cache)
        hag_X, hag_y = list(d["X"]), [str(v) for v in d["y"]]
        log(f"HaGRID (จากแคช): {len(hag_X)} มือ")
    elif len(sys.argv) > 1 and os.path.isdir(sys.argv[1]):
        hag_X, hag_y = load_hagrid(sys.argv[1])
        os.makedirs(os.path.dirname(cache), exist_ok=True)
        np.savez_compressed(cache, X=np.array(hag_X), y=np.array(hag_y))

    if not own_te:
        log("ยังไม่มีชุดทดสอบ (data/own_test.csv) -> เตรียมข้อมูล HaGRID ไว้แล้ว รออัดมือก่อน")
        return
    X_te = featurize(own_te, 640 / 480)
    results = {}
    log("\n--- ผลบนชุดทดสอบ (มือคุณ อัดคนละรอบกับชุดเทรน) ---")
    results["rules"] = (acc_report("กฎเดิม (นับนิ้ว)", y_te, [rule_predict(v) for v in own_te3]), None)

    candidates = {}
    if hag_X:
        candidates["HaGRID อย่างเดียว"] = (featurize(hag_X), np.array(hag_y))
    if own_tr:
        Xo = np.vstack([featurize(own_tr, 640 / 480), featurize(own_tr, 640 / 480, augment=True)])
        candidates["มือคุณอย่างเดียว"] = (Xo, np.array(y_tr + y_tr))
    if hag_X and own_tr:
        Xo3 = np.vstack([featurize(own_tr, 640 / 480, augment=True) for _ in range(3)])
        candidates["ผสม (HaGRID + มือคุณ)"] = (
            np.vstack([candidates["HaGRID อย่างเดียว"][0], candidates["มือคุณอย่างเดียว"][0], Xo3]),
            np.concatenate([candidates["HaGRID อย่างเดียว"][1], np.array(y_tr * 5)]))

    for name, (X, y) in candidates.items():
        clf, mean, std = train_mlp(X, y)
        pred = clf.predict((X_te - mean) / std)
        results[name] = (acc_report(f"โมเดล: {name}", y_te, pred), (clf, mean, std, pred))

    best = max((k for k in results if results[k][1]), key=lambda k: results[k][0], default=None)
    if best:
        clf, mean, std, pred = results[best][1]
        save(clf, mean, std, os.path.join(HERE, "gesture_model.npz"))
        log(f"\nดีที่สุด: {best}  ({results[best][0]:.1f}% vs กฎเดิม {results['rules'][0]:.1f}%)")
        log("บันทึกโมเดล -> gesture_model.npz")
        cm = confusion_matrix(y_te, pred, labels=CLASSES)
        log("\nตารางความสับสน (แถว = ท่าจริง, คอลัมน์ = ที่โมเดลทาย):")
        log("        " + " ".join(f"{c[:6]:>6s}" for c in CLASSES))
        for c, row in zip(CLASSES, cm):
            log(f"{c:7s} " + " ".join(f"{v:6d}" for v in row))
    log(f"\nใช้เวลา {time.time() - t0:.0f} วินาที")
    with open(os.path.join(HERE, "report.txt"), "w") as f:
        f.write("\n".join(REPORT) + "\n")


if __name__ == "__main__":
    main()
