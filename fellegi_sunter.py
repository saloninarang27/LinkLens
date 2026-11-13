"""Fellegi-Sunter record linkage on Magellan Fodors-Zagat restaurants (run: python fellegi_sunter.py)."""
import re
from collections import Counter
from math import log2

import pandas as pd

FIELDS = ["name", "addr", "city", "phone", "type"]   # 'class' is the entity id -> excluded (label leak)
JW_T = 0.85                                         # Jaro-Winkler agreement threshold for text fields


def clean(s):
    s = re.sub(r"[`'\\]", " ", str(s).lower())
    return re.sub(r"\s+", " ", re.sub(r"[^a-z0-9 ]", " ", s)).strip()


def jaro_winkler(s, t, p=0.1):
    if s == t:
        return 1.0
    if not s or not t:
        return 0.0
    r = max(max(len(s), len(t)) // 2 - 1, 0)
    sm, tm, m = [False] * len(s), [False] * len(t), 0
    for i, c in enumerate(s):
        for j in range(max(0, i - r), min(len(t), i + r + 1)):
            if not tm[j] and t[j] == c:
                sm[i] = tm[j] = True
                m += 1
                break
    if not m:
        return 0.0
    a = [c for c, f in zip(s, sm) if f]
    b = [c for c, f in zip(t, tm) if f]
    tr = sum(x != y for x, y in zip(a, b)) / 2
    jaro = (m / len(s) + m / len(t) + (m - tr) / m) / 3
    pre = next((k for k, (x, y) in enumerate(zip(s[:4], t[:4])) if x != y), min(4, len(s), len(t)))
    return jaro + pre * p * (1 - jaro)


def agree(field, x, y):
    if field == "phone":
        return re.sub(r"\D", "", str(x))[:10] == re.sub(r"\D", "", str(y))[:10]
    x, y = clean(x), clean(y)
    if field in ("city", "type") and x and y and (x in y or y in x):   # "new york" vs "new york city"
        return True
    return jaro_winkler(x, y) >= JW_T


A = pd.read_csv("data/tableA.csv", dtype=str).set_index("id")
B = pd.read_csv("data/tableB.csv", dtype=str).set_index("id")


def gammas(split):
    p = pd.read_csv(f"data/{split}.csv", dtype=str)
    g = [tuple(int(agree(f, A.at[l, f], B.at[r, f])) for f in FIELDS) for l, r in zip(p.ltable_id, p.rtable_id)]
    return p, g, p.label.astype(int).tolist()


train_p, train_g, train_y = gammas("train")
valid_p, valid_g, valid_y = gammas("valid")
test_p, test_g, test_y = gammas("test")

# ---- step 3: estimate m and u from training pairs (add-0.5 smoothing) ----
M = [g for g, y in zip(train_g, train_y) if y]
U = [g for g, y in zip(train_g, train_y) if not y]
m = [(sum(g[i] for g in M) + 0.5) / (len(M) + 1) for i in range(len(FIELDS))]
u = [(sum(g[i] for g in U) + 0.5) / (len(U) + 1) for i in range(len(FIELDS))]
w_agree = [log2(mi / ui) for mi, ui in zip(m, u)]
w_dis = [log2((1 - mi) / (1 - ui)) for mi, ui in zip(m, u)]


def W(g):
    return sum(w_agree[i] if a else w_dis[i] for i, a in enumerate(g))


print(f"Dataset: Fodors {len(A)} x Zagat {len(B)} rows; labelled pairs train {len(train_y)} "
      f"({sum(train_y)} M), valid {len(valid_y)} ({sum(valid_y)} M), test {len(test_y)} ({sum(test_y)} M)\n")
print(f"{'field':6} {'m':>6} {'u':>6} {'w_agree':>8} {'w_disagree':>10}")
for f, mi, ui, wa, wd in zip(FIELDS, m, u, w_agree, w_dis):
    print(f"{f:6} {mi:6.3f} {ui:6.3f} {wa:+8.3f} {wd:+10.3f}")

# ---- step 5: thresholds from validation set ----
vW = [W(g) for g in valid_g]
T_U = max(w for w, y in zip(vW, valid_y) if not y)     # above every non-match -> MATCH
T_L = min(w for w, y in zip(vW, valid_y) if y)         # below every match     -> NON-MATCH
if T_L > T_U:                                          # no overlap: one cut point, no review zone
    T_L = T_U = (T_L + T_U) / 2
print(f"\nThresholds (from validation): T_L = {T_L:+.3f}, T_U = {T_U:+.3f}")


def decide(w):
    return "MATCH" if w > T_U else "NON-MATCH" if w < T_L else "POSSIBLE"


# ---- step 7: worked example (first test match) ----
k = test_y.index(1)
l, r = test_p.ltable_id[k], test_p.rtable_id[k]
print(f"\nWorked example: Fodors {l} vs Zagat {r}")
for i, f in enumerate(FIELDS):
    a = test_g[k][i]
    print(f"  {f:6} {clean(A.at[l, f])[:28]:28} | {clean(B.at[r, f])[:28]:28} "
          f"{'Agree   ' if a else 'Disagree'} {w_agree[i] if a else w_dis[i]:+.3f}")
print(f"  W = {W(test_g[k]):+.3f} -> {decide(W(test_g[k]))}")

# ---- prompt 2: weight of every agreement pattern seen in training ----
print("\nAgreement patterns in training (1 = agree):")
print("  " + " ".join(f"{f[:5]:>5}" for f in FIELDS) + f" {'W':>8} {'#M':>4} {'#U':>4}  decision")
pat = Counter(zip(train_g, train_y))
for g in sorted({g for g in train_g}, key=W, reverse=True):
    print("  " + " ".join(f"{a:>5}" for a in g) + f" {W(g):+8.3f} {pat[(g, 1)]:>4} {pat[(g, 0)]:>4}  {decide(W(g))}")

# ---- step 6: evaluate on test ----
tW = [W(g) for g in test_g]
dec = [decide(w) for w in tW]
print("\nTest set decisions:")
for d in ("MATCH", "POSSIBLE", "NON-MATCH"):
    ys = [y for x, y in zip(dec, test_y) if x == d]
    print(f"  {d:9} {len(ys):4} pairs  (true matches {sum(ys)}, non-matches {len(ys) - sum(ys)})")


def prf(pred):
    tp = sum(p and y for p, y in zip(pred, test_y))
    fp = sum(p and not y for p, y in zip(pred, test_y))
    fn = sum(not p and y for p, y in zip(pred, test_y))
    P = tp / (tp + fp) if tp + fp else 0
    R = tp / (tp + fn) if tp + fn else 0
    return P, R, 2 * P * R / (P + R) if P + R else 0, tp, fp, fn


for name, pred in [("POSSIBLE counted as MATCH", [d != "NON-MATCH" for d in dec]),
                   ("POSSIBLE counted as NON-MATCH", [d == "MATCH" for d in dec])]:
    P, R, F, tp, fp, fn = prf(pred)
    print(f"  {name:30} P={P:.3f} R={R:.3f} F1={F:.3f}  (TP {tp}, FP {fp}, FN {fn})")

# ---- reality check: all A x B pairs vs ground truth ('class' = entity id, used ONLY for scoring) ----
gt = {(a, b) for a in A.index for b in B.index if A.at[a, "class"] == B.at[b, "class"]}
labelled = pd.concat([train_p, valid_p, test_p])
assert all((y == "1") == ((l, r) in gt) for l, r, y in zip(labelled.ltable_id, labelled.rtable_id, labelled.label))
full = Counter()
for a in A.index:
    for b in B.index:
        full[(decide(W(tuple(int(agree(f, A.at[a, f], B.at[b, f])) for f in FIELDS))), (a, b) in gt)] += 1
print(f"\nFull run on all {len(A) * len(B)} A x B pairs, ground truth = {len(gt)} true matches:")
for d in ("MATCH", "POSSIBLE", "NON-MATCH"):
    print(f"  {d:9} true matches {full[(d, True)]:4}   non-matches {full[(d, False)]:6}")

# ---- self-checks ----
assert clean("` rainbow room '") == "rainbow room"
assert agree("phone", "212/632 -5000", "212-632-5000")
assert agree("city", "` new york '", "` new york city '")
assert abs(log2(0.9 / 0.1) - 3.1699) < 1e-3          # slide 46 value
assert all(wa > 0 > wd for wa, wd in zip(w_agree, w_dis))
print("\nself-checks passed")
