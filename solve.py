"""Week-9 Entity Matching, slides 42-66: worked solutions (run: python solve.py)."""
from itertools import product
from math import log2

# ---------- Part 1: Fellegi-Sunter (slides 43-50) ----------
m = {"Name": 0.90, "City": 0.80, "DOB": 0.95}
u = {"Name": 0.10, "City": 0.20, "DOB": 0.05}
TL, TU = -2, 3


def w(attr, agree):
    return log2(m[attr] / u[attr]) if agree else log2((1 - m[attr]) / (1 - u[attr]))


def decide(W):
    return "MATCH" if W > TU else "NON-MATCH" if W < TL else "POSSIBLE MATCH"


print("Per-attribute weights:")
for a in m:
    print(f"  {a:5} agree {w(a, True):+.3f}   disagree {w(a, False):+.3f}")

print("\nAll 8 agreement patterns (1=agree):")
for g in product([1, 0], repeat=3):
    W = sum(w(a, x) for a, x in zip(m, g))
    print(f"  Name={g[0]} City={g[1]} DOB={g[2]}  W={W:+.3f}  {decide(W)}")

A = w("Name", 1) + w("City", 1) + w("DOB", 0)
B = w("Name", 1) + w("City", 0) + w("DOB", 1)
assert abs(w("DOB", 0) - log2(0.05 / 0.95)) < 1e-12 and decide(A) == "POSSIBLE MATCH"
assert abs(w("DOB", 1) - log2(19)) < 1e-12 and decide(B) == "MATCH"
print(f"\nCase A W={A:+.3f} -> {decide(A)};  Case B W={B:+.3f} -> {decide(B)}")


# ---------- Part 2: Learning-based matching (slides 53-56) ----------
def lev(a, b):
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1]


def edit_sim(a, b):
    return 1 - lev(a, b) / max(len(a), len(b))


def jaro_winkler(s, t, p=0.1):
    if s == t:
        return 1.0
    r = max(len(s), len(t)) // 2 - 1
    sm, tm = [False] * len(s), [False] * len(t)
    mcount = 0
    for i, c in enumerate(s):
        for j in range(max(0, i - r), min(len(t), i + r + 1)):
            if not tm[j] and t[j] == c:
                sm[i] = tm[j] = True
                mcount += 1
                break
    if not mcount:
        return 0.0
    s1 = [c for c, f in zip(s, sm) if f]
    t1 = [c for c, f in zip(t, tm) if f]
    trans = sum(x != y for x, y in zip(s1, t1)) / 2
    jaro = (mcount / len(s) + mcount / len(t) + (mcount - trans) / mcount) / 3
    prefix = 0
    for x, y in zip(s[:4], t[:4]):
        if x != y:
            break
        prefix += 1
    return jaro + prefix * p * (1 - jaro)


# area codes serving each city (real US assignments)
AREA = {"Seattle": "206", "Redmond": "425", "Renton": "425", "Milwaukee": "414"}

pairs = [
    (("Mike Williams", "(425) 247 4893", "Seattle", "WA"), ("M. Williams", "247 4893", "Redmond", "WA"), "yes"),
    (("Richard Pike", "(414) 256 1257", "Milwaukee", "WI"), ("R. Pike", "256 1237", "Milwaukee", "WI"), "yes"),
    (("Jane McCain", "(206) 111 4215", "Renton", "WA"), ("J. M. McCain", "112 5200", "Renton", "WA"), "no"),
]


def features(a, b):
    return [
        jaro_winkler(a[0], b[0]),                    # s1 names, Jaro-Winkler
        edit_sim(a[0], b[0]),                        # s2 names, edit distance
        edit_sim(a[1].split(") ")[-1], b[1]),        # s3 phones, ignore a's area code
        float(a[2] == b[2]),                         # s4 city exact
        float(a[3] == b[3]),                         # s5 state exact
        float(AREA.get(b[2]) == a[1][1:4]),          # s6 a's area code fits b's city
    ]


def tree(s):  # decision tree from slide 56
    if s[0] <= 0.8:
        return "no"
    if s[2] > 0.8:
        return "yes"
    if s[3] <= 0.8:
        return "no"
    return "yes" if s[4] > 0.8 else "no"


print("\nFeature vectors v_i = <s1..s6>, label, tree prediction:")
for k, (a, b, lab) in enumerate(pairs, 1):
    v = features(a, b)
    print(f"  v{k} = <{', '.join(f'{x:.3f}' for x in v)}>  label={lab}  tree={tree(v)}")


# ---------- Part 3: AHC canonical tuple (slides 59-61) ----------
def sim(sname, sphone, scity, sstate):
    return 0.3 * sname + 0.3 * sphone + 0.1 * scity + 0.3 * sstate


a, b, _ = pairs[0]
s = features(a, b)
print(f"\nAHC sim(a1,b1) = 0.3*{s[0]:.3f} + 0.3*{s[2]:.3f} + 0.1*{s[3]:.0f} + 0.3*{s[4]:.0f}"
      f" = {sim(s[0], s[2], s[3], s[4]):.3f}")
print("Canonical tuple of {a1,b1}: (Mike Williams, (425) 247 4893, Seattle|Redmond, WA)")
