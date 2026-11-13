# Fellegi-Sunter Record Linkage on the Magellan Fodors-Zagat Dataset

## 1. Dataset

Taken from the Magellan Data Repository (https://sites.google.com/site/anhaidgroup/useful-stuff/the-magellan-data-repository): **Fodors-Zagat restaurants**.

| Table | Rows | Columns |
|---|---|---|
| A: Fodors | 533 | id, name, addr, city, phone, type, class |
| B: Zagat | 331 | id, name, addr, city, phone, type, class |

| Labelled pairs | Pairs | Matches | Non-matches |
|---|---|---|---|
| train | 567 | 66 | 501 |
| valid | 190 | 22 | 168 |
| test | 189 | 22 | 167 |

The `class` column is **not used**. It is the true entity ID (matched records share it), so using it would leak the answer.

## 2. Method (same steps as lecture slides 43–48)

1. **Clean** the values: lowercase them, remove the stray quotes and backticks (`` ` rainbow room ' `` becomes `rainbow room`), and keep only the digits of phone numbers (`212/632 -5000` and `212-632-5000` both become `2126325000`).
2. **Compare** each record pair on 5 fields, giving agree (1) or disagree (0):
   - name, addr: Jaro–Winkler similarity ≥ 0.85
   - city, type: Jaro–Winkler ≥ 0.85, or one value contains the other ("new york" / "new york city")
   - phone: the digits are exactly equal
3. **Estimate** mᵢ = P(agree | Match) and uᵢ = P(agree | Unmatch) from the training pairs, with add-0.5 smoothing.
4. **Weights:** an agreeing field scores wᵢ = log₂(mᵢ/uᵢ), a disagreeing field scores wᵢ = log₂((1−mᵢ)/(1−uᵢ)), and the pair's total is **W = Σ wᵢ**.
5. **Thresholds** are set on the validation pairs. T_U = the highest W of any non-match, T_L = the lowest W of any match. Then W > T_U means Match, W < T_L means Non-Match, and anything in between is a Possible Match (manual review).
6. **Evaluate** on the test pairs.

## 3. Learned m, u and weights

| Field | m | u | Weight if agree | Weight if disagree |
|---|---|---|---|---|
| name | 0.918 | 0.029 | **+4.990** | −3.564 |
| addr | 0.903 | 0.670 | +0.430 | −1.765 |
| city | 0.918 | 0.806 | +0.188 | −1.242 |
| **phone** | 0.948 | 0.007 | **+7.087** | **−4.249** |
| type | 0.560 | 0.067 | +3.068 | −1.084 |

- **Phone is the strongest evidence.** Different restaurants almost never share a phone number (u = 0.007).
- **Name is second.** Agreeing on name is worth +4.99.
- **City and address are weak.** The labelled non-match pairs are deliberately hard cases from the same city, so city agrees for 81% of non-matches, and address agrees for 67% under the lenient Jaro–Winkler threshold. Agreeing on them proves little.

## 4. Thresholds

On the validation pairs, T_L = T_U = **+3.056**. Only the single pattern with W = 3.056 (name disagrees, address, city and phone agree, type disagrees) falls in the Possible Match region.

## 5. Worked example (test pair: Fodors 22 vs Zagat 240)

| Field | Fodors | Zagat | Comparison | Weight |
|---|---|---|---|---|
| name | yujean kang s gourmet chinese… | yujean kang s | Agree | +4.990 |
| addr | 67 n raymond ave | 67 n raymond ave | Agree | +0.430 |
| city | los angeles | pasadena | Disagree | −1.242 |
| phone | 818 585 0855 | 818 585 0855 | Agree | +7.087 |
| type | asian | chinese | Disagree | −1.084 |
| **W** | | | | **+10.180, so MATCH** |

Two fields disagree, but agreeing on phone and name is strong enough evidence to call it a match.

## 6. Which combination of agreeing fields gives the strongest evidence (training pairs)

| name | addr | city | phone | type | W | # Match | # Non-match | Decision |
|---|---|---|---|---|---|---|---|---|
| 1 | 1 | 1 | 1 | 1 | +15.763 | 30 | 0 | MATCH |
| 1 | 1 | 1 | 1 | 0 | +11.611 | 20 | 0 | MATCH |
| 0 | 1 | 1 | 1 | 1 | +7.208 | 2 | 0 | MATCH |
| 0 | 1 | 1 | 1 | 0 | +3.056 | 1 | 3 | POSSIBLE |
| 1 | 1 | 1 | 0 | 0 | +0.275 | 2 | 0 | NON-MATCH |
| 0 | 1 | 1 | 0 | 0 | −8.279 | 0 | 271 | NON-MATCH |
| 0 | 0 | 0 | 0 | 0 | −11.904 | 0 | 39 | NON-MATCH |

(Selected rows. All 22 patterns are in `results_output.txt`.)

- Every pattern where **name and phone both agree** is a match, with W of +9.4 or more.
- **Address, city and type agreeing without name and phone** (W = −4.127, 18 non-matches) is never a match. This is the slide-50 point: counting how many fields agree is misleading, and what matters is *which* fields agree.

## 7. Test-set results

| Region | Pairs | True matches | Non-matches |
|---|---|---|---|
| MATCH | 20 | 20 | 0 |
| POSSIBLE MATCH | 2 | 1 | 1 |
| NON-MATCH | 167 | 1 | 166 |

| Possible Match treated as | Precision | Recall | F1 |
|---|---|---|---|
| Match (after review) | 0.955 | 0.955 | **0.955** |
| Non-match | **1.000** | 0.909 | 0.952 |

- Every pair the model calls a Match really is one: precision is 1.0 for the Match region.
- Only 2 of 189 pairs (1%) need manual review.
- One true match was missed (W = +0.275). Name, address and city agree, but the phone number and type differ between the two sources.

## 8. Conclusion

Fellegi–Sunter works well on Fodors-Zagat (**F1 = 0.955**). It learns from the data that **phone (+7.09) and name (+4.99)** are the most informative fields, and that city and address carry little weight. Two pairs that agree on the same number of fields can still get very different decisions.

## Files

| File | Contents |
|---|---|
| `PROMPT.md` | The prompts used to generate the code |
| `fellegi_sunter.py` | The code (`python fellegi_sunter.py`) |
| `results_output.txt` | The full output of the run |
| `data/` | Fodors-Zagat tables and the labelled train/valid/test pairs |
