# MatchCraft: Probabilistic Restaurant Record Linkage

Fellegi–Sunter probabilistic record linkage for identifying matching restaurant records across the Fodors and Zagat datasets.

The project learns how informative each field is from labelled record pairs, scores candidate pairs using agreement weights, and separates them into **Match**, **Possible Match**, and **Non-Match** regions.

## Highlights

- Cleans inconsistent names, addresses, cities, types, and phone numbers.
- Compares text fields with Jaro–Winkler similarity.
- Compares phone numbers using normalized digit strings.
- Learns match (`m`) and non-match (`u`) probabilities from training data.
- Uses add-0.5 smoothing to avoid zero or one probabilities.
- Selects decision thresholds using a validation set.
- Reports agreement patterns, worked examples, classification regions, and precision/recall/F1.
- Includes assert-based self-checks for important behavior.

## Dataset

The data comes from the [Magellan Data Repository](https://sites.google.com/site/anhaidgroup/useful-stuff/the-magellan-data-repository).

| Source | Rows |
|---|---:|
| Fodors (`tableA.csv`) | 533 |
| Zagat (`tableB.csv`) | 331 |

The labelled pairs are split as follows:

| Split | Pairs | Matches | Non-matches |
|---|---:|---:|---:|
| Train | 567 | 66 | 501 |
| Validation | 190 | 22 | 168 |
| Test | 189 | 22 | 167 |

The `class` column is the entity identifier and is intentionally excluded from matching features to prevent label leakage. It is used only for the final full-dataset reality check.

## Method

Each pair is represented by five binary agreement features:

- `name`
- `addr`
- `city`
- `phone`
- `type`

For each field, the implementation estimates:

- `mᵢ = P(agree | Match)`
- `uᵢ = P(agree | Non-match)`

The Fellegi–Sunter weight for a pair is:

```text
W = Σ log₂(mᵢ / uᵢ)                 when a field agrees
    log₂((1 - mᵢ) / (1 - uᵢ))       when a field disagrees
```

The validation set determines the two thresholds:

- Above `T_U`: **Match**
- Below `T_L`: **Non-Match**
- Between them: **Possible Match** for manual review

## Results

The learned model identifies phone and name as the strongest evidence:

| Field | Agree weight | Disagree weight |
|---|---:|---:|
| Name | +4.990 | -3.564 |
| Address | +0.430 | -1.765 |
| City | +0.188 | -1.242 |
| Phone | **+7.087** | **-4.249** |
| Type | +3.068 | -1.084 |

Test-set decisions:

| Region | Pairs | True matches | Non-matches |
|---|---:|---:|---:|
| Match | 20 | 20 | 0 |
| Possible match | 2 | 1 | 1 |
| Non-match | 167 | 1 | 166 |

| Possible matches treated as | Precision | Recall | F1 |
|---|---:|---:|---:|
| Matches | 0.955 | 0.955 | **0.955** |
| Non-matches | **1.000** | 0.909 | 0.952 |

See [RESULTS.md](RESULTS.md) for the full analysis and [results_output.txt](results_output.txt) for the complete program output.

## Run

Requirements:

- Python 3
- pandas

From the project directory, run:

```bash
python fellegi_sunter.py
```

The script prints the learned probabilities and weights, validation thresholds, a worked example, all training agreement patterns, test metrics, a full-table check, and self-check results.

## Repository structure

```text
.
├── data/
│   ├── tableA.csv
│   ├── tableB.csv
│   ├── train.csv
│   ├── valid.csv
│   └── test.csv
├── fellegi_sunter.py
├── solve.py
├── PROMPT.md
├── RESULTS.md
└── results_output.txt
```

`fellegi_sunter.py` contains the dataset-specific record-linkage implementation. `solve.py` contains additional lecture-style examples covering probabilistic linkage, decision trees, and AHC similarity.

## License and data note

This repository is intended for educational and experimental use. Please follow the terms and attribution requirements of the original Magellan Data Repository when redistributing the dataset.
