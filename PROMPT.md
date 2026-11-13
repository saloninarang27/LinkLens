# Prompts used

## Prompt 1: generate the code

> I am using the **Fodors-Zagat restaurants** dataset from the Magellan Data Repository
> (https://sites.google.com/site/anhaidgroup/useful-stuff/the-magellan-data-repository).
> In `data/` I have `tableA.csv` (Fodors, 533 rows) and `tableB.csv` (Zagat, 331 rows) with columns
> `id, name, addr, city, phone, type, class`, and labelled pairs `train.csv`, `valid.csv` and `test.csv`
> with columns `ltable_id, rtable_id, label` (1 = match, 0 = non-match).
>
> Write a Python script that applies **Fellegi-Sunter probabilistic record linkage**:
> 1. Clean the values. They contain stray quotes and backticks (`` ` rainbow room ' ``), and phones are
>    written differently in the two tables (`212/632 -5000` vs `212-632-5000`).
> 2. For each pair, build a binary agreement vector over **name, addr, city, phone, type**.
>    Use Jaro-Winkler similarity with a threshold for the text fields and exact digit match for phone.
>    Do **not** use the `class` column: it is the entity ID and would leak the label.
> 3. Estimate m_i = P(agree | M) and u_i = P(agree | U) from the training pairs, with
>    add-0.5 smoothing so no probability is 0 or 1.
> 4. Compute the weights: agree w_i = log2(m_i / u_i), disagree w_i = log2((1-m_i)/(1-u_i)),
>    and the total weight W = sum of w_i.
> 5. Choose two thresholds on the validation set: T_U = the highest W of any non-match and
>    T_L = the lowest W of any match, giving three regions: Match, Possible Match (manual review),
>    Non-Match.
> 6. On the test set, report the counts in each region, plus precision, recall and F1, both when
>    Possible Match is treated as a match and when it is treated as a non-match.
> 7. Print the m/u/weight table and a worked example for one pair, like the lecture slides.
>
> Use only pandas and the standard library. Keep it short and include assert-based self-checks.

## Prompt 2: follow-up after the first run

> Also print the weight W for every agreement pattern seen in the training data, with how many
> matches and non-matches have that pattern, so I can see which combination of agreeing
> fields gives the strongest evidence.
