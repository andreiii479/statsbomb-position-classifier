# StatsBomb Position Classifier

A machine-learning project (originally a PCLP3 course project) that predicts a
football player's **position** from their **per-90-minute match statistics**,
using StatsBomb's free event data from the **2018 and 2022 FIFA World Cups**.

Everything lives in a single Google Colab notebook:
[`Proiect_PCLP3_Statsbomb.ipynb`](Proiect_PCLP3_Statsbomb.ipynb). Code comments
and labels are in Romanian.

## What the notebook does

| Cell | Step | Notes |
|------|------|-------|
| 0 | Install `statsbombpy`, configure Colab table display | Colab-specific |
| 1 | **Load event data** for all 128 World Cup matches (64 + 64) | ~458k events × 115 columns. Cached to Parquet after the first run (see below) |
| 2 | Count relevant actions per player | Pass, Shot, Duel, Clearance, Interception, Foul Committed/Won, Dribble, Block, Pressure, Dispossessed, Ball Recovery, Goal Keeper |
| 3 | Add derived stats | Crosses, tackles, xG, xA (xG of the shot a pass assisted), avg. pass length, avg. shot distance. Missing values → 0 |
| 4 | Estimate minutes played | Per match: `minute_off − minute_on`, using substitution events and the last event minute of the match |
| 5–6 | Normalise to **per 90** and drop players with < 90 total minutes | Produces `tabel_p90` |
| 7 | Build the target label | Most-frequent StatsBomb position per player → grouped into 6 classes (see below). Adds 3 ratio features. Result: `dataset_final`, 839 players |
| 8–11 | EDA | Class pie chart, minutes histogram, box plots, correlation heatmap, scatter plot |
| 12 | Stratified 70/30 train/test split | Also writes `train.csv` / `test.csv` |
| 15 | Random Forest | Saved as `Position_Predictor.pkl` |
| 16–17 | Compare RF, Logistic Regression, KNN, SVM | Writes `train_scaled.csv` / `test_scaled.csv` |
| 18 | RF feature importances | |
| 19–20 | **Gradio app** | Sliders for each per-90 stat → predicted position probabilities; second app shows the confusion matrices |

### Target classes

| Label (RO) | Meaning | StatsBomb positions |
|---|---|---|
| `Portar` | Goalkeeper | Goalkeeper |
| `Fundas` | Centre-back | Center / Left CB / Right CB |
| `Fundas Lateral` | Full-back / wing-back | Left/Right Back, Left/Right Wing Back |
| `Mijlocas` | Central / defensive midfielder | CDM, CM, Left/Right DM |
| `Mijlocas Ofensiv` | Attacking mid / winger | LM, RM, CAM, LAM, RAM, Secondary Striker, LW, RW, **Left/Right Center Forward** |
| `Varf` | Striker | Center Forward |

An earlier 4-class version (GK/DEF/MID/FWD) scored about 0.78, but lumped
full-backs together with centre-backs even though full-backs play more like
midfielders.

### Features (22)

17 per-90 counts (`Pass_p90`, `Shot_p90`, …, `xG_p90`, `xA_p90`, `Goal Keeper_p90`),
`avg_pass_length`, `avg_shot_distance`, and three ratios:

- `Defensive_Work_Ratio = (Tackle + Interception + Ball Recovery) / (Pass + 1)`
- `xG_to_xA_Ratio = xG / (xA + 0.01)`
- `Touches_per_Shot = (Pass + Dribble + Duel) / (Shot + 1)`

### Last recorded results (test set, 252 players)

| Model | Accuracy | Weighted F1 |
|---|---|---|
| Random Forest | 0.774 | 0.772 |
| Logistic Regression | 0.766 | 0.765 |
| SVM (RBF) | 0.730 | 0.725 |
| KNN (k=5) | 0.663 | 0.658 |

With Random Forest, goalkeepers are perfect (F1 1.00) and centre-backs are
strong (0.90). `Varf` (0.59) and `Mijlocas Ofensiv` (0.69) are the weakest
classes, and they get confused with each other.

## Running it

### In Google Colab (intended)

Open the notebook in Colab and run the cells in order. Cell 1 asks for access
to your Google Drive. That's where the event cache is stored.

### Locally

```bash
pip install -r requirements.txt
jupyter notebook Proiect_PCLP3_Statsbomb.ipynb
```

Cell 0 imports `google.colab`, which fails outside Colab. Skip that cell
(and remove the `!pip install` line).

## Event data cache (no more re-downloading)

Downloading the events with `statsbombpy` takes about **2.5–3 minutes**
(128 HTTP requests). Cell 1 now saves the combined events DataFrame to a
Parquet file the first time it runs, and loads it from there on every later
run:

| | Time | Size |
|---|---|---|
| First run (download + save) | ~170 s | 52 MB |
| Later runs (read Parquet) | ~5 s | |

- **In Colab**, the cache is saved to
  `MyDrive/statsbomb_cache/evenimente_wc2018_wc2022.parquet` because Colab's
  local disk is wiped when the session ends. You'll need to approve the Drive
  mount once per session.
- **Locally**, it goes to `./statsbomb_cache/` (git-ignored).
- To force a fresh download, set `FORTEAZA_DESCARCARE = True` in cell 1 or
  delete the file.

Parquet was chosen over pickle because pickle was 10× larger (550 MB) and 3×
slower to load. One side effect is that list columns such as `location` come
back as numpy arrays instead of Python lists. I checked that the rest of the
notebook produces an identical `dataset_final` either way.

## Known issues

These were found while reviewing the notebook. None of them have been fixed
yet, apart from the caching.

### Bugs / correctness

1. **Scaler fitted on the test set (data leakage).** Cell 16 calls
   `scale.fit_transform(X_test)`. It should be `scale.transform(X_test)`.
   The test data gets scaled with its own mean and std instead of the
   training set's. Fixing it raises SVM accuracy from 0.730 to 0.758
   (LR 0.766 → 0.770), so the reported comparison is off.
2. **Players split by name spelling.** Everything is grouped by
   `['player_id', 'player']`, but StatsBomb spells some players differently
   between 2018 and 2022 (`Phil Foden` / `Philip Foden`, `N'Golo Kanté` /
   `N''Golo Kanté`, `Steven N'Kemboanza…` / `Steven N''Kemboanza…`,
   `Mathias Jørgensen Jatta` / `Mathias Jattah-Njie Jørgensen`). Each player
   ends up with two partial rows. This is where the "o dublura, scapata pe
   undeva" duplicate in cell 7 comes from. `drop_duplicates(subset=['player_id'])`
   then throws away one row, so Foden loses 148 of his 274 minutes of data.
   **Fix:** group by `player_id` only, and keep one name per id (e.g. the
   most recent).
3. **The substitution merge relies on names too.** Minutes played are matched
   on `substitution_replacement` (a name) against `player`. This is fragile
   for the same reason. It's better to use `substitution_replacement_id`.
4. **Estimated minutes played are rough.** Match length is the last minute
   in which any player event happened. Stoppage time and extra time are
   folded in inconsistently, and red cards aren't handled, so a sent-off
   player is credited with the full match.
5. **`Left/Right Center Forward` → `Mijlocas Ofensiv`.** In a two-striker
   system these are strikers, but they're labelled attacking midfielders while
   `Center Forward` is `Varf`. This probably explains a lot of the
   Varf ↔ Mijlocas Ofensiv confusion. If it was intentional, it's worth a
   comment.
6. **Unknown positions silently become `Mijlocas`** via `.fillna('Mijlocas')`.
   This hides unmapped positions (for example the `Substitute` case that is
   patched by hand for Predrag Rajković). It's better to fail loudly or drop
   those rows.
7. **Missing distances are filled with 0.** A player who never shot gets
   `avg_shot_distance = 0`, which reads as "shoots from the goal line".
   That's misleading for LR, KNN and SVM, and it also conflicts with the
   Gradio default of 16.

### Evaluation methodology

8. **The test set is used for model selection.** The RF hyper-parameters
   (`max_features=5`, `max_depth=10`, …) and the choice between models were
   evidently tuned against the same 252-player test set that the final
   scores are reported on, so those scores are optimistic. Use
   cross-validation (`GridSearchCV` / `cross_val_score`) on the training set
   and touch the test set once.
9. **One random split of ~840 samples.** With 19–21 test samples in some
   classes, a single split is noisy. Report CV mean ± std.
10. **`xG_to_xA_Ratio` has extreme outliers** (median 0.8, max 94) because of
    the `+ 0.01` denominator. This hurts the scaled models. Consider
    `xG / (xG + xA + ε)` (bounded 0–1) or a log transform.
11. **`warnings.filterwarnings('ignore')`** hides everything, including
    `LogisticRegression` convergence warnings and pandas warnings.

### Code structure / reproducibility

12. **Cells aren't idempotent.** Re-running cell 3 merges the same columns
    again (`Cross_x`, `Cross_y`, …). Cell 5 works around the same problem
    with `if 'minute_jucate' not in ...`. Build `statistici_finale` fresh
    inside the cell, or wrap the steps in functions.
13. **The feature list is defined three times** (cells 10 and 12, and
    implicitly again in the Gradio function). The ratio formulas are also
    duplicated in cell 7 and cell 19. If one changes, the app silently
    disagrees with the model. Define them once.
14. **Unused or duplicated work:** `suturi` is computed twice in cell 3.
    `competitii` is fetched but unused. The imports in cell 16 repeat
    cell 15, and the RF is trained twice.
15. **The scaled CSVs lose their column names and labels**
    (`pd.DataFrame(x_train_scaled)`), so they aren't very useful on their own.
16. **Colab-only code:** `google.colab.data_table` in cell 0 and `!pip`.
    `gradio`, `tabulate` and `seaborn` rely on Colab's pre-installed
    packages. There were no pinned dependencies, so `requirements.txt` has
    been added.
17. **The Gradio app only uses RF**, and only works if cell 15 ran in the same
    session (it loads `Position_Predictor.pkl` from local disk).
