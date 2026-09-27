# Code review notes

Issues found while reviewing the notebook, and what has been done about them.
Cell numbers refer to [`statsbomb_position_classifier.ipynb`](../statsbomb_position_classifier.ipynb).

## Fixed

1. **Scaler fitted on the test set (data leakage).** Cell 16 called
   `scale.fit_transform(X_test)`, which scaled the test data with its own
   mean and std instead of the training set's. It now calls
   `scale.transform(X_test)`. On the split used at the time, this alone
   raised SVM accuracy from 0.730 to 0.758.
2. **Players split by name spelling.** Stats were grouped by
   `['player_id', 'player']`, but StatsBomb spells some players differently
   between 2018 and 2022 (`Phil Foden` / `Philip Foden`, `N'Golo Kanté` /
   `N''Golo Kanté`, `Mathias Jørgensen Jatta` / `Mathias Jattah-Njie Jørgensen`, …).
   Each of these players ended up as two partial rows, and a
   `drop_duplicates` then threw one away. For example, Foden lost 148 of his
   274 minutes. Cell 1 now gives each `player_id` a single name, and cell 7
   asserts there are no duplicates, so the notebook stops instead of
   silently dropping data.
3. **The substitution merge relied on names.** Minutes played are matched
   on `substitution_replacement` (a name). This column now uses the same
   id → name map as `player`, so the names always agree.
4. **The test set was used for model selection.** The RF hyperparameters
   were picked against the test set. Cell 18 now compares models with
   5-fold cross-validation on the training set, and cell 19 tunes the RF
   with `GridSearchCV` on the training set only.
5. **Results came from one noisy split.** Cell 18 reports CV mean ± std
   and a most-frequent-class baseline.
6. **Duplicated work.** Shots were computed twice, `sb.competitions()` was
   fetched but never used, the RF was trained twice, and the feature list
   was defined in two cells. All of this has been cleaned up.
7. **The notebook only ran in Colab.** Cell 0 now only imports
   `google.colab` when running in Colab, and uses `%pip`.
8. **Romanian identifiers and labels.** The code, comments, plots, class
   names and app are now in English.

## Open

1. **`Left/Right Center Forward` are labelled `Attacking Midfielder`**,
   while `Center Forward` is `Striker`. In a two-striker system these are
   strikers. This is the likely cause of the biggest error in the confusion
   matrix: 16 of 62 attacking midfielders are predicted as strikers.
2. **Estimated minutes played are rough.** Match length is the last minute
   with a player event, so stoppage time and extra time are handled
   inconsistently, and red cards are ignored. A sent-off player is credited
   with the full match.
3. **Unknown positions silently become `Central Midfielder`** via `.fillna(...)`.
   Failing loudly or dropping those rows would be safer.
4. **Missing distances are filled with 0.** A player who never shot gets
   `avg_shot_distance = 0` ("shoots from the goal line"). This misleads the
   distance-based models, and it conflicts with the app's default of 16.
5. **`xG_to_xA_Ratio` has extreme outliers** (median 0.8, max 94) because of
   the `+ 0.01` denominator. `xG / (xG + xA + ε)` or a log transform would
   bound it.
6. **`warnings.filterwarnings('ignore')`** hides every warning, including
   convergence warnings.
7. **Cells aren't idempotent.** Re-running cell 3 merges the same columns
   again (`Cross_x`, `Cross_y`, …).
8. **The ratio formulas are duplicated** in cell 7 and in the Gradio
   function (cell 21). If one changes, the app silently disagrees with the
   model.
9. **The scaled CSVs lose their column names and labels.**
10. **The app needs cells 15 and 19 to have run in the same session**,
    because it loads `Position_Predictor.pkl` from local disk.
