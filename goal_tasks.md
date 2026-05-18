# tianchi_car Goal Tasks

## Current Baseline

- Stable commit: `8190f22`
- Stable experiment: `task1_lgb_refresh_cfg_q_eval`
- Stable candidate: `cfg_q_iterations_4400`
- Stable CatBoost params: `iterations=4400`, `learning_rate=0.03`, `depth=8`, `l2_leaf_reg=9.0`, `od_wait=140`
- LightGBM OOF MAE: `530.2642031884868`
- CatBoost OOF MAE: `490.8413807472432`
- Blend OOF MAE: `483.17528941350145`
- Best weights: LightGBM `0.28`, CatBoost `0.72`
- Goal threshold: `Blend OOF MAE <= 482.8`

## Standing Checklist For Every Round

- Check `git status --short --branch`; continue only if the server worktree is clean or the dirty files are known documentation updates from the current goal run.
- Check `git rev-parse --short HEAD` and compare it with the current stable commit recorded above.
- Check `user_data/metrics.json` and confirm the current stable `blend_oof_mae` before choosing a new experiment.
- Confirm LightGBM cache exists: `user_data/lgb_cache_predictions.npz` and `user_data/lgb_cache_meta.json`.
- Check no duplicate training process is running with `pgrep -af 'code/main.py|catboost_only_sweep'`.
- Back up `user_data/metrics.json` and `prediction_result/predictions.csv` before training.
- Run one experiment only, then decide keep or rollback using `blend_oof_mae`.
- Append result details to the root Markdown record file identified by UTF-8 byte prefix `23 20 e8 ae b0 e5 bd 95`; do not hard-code the displayed filename, and update this file.

## Task Queue

### Task 1 - LightGBM branch refresh on current feature set

- Status: `keep`
- Direction: LightGBM improvement
- Code area: `model/train.py` and only the minimal calling path required by the existing pipeline.
- Primary purpose: improve the weaker LightGBM branch now that `v_*` aggregate features have already proven useful.
- Initial change scope: small LightGBM parameter search around the current settings, keeping the training entry shape unchanged.
- Recommended first candidate adjustment: modest leaf and regularization tuning while retaining the current learning rate and estimator scale.
- Required execution rule: rebuild the LightGBM cache before evaluating the blended result if the cache would otherwise stay stale.
- Keep rule: keep only if the final blended `blend_oof_mae` is strictly lower than `483.96430195171246`.
- If kept: commit and push, then update the baseline block above.
- If failed: revert code, restore backed-up artifacts, record the failure, and continue to Task 2.

### Task 2 - One additional low-risk v-derived interaction block

- Status: `pending`
- Direction: lightweight feature engineering
- Code area: `feature/preprocess.py` only.
- Allowed family: one coherent group of low-risk derived features built from existing stable columns, for example interactions involving `v_mean`, `v_std`, `v_range`, `power_log1p`, `kilometer_log1p`, or `car_age_years`.
- Constraint: do not mix multiple unrelated feature families in one round.
- Constraint: no target leakage and no train-only statistics unless implemented with fold-aware logic already present in the pipeline.
- Keep rule: keep only if the final blended `blend_oof_mae` is strictly lower than the latest stable baseline.
- If failed: revert code, restore backed-up artifacts, record the failure, and continue to Task 3.

### Task 3 - Blend refinement after branch updates

- Status: `blocked until Task 1 or Task 2 yields a plausible branch improvement`
- Direction: blend search refinement
- Code area: `code/main.py` only if search resolution or blend logic actually needs a controlled adjustment.
- Purpose: capture additional gain from better branch balance after LightGBM or feature improvements.
- Constraint: do not broaden this into a multi-model redesign; keep it as a small refinement to the current two-branch blend process.
- Keep rule: keep only if the final blended `blend_oof_mae` is strictly lower than the latest stable baseline.

### Task 4 - CatBoost structure tuning only after non-iteration work

- Status: `blocked until at least two non-iteration directions are tried`
- Direction: CatBoost structure tuning
- Code area: `code/main.py` candidate block only.
- Allowed parameters: choose one of `depth`, `l2_leaf_reg`, or another bounded overfitting-control parameter per round.
- Constraint: do not resume pure `iterations` growth first.
- Keep rule: keep only if the final blended `blend_oof_mae` is strictly lower than the latest stable baseline.

## Result Log Template

Use this template after each round:

```text
### Task N Result - <experiment_note>

- Status: keep | rollback | running | blocked
- Started at:
- Finished at:
- Baseline commit:
- Baseline Blend OOF MAE:
- Candidate/change:
- Command:
- LightGBM OOF MAE:
- CatBoost OOF MAE:
- Blend OOF MAE:
- Best weights:
- Fold scores:
- Prediction file check:
- Commit pushed:
- Rollback commit if any:
- Next task:
```


### Task 1 Result - task1_lgb_refresh_cfg_q_eval

- Status: keep
- Started at: 2026-05-18T15:36:06
- Finished at: 2026-05-18T16:28:40
- Baseline commit: 62dffdb
- Baseline Blend OOF MAE: 483.96430195171246
- Candidate/change: LightGBM parameter refresh on current feature set; num_leaves 63 -> 95, reg_alpha 0.1 -> 0.2, reg_lambda 0.1 -> 0.2, then refresh cache and rerun cfg_q_iterations_4400
- Command: python -u code/main.py catboost_only_sweep --candidate cfg_q_iterations_4400 --experiment-note task1_lgb_refresh_cfg_q_eval --baseline-blend 483.96430195171246 --baseline-commit 62dffdb
- LightGBM OOF MAE: 530.2642031884868
- CatBoost OOF MAE: 490.8413807472432
- Blend OOF MAE: 483.17528941350145
- Best weights: LightGBM 0.28, CatBoost 0.72
- Fold scores: LightGBM [535.2210926036495, 532.9259499831855, 526.8230300397627, 528.7427922007341, 527.6081511151025]; CatBoost [494.12883918397375, 496.17446295398645, 487.87079826675654, 486.114736507389, 489.9180668241097]
- Prediction file check: passed, header SaleID,price
- Commit pushed: 8190f22
- Rollback commit if any: none
- Next task: Task 2 low-risk v-derived interaction block

## Current Recommendation

Task 1 improved the LightGBM branch and lowered the blended score to `483.17528941350145`, but the goal `<= 482.8` is still not met. Next run Task 2 with one coherent low-risk v-derived interaction block; do not add more LightGBM changes in the same round.

