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

- Status: `rollback`
- Direction: lightweight feature engineering
- Code area: `feature/preprocess.py` only.
- Allowed family: one coherent group of low-risk derived features built from existing stable columns, for example interactions involving `v_mean`, `v_std`, `v_range`, `power_log1p`, `kilometer_log1p`, or `car_age_years`. Current subtask: v-aggregate interactions with `power_log1p` and `kilometer_log1p`.
- Constraint: do not mix multiple unrelated feature families in one round.
- Constraint: no target leakage and no train-only statistics unless implemented with fold-aware logic already present in the pipeline.
- Keep rule: keep only if the final blended `blend_oof_mae` is strictly lower than the latest stable baseline.
- If failed: revert code, restore backed-up artifacts, record the failure, and continue to Task 3.

### Task 3 - Blend refinement after branch updates

- Status: `pending`
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


### Task 2 Result - task2_v_interactions_cfg_q_eval

- Status: rollback
- Started at: 2026-05-18T16:35:01
- Finished at: 2026-05-18T17:17:14
- Baseline commit: 8190f22
- Baseline Blend OOF MAE: 483.17528941350145
- Candidate/change: add one v-derived interaction block using v_mean/v_std/v_range with power_log1p and kilometer_log1p, then refresh cache and rerun cfg_q_iterations_4400
- Command: python -u code/main.py catboost_only_sweep --candidate cfg_q_iterations_4400 --experiment-note task2_v_interactions_cfg_q_eval --baseline-blend 483.17528941350145 --baseline-commit 8190f22
- LightGBM OOF MAE: 531.5371602819139
- CatBoost OOF MAE: 492.2656291284221
- Blend OOF MAE: 484.2770503355926
- Best weights: LightGBM 0.28, CatBoost 0.72
- Fold scores: LightGBM [539.4349873980088, 533.2364755242635, 527.9325232074102, 529.5131165854559, 527.5686986944302]; CatBoost [496.2768103726382, 496.3073190836549, 488.7926459372694, 488.1527772014676, 491.7985930470809]
- Prediction file check: restored to task1 stable prediction, header SaleID,price
- Commit pushed: pending documentation commit
- Rollback commit if any: code restored to 8190f22
- Next task: Task 3 blend refinement

## Current Recommendation

Task 2 v-derived interactions rolled back because the blended score worsened to `484.2770503355926`. Next run Task 3 blend refinement on top of the current stable branches; do not add new branch features in the same round.

