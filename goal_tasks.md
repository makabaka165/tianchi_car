# tianchi_car Goal Tasks

## Current Baseline

- Stable commit: `9697305`
- Stable experiment: `cfg_q_iterations_4400_eval`
- Stable candidate: `cfg_q_iterations_4400`
- Stable CatBoost params: `iterations=4400`, `learning_rate=0.03`, `depth=8`, `l2_leaf_reg=9.0`, `od_wait=140`
- LightGBM OOF MAE: `538.2950818411284`
- CatBoost OOF MAE: `496.5144844289851`
- Blend OOF MAE: `489.3423041102956`
- Best weights: LightGBM `0.28`, CatBoost `0.72`
- Goal threshold: `Blend OOF MAE <= 488.8`

## Standing Checklist For Every Round

- Check `git status --short --branch`; continue only if the server worktree is clean or the dirty files are known documentation updates from the current goal run.
- Check `git rev-parse --short HEAD` and compare it with the current stable commit recorded above.
- Check `user_data/metrics.json` and confirm the current stable `blend_oof_mae` before choosing a new experiment.
- Confirm LightGBM cache exists: `user_data/lgb_cache_predictions.npz` and `user_data/lgb_cache_meta.json`.
- Check no duplicate training process is running with `pgrep -af 'code/main.py|catboost_only_sweep'`.
- Back up `user_data/metrics.json` and `prediction_result/predictions.csv` before training.
- Run one experiment only, then decide keep or rollback using `blend_oof_mae`.
- Append result details to the root Markdown record file identified by UTF-8 byte prefix `23 20 e8 ae b0 e5 bd 95`; do not hard-code the displayed filename and update this file.

## Task Queue

### Task 1 - Pure iteration neighbor after current best

- Status: `keep`
- Candidate name: `cfg_q_iterations_4400`
- Code change: add one CatBoost candidate in `code/main.py`.
- Params: `iterations=4400`, `learning_rate=0.03`, `depth=8`, `l2_leaf_reg=9.0`, `od_wait=140`.
- Command:
  `python -u code/main.py catboost_only_sweep --candidate cfg_q_iterations_4400 --experiment-note cfg_q_iterations_4400_eval --baseline-blend 490.2366133835716 --baseline-commit b031884`
- Keep rule: keep only if `blend_oof_mae < 490.2366133835716`.
- If kept: commit message `Promote cfg_q iterations 4400`, push `origin/main`, update the current baseline block.
- If failed: revert `code/main.py` to `b031884`, restore backed-up metrics and predictions, record failure, then continue to Task 2.

### Task 2 - Same-compute regularization after iteration plateau

- Status: `rollback`
- Candidate name: `cfg_r_l2_10_iter4200`
- Code change: add one CatBoost candidate in `code/main.py`.
- Params: `iterations=4200`, `learning_rate=0.03`, `depth=8`, `l2_leaf_reg=10.0`, `od_wait=140`.
- Baseline: use the latest stable commit and score after Task 1, not stale values.
- Keep rule: keep only if `blend_oof_mae` strictly improves over the latest stable baseline.
- If failed: revert code, restore backed-up artifacts, record failure, then continue to Task 3.

### Task 3 - Slightly stronger overfitting control

- Status: `rollback`
- Candidate name: `cfg_s_od160_iter4200`
- Code change: add one CatBoost candidate in `code/main.py`.
- Params: `iterations=4200`, `learning_rate=0.03`, `depth=8`, `l2_leaf_reg=9.0`, `od_wait=160`.
- Purpose: test whether longer patience improves fold stability without changing model depth or learning rate.
- Keep rule: keep only if `blend_oof_mae` strictly improves over the latest stable baseline.

### Task 4 - Lightweight feature experiment if parameter gains stall

- Status: `pending`
- Code area: `feature/preprocess.py` only.
- Allowed feature family: low-risk numeric interaction or consistency indicators available in both train and test.
- Avoid target leakage and avoid train-only statistics unless implemented with fold-aware logic.
- After feature changes, run the current best CatBoost candidate and refresh any incompatible caches according to the existing pipeline behavior.
- Keep rule: keep only if `blend_oof_mae` strictly improves over the latest stable baseline and fold volatility does not visibly worsen.

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

### Task 1 Result - cfg_q_iterations_4400_eval

- Status: keep
- Started at: 2026-05-18T12:44:07
- Finished at: 2026-05-18T13:09:47
- Baseline commit: b031884
- Baseline Blend OOF MAE: 490.2366133835716
- Candidate/change: cfg_q_iterations_4400, iterations=4400, learning_rate=0.03, depth=8, l2_leaf_reg=9.0, od_wait=140
- Command: python -u code/main.py catboost_only_sweep --candidate cfg_q_iterations_4400 --experiment-note cfg_q_iterations_4400_eval --baseline-blend 490.2366133835716 --baseline-commit b031884
- LightGBM OOF MAE: 538.2950818411284
- CatBoost OOF MAE: 496.5144844289851
- Blend OOF MAE: 489.3423041102956
- Best weights: LightGBM 0.28, CatBoost 0.72
- Fold scores: [502.2674446842645, 500.4405521813541, 493.1370185470812, 491.02515317324384, 495.702253558982]
- Prediction file check: passed, header SaleID,price
- Commit pushed: 9697305
- Rollback commit if any: none
- Next task: Task 2 cfg_r_l2_10_iter4200


### Task 2 Result - cfg_r_l2_10_iter4200_eval

- Status: rollback
- Started at: 2026-05-18T13:19:35
- Finished at: 2026-05-18T13:44:13
- Baseline commit: 9697305
- Baseline Blend OOF MAE: 489.3423041102956
- Candidate/change: cfg_r_l2_10_iter4200, iterations=4200, learning_rate=0.03, depth=8, l2_leaf_reg=10.0, od_wait=140
- Command: python -u code/main.py catboost_only_sweep --candidate cfg_r_l2_10_iter4200 --experiment-note cfg_r_l2_10_iter4200_eval --baseline-blend 489.3423041102956 --baseline-commit 9697305
- LightGBM OOF MAE: 538.2950818411284
- CatBoost OOF MAE: 498.7893635663284
- Blend OOF MAE: 490.93110818274977
- Best weights: LightGBM 0.28, CatBoost 0.72
- Fold scores: [502.8952189693916, 502.27204317354324, 497.8279570963196, 494.30377639158615, 496.6478222008014]
- Prediction file check: restored to cfg_q stable prediction, header SaleID,price
- Commit pushed: pending documentation commit
- Rollback commit if any: code restored to 9697305
- Next task: Task 3 cfg_s_od160_iter4200


### Task 3 Result - cfg_s_od160_iter4200_eval

- Status: rollback
- Started at: 2026-05-18T13:53:18
- Finished at: 2026-05-18T14:18:00
- Baseline commit: 9697305
- Baseline Blend OOF MAE: 489.3423041102956
- Candidate/change: cfg_s_od160_iter4200, iterations=4200, learning_rate=0.03, depth=8, l2_leaf_reg=9.0, od_wait=160
- Command: python -u code/main.py catboost_only_sweep --candidate cfg_s_od160_iter4200 --experiment-note cfg_s_od160_iter4200_eval --baseline-blend 489.3423041102956 --baseline-commit 9697305
- LightGBM OOF MAE: 538.2950818411284
- CatBoost OOF MAE: 497.7227619032235
- Blend OOF MAE: 490.2366133835716
- Best weights: LightGBM 0.28, CatBoost 0.72
- Fold scores: [503.61517916304774, 501.74759209051547, 494.2228763268931, 492.1676202537083, 496.860541681953]
- Prediction file check: restored to cfg_q stable prediction, header SaleID,price
- Commit pushed: pending documentation commit
- Rollback commit if any: code restored to 9697305
- Next task: Task 4 lightweight feature experiment

## Current Recommendation

Task 3 `cfg_s_od160_iter4200` also rolled back. Two parameter directions have now failed after the `cfg_q` baseline, so the next stage must move to Task 4 lightweight no-leakage feature work rather than continuing CatBoost parameter tweaks.
