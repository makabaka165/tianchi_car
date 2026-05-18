# tianchi_car Goal Tasks

## Current Baseline

- Stable commit: `7eb888a`
- Stable experiment: `task4_lgb_leaves127_cfg_t_eval`
- Stable candidate: `cfg_t_depth_9_iter4400`
- Stable CatBoost params: `iterations=4400`, `learning_rate=0.03`, `depth=9`, `l2_leaf_reg=9.0`, `od_wait=140`
- LightGBM OOF MAE: `526.6174419083302`
- CatBoost OOF MAE: `489.53917414313685`
- Blend OOF MAE: `481.39377557629706`
- Best weights: LightGBM `0.29`, CatBoost `0.71`
- Goal threshold: `Blend OOF MAE <= 481.2`

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

### Task 1 - CatBoost regularization around current best depth 9 candidate

- Status: `rollback`
- Direction: CatBoost structure tuning
- Code area: `code/main.py` candidate block only.
- Candidate name: `cfg_u_l2_10_depth9_iter4400`
- Change: keep the current best candidate shape and change only `l2_leaf_reg` from `9.0` to `10.0`.
- Params: `iterations=4400`, `learning_rate=0.03`, `depth=9`, `l2_leaf_reg=10.0`, `od_wait=140`.
- Command:
  `python -u code/main.py catboost_only_sweep --candidate cfg_u_l2_10_depth9_iter4400 --experiment-note task1_l2_10_depth9_eval --baseline-blend 482.03784116490414 --baseline-commit e1ae9fc`
- Keep rule: keep only if the final blended `blend_oof_mae` is strictly lower than `482.03784116490414`.
- If failed: revert code, restore backed-up artifacts, record failure, and continue to Task 2.

### Task 2 - CatBoost overfitting control follow-up

- Status: `rollback`
- Direction: CatBoost structure tuning
- Code area: `code/main.py` candidate block only.
- Candidate name: `cfg_v_od160_depth9_iter4400`
- Change: keep the current best candidate shape and change only `od_wait` from `140` to `160`.
- Params: `iterations=4400`, `learning_rate=0.03`, `depth=9`, `l2_leaf_reg=9.0`, `od_wait=160`.
- Keep rule: keep only if the final blended `blend_oof_mae` is strictly lower than the latest stable baseline.
- If failed: revert code, restore backed-up artifacts, record failure, and continue to Task 3.

### Task 3 - One narrowly scoped feature extension

- Status: `rollback`
- Direction: lightweight feature engineering
- Code area: `feature/preprocess.py` only.
- Allowed family: one tightly bounded feature block built from already-successful stable signals, for example `v_range`, `v_std`, `power_log1p`, `kilometer_log1p`, or `car_age_years`. Last attempted subtask: age-normalized interactions for power/kilometer/v statistics.
- Constraint: add one coherent family only; do not bundle unrelated features.
- Constraint: no target leakage and no train-only statistics unless implemented through existing fold-aware logic.
- If the change affects LightGBM inputs, refresh the LightGBM cache before evaluation.
- Keep rule: keep only if the final blended `blend_oof_mae` is strictly lower than the latest stable baseline.

### Task 4 - LightGBM follow-up only if needed

- Status: `rollback`
- Direction: LightGBM follow-up tuning
- Code area: `model/train.py` only, plus minimal calling path if necessary.
- Purpose: revisit the LightGBM branch only if CatBoost structure tuning and the next feature block fail to produce enough gain. Last attempted subtask: increase reg_alpha from 0.2 to 0.3 with all other LightGBM and CatBoost settings fixed at the current stable baseline.
- Constraint: keep the training entry shape stable and make only one parameter change group per round.
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


### Task 1 Result - task1_l2_10_depth9_eval

- Status: rollback
- Started at: 2026-05-18T19:37:24
- Finished at: 2026-05-18T20:10:22
- Baseline commit: e1ae9fc
- Baseline Blend OOF MAE: 482.03784116490414
- Candidate/change: cfg_u_l2_10_depth9_iter4400, change only l2_leaf_reg from 9.0 to 10.0 on the current best depth-9 candidate
- Command: python -u code/main.py catboost_only_sweep --candidate cfg_u_l2_10_depth9_iter4400 --experiment-note task1_l2_10_depth9_eval --baseline-blend 482.03784116490414 --baseline-commit e1ae9fc
- LightGBM OOF MAE: 530.2642031884868
- CatBoost OOF MAE: 489.8325008519237
- Blend OOF MAE: 482.4913544274493
- Best weights: LightGBM 0.27, CatBoost 0.73
- Fold scores: LightGBM [535.2210926036495, 532.9259499831855, 526.8230300397627, 528.7427922007341, 527.6081511151025]; CatBoost [493.31593029580733, 492.32610039161375, 488.00924971638057, 488.2480633778327, 487.2631604779842]
- Prediction file check: restored to stable prediction, header SaleID,price
- Commit pushed: pending documentation commit
- Rollback commit if any: code restored to e1ae9fc
- Next task: Task 2 cfg_v_od160_depth9_iter4400


### Task 2 Result - task2_od160_depth9_eval

- Status: rollback
- Started at: 2026-05-18T20:40:09
- Finished at: 2026-05-18T21:13:08
- Baseline commit: e1ae9fc
- Baseline Blend OOF MAE: 482.03784116490414
- Candidate/change: cfg_v_od160_depth9_iter4400, change only od_wait from 140 to 160 on the current best depth-9 candidate
- Command: python -u code/main.py catboost_only_sweep --candidate cfg_v_od160_depth9_iter4400 --experiment-note task2_od160_depth9_eval --baseline-blend 482.03784116490414 --baseline-commit e1ae9fc
- LightGBM OOF MAE: 530.2642031884868
- CatBoost OOF MAE: 489.53917414313685
- Blend OOF MAE: 482.03784116490414
- Best weights: LightGBM 0.27, CatBoost 0.73
- Fold scores: LightGBM [535.2210926036495, 532.9259499831855, 526.8230300397627, 528.7427922007341, 527.6081511151025]; CatBoost [493.6952723244014, 491.33982441310206, 486.93385262494326, 487.25305623185795, 488.4738651213799]
- Prediction file check: restored to stable prediction, header SaleID,price
- Commit pushed: pending documentation commit
- Rollback commit if any: code restored to e1ae9fc
- Next task: Task 3 one narrowly scoped feature extension

### Task 3 Result - task3_age_usage_cfg_t_eval

- Status: rollback
- Started at: 2026-05-18T21:15:14
- Finished at: 2026-05-18T22:00:30
- Baseline commit: e1ae9fc
- Baseline Blend OOF MAE: 482.03784116490414
- Candidate/change: age-normalized interactions for power_log1p, kilometer_log1p, v_range, and v_std against car_age_years
- Command: python -u code/main.py catboost_only_sweep --candidate cfg_t_depth_9_iter4400 --experiment-note task3_age_usage_cfg_t_eval --baseline-blend 482.03784116490414 --baseline-commit e1ae9fc
- LightGBM OOF MAE: 531.1280636750124
- CatBoost OOF MAE: 492.4269839188377
- Blend OOF MAE: 484.3071509217038
- Best weights: LightGBM 0.28, CatBoost 0.72
- Fold scores: LightGBM [536.7833856716709, 534.3058305634485, 529.4267280317765, 528.2944635575448, 526.8299105506218]; CatBoost [495.1977134628182, 495.1625166727634, 489.20774655406115, 490.22135062299276, 492.3455922815534]
- Prediction file check: restored to stable prediction, header SaleID,price
- Commit pushed: pending documentation commit
- Rollback commit if any: code restored to e1ae9fc; metrics and predictions restored from 20260518T211514 backup
- Next task: Task 4 LightGBM single-variable follow-up

### Task 4 Result - task4_lgb_leaves127_cfg_t_eval

- Status: keep
- Started at: 2026-05-18T23:46:00
- Finished at: 2026-05-19T00:33:25
- Baseline commit: e1ae9fc
- Baseline Blend OOF MAE: 482.03784116490414
- Candidate/change: LightGBM num_leaves 95 -> 127; CatBoost kept at cfg_t_depth_9_iter4400
- Command: python -u code/main.py catboost_only_sweep --candidate cfg_t_depth_9_iter4400 --experiment-note task4_lgb_leaves127_cfg_t_eval --baseline-blend 482.03784116490414 --baseline-commit e1ae9fc
- LightGBM OOF MAE: 526.6174419083302
- CatBoost OOF MAE: 489.53917414313685
- Blend OOF MAE: 481.39377557629706
- Best weights: LightGBM 0.29, CatBoost 0.71
- Fold scores: LightGBM [534.663656597508, 529.9643914717552, 522.2402307516778, 525.1418546805704, 521.0770760401389]; CatBoost [493.6952723244014, 491.33982441310206, 486.93385262494326, 487.25305623185795, 488.4738651213799]
- Prediction file check: current prediction retained, header SaleID,price
- Commit pushed: pending keep commit
- Rollback commit if any: none
- Next task: LightGBM near-field follow-up around the kept leaves-127 branch

### Task 5 Result - task5_lgb_reglambda03_cfg_t_eval

- Status: rollback
- Started at: 2026-05-19T02:07:10
- Finished at: 2026-05-19T02:54:51
- Baseline commit: 7eb888a
- Baseline Blend OOF MAE: 481.39377557629706
- Candidate/change: LightGBM reg_lambda 0.2 -> 0.3; CatBoost kept at cfg_t_depth_9_iter4400
- Command: python -u code/main.py catboost_only_sweep --candidate cfg_t_depth_9_iter4400 --experiment-note task5_lgb_reglambda03_cfg_t_eval --baseline-blend 481.39377557629706 --baseline-commit 7eb888a
- LightGBM OOF MAE: 527.1386978856078
- CatBoost OOF MAE: 489.53917414313685
- Blend OOF MAE: 481.4606157775679
- Best weights: LightGBM 0.28, CatBoost 0.72
- Fold scores: LightGBM [534.0191487528388, 529.4135666596577, 523.9159609023715, 527.2355111152879, 521.1093019978832]; CatBoost [493.6952723244014, 491.33982441310206, 486.93385262494326, 487.25305623185795, 488.4738651213799]
- Prediction file check: restored to stable prediction, header SaleID,price
- Commit pushed: pending rollback commit
- Rollback commit if any: code restored to 7eb888a; metrics, predictions, and LightGBM cache restored from 20260519T020710 backup
- Next task: LightGBM near-field follow-up with a different single variable, preferably reg_alpha 0.2 -> 0.3

### Task 6 Result - task6_lgb_regalpha03_cfg_t_eval

- Status: rollback
- Started at: 2026-05-19T04:23:34
- Finished at: 2026-05-19T05:10:34
- Baseline commit: 7eb888a
- Baseline Blend OOF MAE: 481.39377557629706
- Candidate/change: LightGBM reg_alpha 0.2 -> 0.3; CatBoost kept at cfg_t_depth_9_iter4400
- Command: python -u code/main.py catboost_only_sweep --candidate cfg_t_depth_9_iter4400 --experiment-note task6_lgb_regalpha03_cfg_t_eval --baseline-blend 481.39377557629706 --baseline-commit 7eb888a
- LightGBM OOF MAE: 526.873219018996
- CatBoost OOF MAE: 489.53917414313685
- Blend OOF MAE: 481.4045374727021
- Best weights: LightGBM 0.29, CatBoost 0.71
- Fold scores: LightGBM [534.785069591943, 528.6353825690522, 522.9684995520128, 524.5795968954623, 523.3975464865093]; CatBoost [493.6952723244014, 491.33982441310206, 486.93385262494326, 487.25305623185795, 488.4738651213799]
- Prediction file check: restored to stable prediction, header SaleID,price
- Commit pushed: pending rollback commit
- Rollback commit if any: code restored to 7eb888a; metrics, predictions, and LightGBM cache restored from 20260519T042334 backup
- Next task: switch direction to blend refinement on the kept leaves-127 branch

## Current Recommendation

Two consecutive LightGBM regularization follow-ups have failed on the kept leaves-127 branch, so this direction is temporarily exhausted under the current experiment discipline. The next round should switch direction and test a single blend-refinement change in `code/main.py`, specifically a finer blend-weight search around the current best region, while keeping the stable LightGBM and CatBoost branches unchanged.

