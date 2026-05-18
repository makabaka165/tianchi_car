# tianchi_car Goal Tasks

## Current Baseline

- Stable commit: `62dffdb`
- Stable experiment: `task4_vstats_cfg_q_eval`
- Stable candidate: `cfg_q_iterations_4400`
- Stable CatBoost params: `iterations=4400`, `learning_rate=0.03`, `depth=8`, `l2_leaf_reg=9.0`, `od_wait=140`
- LightGBM OOF MAE: `535.3462383501875`
- CatBoost OOF MAE: `490.8413807472432`
- Blend OOF MAE: `483.96430195171246`
- Best weights: LightGBM `0.26`, CatBoost `0.74`
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

- Status: `pending`
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

- Status: `blocked until Task 1 finishes or is skipped for a documented reason`
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

## Current Recommendation

Start with Task 1. The current blended result is already strong, but the LightGBM branch remains materially weaker than CatBoost, even after the new `v_*` aggregate block. The most defensible next gain source is to refresh and strengthen LightGBM first, then re-check the blend before adding more CatBoost complexity.

