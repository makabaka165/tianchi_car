# tianchi_car Goal Plan

## Goal

Use the remote server repository `~/tianchi_car` as the only execution environment and continue the used-car price prediction optimization until the best local validation result reaches:

- Target metric: `user_data/metrics.json -> blend_oof_mae`
- Current stable baseline: commit `b031884`, `Blend OOF MAE = 490.2366133835716`
- Goal threshold: `Blend OOF MAE <= 488.8`

This is intentionally a medium-range goal. It should require multiple experiment rounds across parameters, regularization, feature engineering, and blending. Do not treat the goal as complete after one or two short changes unless the metric actually reaches `<= 488.8` and the result is committed and pushed.

## Execution Rules

- All code changes, training, evaluation, commits, and pushes must happen on the remote server in `~/tianchi_car`.
- The local repository is only for pulling or viewing code; do not use it for implementation or training.
- The `/goal` execution should not create the planning documents from scratch. Read and follow `goal_plan.md`, update `goal_tasks.md`, and append normal experiment results to the root Markdown record file identified by UTF-8 byte prefix `23 20 e8 ae b0 e5 bd 95`; do not hard-code the displayed filename.
- Before each experiment, check `git status --short --branch`, current `HEAD`, `user_data/metrics.json`, LightGBM cache files, and whether any training process is already running.
- Before each experiment, back up `user_data/metrics.json` and `prediction_result/predictions.csv` into `user_data/experiment_backups/` with a timestamped filename.
- Each round must change exactly one clear experimental factor unless the change is only documentation or recovery logic.
- Official scoring uses only `blend_oof_mae` from `user_data/metrics.json`.
- Keep a result only if `blend_oof_mae` is strictly lower than the current best baseline.
- If a result is not better, keep the experiment record, revert code to the previous stable commit, restore the backed-up metrics and prediction files, then commit only the failed experiment documentation if needed.
- After every kept experiment, commit and push to `origin/main`, then update the current best commit and score in `goal_tasks.md`.

## Experiment Strategy

Prioritize experiments in this order, switching direction when a direction stalls:

1. CatBoost near-neighbor parameter experiments around the current best candidate.
2. CatBoost regularization or overfitting-control experiments at the best compute level.
3. Lightweight no-leakage feature experiments that are consistent for train and test.
4. Blend-weight or blend-resolution experiments if model outputs are available and useful.
5. LightGBM parameter work only after CatBoost and lightweight feature directions show limited gains.

The current trend shows that increasing CatBoost iterations from `3400` through `4200` has repeatedly improved OOF. The next attempt may test `cfg_q_iterations_4400`, but it must be evaluated against runtime and diminishing returns. If the next pure-iteration experiment fails, stop blindly increasing iterations and switch to same-compute regularization such as `iterations=4200, l2_leaf_reg=10.0`.

## Stability And Runtime Constraints

- Long runs are acceptable, but the agent must report progress and avoid starting duplicate training processes.
- A single experiment can exceed 90 minutes if the process is active and CPU usage indicates normal training.
- Do not start a second training process while another `code/main.py` or `catboost_only_sweep` process is running.
- Do not delete failed experiment records from the root Markdown record file identified by UTF-8 byte prefix `23 20 e8 ae b0 e5 bd 95`; do not hard-code the displayed filename or `goal_tasks.md`.
- Do not use leaderboard assumptions. Local OOF MAE is the decision metric.

## Completion Criteria

The goal is complete only when all of the following are true:

- `blend_oof_mae <= 488.8` in `user_data/metrics.json`.
- The improving code and experiment record are committed and pushed to `origin/main`.
- `goal_tasks.md` records the final best commit, metric, candidate, and next recommendation.
- The server working tree is clean.

If progress stalls before reaching the threshold, summarize the plateau only after at least two different directions fail to improve, for example pure iterations and regularization, or regularization and lightweight features.
