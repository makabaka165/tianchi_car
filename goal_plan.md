# tianchi_car Goal Plan

## Goal

Use the remote server repository `~/tianchi_car` as the only execution environment and continue optimizing the used-car price prediction pipeline from the current stable baseline:

- Target metric: `user_data/metrics.json -> blend_oof_mae`
- Current stable baseline: commit `e1ae9fc`, experiment `task4_depth9_cfg_t_eval`
- Current stable score: `Blend OOF MAE = 482.03784116490414`
- New goal threshold: `Blend OOF MAE <= 481.2`

This is a tighter follow-up target. It should require several more controlled experiments and should not be treated as a one- or two-edit objective. Success requires a real kept result at or below `481.2`, committed and pushed from the server.

## Execution Rules

- All code changes, training, evaluation, commits, and pushes must happen on the remote server in `~/tianchi_car`.
- The local repository is only for viewing or pulling; do not use it for implementation or training.
- The `/goal` execution must read `goal_plan.md`, update `goal_tasks.md`, and append normal experiment results to the root Markdown record file identified by UTF-8 byte prefix `23 20 e8 ae b0 e5 bd 95`; do not hard-code the displayed filename.
- Do not recreate planning files during execution. Only update the existing planning and task files as work progresses.
- Before every experiment, check `git status --short --branch`, current `HEAD`, `user_data/metrics.json`, LightGBM cache files, and whether any training process is already running.
- Before every experiment, back up `user_data/metrics.json` and `prediction_result/predictions.csv` into `user_data/experiment_backups/` with timestamped filenames.
- Each experiment round must change exactly one clear factor unless the change is documentation or artifact recovery.
- Official decision metric is only `blend_oof_mae` from `user_data/metrics.json`.
- Keep a result only if `blend_oof_mae` is strictly lower than the current best stable baseline.
- If a result does not improve the baseline, keep the experiment record, revert code to the previous stable commit, restore backed-up metrics and prediction files, then commit only the failed experiment documentation if needed.
- After every kept experiment, commit and push to `origin/main`, then update the current best commit, score, and next recommendation in `goal_tasks.md`.

## Strategy For This Goal

The latest gains came from strengthening LightGBM, refining blend resolution, and changing CatBoost depth from `8` to `9`. The next goal should build on that state without reverting to broad or noisy experimentation. Prioritize work in this order:

1. CatBoost bounded structure tuning around the current best candidate, one parameter at a time.
2. One small additional feature family that is tightly connected to already successful `v_*` statistics or age/power usage patterns.
3. LightGBM follow-up tuning only if the branch still shows meaningful headroom after the current refresh.
4. Blend refinement only after a branch meaningfully changes again; do not spend rounds on blend-only tweaks without upstream movement.
5. Do not resume pure iteration growth first unless multiple other directions fail and runtime tradeoff is explicitly justified by results.

## Stability And Validation Constraints

- Treat local OOF MAE as the official keep/rollback metric.
- If two consecutive experiments in the same direction fail, switch direction.
- Long runs are acceptable if the process is active and system usage indicates normal training, but do not start duplicate training processes.
- If a feature experiment affects LightGBM inputs, refresh the LightGBM cache through the existing pipeline before evaluating the blended result.
- Do not delete failed experiment records from the root Markdown record file identified by UTF-8 byte prefix `23 20 e8 ae b0 e5 bd 95`; do not hard-code the displayed filename, and do not delete failure records from `goal_tasks.md`.
- Do not rely on leaderboard assumptions. Preserve local experiment discipline.

## Completion Criteria

The goal is complete only when all of the following are true:

- `blend_oof_mae <= 481.2` in `user_data/metrics.json`.
- The improving code and experiment record are committed and pushed to `origin/main`.
- `goal_tasks.md` records the final best commit, score, core change, and next recommendation.
- The server working tree is clean.

If progress stalls before reaching the threshold, summarize the plateau only after at least two different directions fail, for example CatBoost structure tuning and lightweight feature extension, or lightweight feature extension and LightGBM follow-up tuning.

