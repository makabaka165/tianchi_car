# tianchi_car Goal Plan

## Goal

Use the remote server repository `~/tianchi_car` as the only execution environment and continue optimizing the used-car price prediction pipeline from the current stable baseline:

- Target metric: `user_data/metrics.json -> blend_oof_mae`
- Current stable baseline: commit `62dffdb`, experiment `task4_vstats_cfg_q_eval`
- Current stable score: `Blend OOF MAE = 483.96430195171246`
- New goal threshold: `Blend OOF MAE <= 482.8`

This target is intentionally tighter than the previous one but still realistic. It should require multiple experiment rounds across LightGBM recovery, lightweight features, CatBoost structure tuning, and blend refinement. Do not declare success after one or two easy tweaks unless the score truly reaches `<= 482.8` and is committed and pushed.

## Execution Rules

- All code changes, training, evaluation, commits, and pushes must happen on the remote server in `~/tianchi_car`.
- The local repository is only for viewing or pulling; do not use it for implementation or training.
- The `/goal` execution should not recreate planning files. It must read `goal_plan.md`, update `goal_tasks.md`, and append normal experiment results to the root Markdown record file identified by UTF-8 byte prefix `23 20 e8 ae b0 e5 bd 95`; do not hard-code the displayed filename.
- Before every experiment, check `git status --short --branch`, current `HEAD`, `user_data/metrics.json`, LightGBM cache files, and whether any training process is already running.
- Before every experiment, back up `user_data/metrics.json` and `prediction_result/predictions.csv` into `user_data/experiment_backups/` with timestamped filenames.
- Each experiment round must change exactly one clear factor unless the change is documentation or artifact recovery.
- Official decision metric is only `blend_oof_mae` from `user_data/metrics.json`.
- Keep a result only if `blend_oof_mae` is strictly lower than the current best stable baseline.
- If a result does not improve the baseline, keep the experiment record, revert code to the previous stable commit, restore backed-up metrics and prediction files, then commit only the failed experiment documentation if needed.
- After every kept experiment, commit and push to `origin/main`, then update the current best commit, score, and next recommendation in `goal_tasks.md`.

## Strategy For This Goal

The previous goal was solved mainly by improving CatBoost and adding `v_0..v_14` aggregate features. The next goal should not rely primarily on further iteration growth. Prioritize work in this order:

1. Strengthen the LightGBM branch using the newly proven `v_*` aggregate features and small LightGBM parameter adjustments.
2. Re-evaluate blend behavior after LightGBM changes; keep only if the blended result improves.
3. Explore one low-risk additional feature family at a time, especially interactions that build on `v_sum`, `v_mean`, `v_std`, `v_range`, `power`, `kilometer`, and age features.
4. Only after the above, test CatBoost structure parameters around the current best candidate, such as `depth`, `l2_leaf_reg`, or other bounded controls.
5. Do not resume pure CatBoost iteration scaling first unless new evidence suggests the gain curve is still strong enough to justify the runtime.

## Stability And Validation Constraints

- Treat local OOF MAE as the official keep/rollback metric.
- Avoid repeated overfitting to one narrow direction. If two consecutive experiments in the same direction fail, switch direction.
- Long runs are acceptable if the process is active and system usage indicates normal training, but do not start duplicate training processes.
- If a feature experiment affects LightGBM inputs, refresh the LightGBM cache through the existing pipeline before evaluating the blended result.
- Do not delete failed experiment records from the root Markdown record file identified by UTF-8 byte prefix `23 20 e8 ae b0 e5 bd 95`; do not hard-code the displayed filename, and do not delete failure records from `goal_tasks.md`.
- Do not use leaderboard assumptions. Preserve local experiment discipline.

## Completion Criteria

The goal is complete only when all of the following are true:

- `blend_oof_mae <= 482.8` in `user_data/metrics.json`.
- The improving code and experiment record are committed and pushed to `origin/main`.
- `goal_tasks.md` records the final best commit, score, core change, and next recommendation.
- The server working tree is clean.

If progress stalls before reaching the threshold, summarize the plateau only after at least two different directions fail, for example LightGBM improvement and feature extension, or feature extension and CatBoost structure tuning.

