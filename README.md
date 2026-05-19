# Tianchi Used Car Price Prediction

## Overview

This repository contains a competition solution for the Tianchi used car price prediction task.

The goal is to predict `price` for each vehicle in the test set using:

- structured vehicle attributes
- anonymized `v_0 ~ v_14` features
- date information from registration and listing time

The project is built around a disciplined offline validation workflow:

- fixed 5-fold OOF evaluation
- single-change experiments
- keep or rollback based on local `blend_oof_mae`
- experiment logs preserved for both successful and failed trials

## Current Best Result

Current final stable result on the server:

- Stable code commit: `53c36fa`
- Final documentation sync commit: `68d73b2`
- Experiment note: `task8_cat_iter4600_cfg_w_eval`
- LightGBM OOF MAE: `526.6174419083302`
- CatBoost OOF MAE: `488.8264628108603`
- Blend OOF MAE: `480.87277472593655`
- Best blend weights:
  LightGBM `0.282`
  CatBoost `0.718`

## Main Idea

The final solution is not a single-model brute-force approach.
It is a two-branch tree-model ensemble:

1. A LightGBM branch enhanced by fold-safe target statistics and categorical frequency features.
2. A CatBoost branch built on a cleaner feature table with strong categorical handling.
3. An OOF-based weighted blend to combine both branches.

The final performance came from combining:

- stable feature engineering
- branch specialization
- cache-first experimentation
- narrow single-variable tuning
- refined blend search

## Pipeline Summary

### 1. Feature Engineering

Implemented mainly in `feature/preprocess.py`.

Core feature groups:

- parsed and repaired date fields
- car age features
- clipped power and mileage features
- log-transformed numeric features
- interaction features such as power vs. mileage and age
- aggregated anonymous `v_*` statistics:
  `v_sum`, `v_mean`, `v_std`, `v_min`, `v_max`, `v_range`
- categorical combination features:
  `brand_model`, `brand_bodyType`
- missing value count
- categorical frequency encoding

### 2. LightGBM Branch

Implemented in `model/train.py`.

Characteristics:

- trains on `log1p(price)`
- uses fold-safe target statistics
- uses cached OOF/test predictions for fast follow-up experiments

Final kept LightGBM configuration:

- `n_estimators=4000`
- `learning_rate=0.03`
- `num_leaves=127`
- `subsample=0.8`
- `colsample_bytree=0.8`
- `reg_alpha=0.2`
- `reg_lambda=0.2`

### 3. CatBoost Branch

Also implemented in `model/train.py`, with candidate orchestration in `code/main.py`.

Characteristics:

- predicts raw `price`
- uses a relatively clean feature branch
- tuned through controlled single-candidate experiments

Final kept CatBoost candidate:

- `cfg_w_depth_9_iter4600`
- `iterations=4600`
- `learning_rate=0.03`
- `depth=9`
- `l2_leaf_reg=9.0`
- `od_wait=140`

### 4. OOF Blend

Implemented in `code/main.py`.

Blend weights are searched on OOF predictions instead of being fixed manually.

Final kept refinement:

- search resolution upgraded from `0.01` to `0.001`

This gave a small but real gain in the late optimization stage.

## Project Structure

```text
project
|-- README.md
|-- 比赛总结.md
|-- goal_plan.md
|-- goal_tasks.md
|-- data
|-- user_data
|-- feature
|-- model
|-- prediction_result
|-- code
```

Important directories:

- `feature/`
  preprocessing and feature engineering

- `model/`
  model training logic

- `code/`
  training entrypoint and experiment orchestration

- `user_data/`
  caches, metrics, intermediate artifacts

- `prediction_result/`
  final submission file

## Entrypoints

Main training entry:

```bash
python code/main.py
```

Useful modes:

```bash
python code/main.py prepare_lgb_cache
python code/main.py catboost_only_sweep --candidate cfg_w_depth_9_iter4600
python code/main.py full
```

## Output Artifacts

Main runtime artifacts:

- `user_data/metrics.json`
- `user_data/lgb_cache_predictions.npz`
- `user_data/lgb_cache_meta.json`
- `prediction_result/predictions.csv`

Prediction file format:

```csv
SaleID,price
150000,37010.1499
150001,333.590328
```

## Experiment Discipline

This repository was optimized using a strict experiment protocol:

- one round, one clear change
- same 5-fold validation for every round
- compare only by local `blend_oof_mae`
- keep if strictly better
- rollback if equal or worse
- preserve every experiment record

This discipline is a major reason the final result is trustworthy and reproducible.

## Reading Guide

If you want to understand the solution quickly, read files in this order:

1. `feature/preprocess.py`
2. `model/train.py`
3. `code/main.py`
4. `比赛总结.md`

## Notes

- Raw competition data files are not included here.
- Relative paths are used for data access and outputs.
- The final stage summary and algorithm review are documented in `比赛总结.md`.
