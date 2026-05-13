# Tianchi Used Car Price Prediction

## Overview

This project predicts used car prices for the Tianchi competition. The pipeline reads the raw competition files, performs preprocessing and feature engineering, trains baseline tree models, and writes predictions for the A leaderboard test set.

## Project Structure

```text
project
|-- README.md
|-- data
|-- user_data
|-- feature
|-- model
|-- prediction_result
|-- code
```

## Run

From the project root:

```powershell
E:\tianchi_car\.conda\python.exe code\main.py
```

The script will:

1. Read `data/used_car_train_20200313.csv`
2. Read `data/used_car_testA_20200313.csv`
3. Build features
4. Train LightGBM and CatBoost baseline models
5. Save `prediction_result/predictions.csv`

## Notes

- Raw Tianchi data files are not included in submissions.
- Intermediate files are stored under `user_data/`.
- The current baseline uses a 50/50 blend of LightGBM and CatBoost predictions.
- Input paths use relative project paths as required by the contest specification.
