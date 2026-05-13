from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import KFold

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from feature.loader import load_space_separated_table
from feature.preprocess import add_fold_target_stat_features, add_frequency_features, build_features, fill_missing_values
from model.train import train_catboost, train_lightgbm


DATA_DIR = ROOT / "data"
USER_DATA_DIR = ROOT / "user_data"
PREDICTION_DIR = ROOT / "prediction_result"

N_SPLITS = 5
RANDOM_STATE = 42


def search_blend_weight(
    y_true: pd.Series,
    lightgbm_oof: np.ndarray,
    catboost_oof: np.ndarray,
) -> tuple[float, float]:
    best_weight = 0.0
    best_mae = float("inf")
    for cat_weight in np.linspace(0.0, 1.0, 21):
        blended = (1.0 - cat_weight) * lightgbm_oof + cat_weight * catboost_oof
        score = mean_absolute_error(y_true, blended)
        if score < best_mae:
            best_mae = score
            best_weight = float(cat_weight)
    return best_weight, best_mae


def main() -> None:
    USER_DATA_DIR.mkdir(exist_ok=True)
    PREDICTION_DIR.mkdir(exist_ok=True)

    train_raw = load_space_separated_table(DATA_DIR / "used_car_train_20200313.csv")
    test_raw = load_space_separated_table(DATA_DIR / "used_car_testA_20200313.csv")

    train_df = build_features(train_raw, is_train=True)
    test_df = build_features(test_raw, is_train=False)

    train_df, test_df = add_frequency_features(train_df, test_df)
    folds = list(KFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE).split(train_df))
    train_df, test_df = add_fold_target_stat_features(train_df, test_df, folds=folds)
    train_df, test_df = fill_missing_values(train_df, test_df)

    y = train_df["price"].astype(float)
    X = train_df.drop(columns=["price"])
    X_test = test_df.copy()

    categorical_features = [
        col
        for col in X.columns
        if pd.api.types.is_string_dtype(X[col]) or str(X[col].dtype) == "category"
    ]

    lgb_artifacts = train_lightgbm(
        X.copy(),
        y,
        X_test.copy(),
        categorical_features,
        use_log_target=True,
        folds=folds,
    )
    cat_artifacts = train_catboost(
        X.copy(),
        y,
        X_test.copy(),
        categorical_features,
        use_log_target=True,
        folds=folds,
    )

    lgb_mae = mean_absolute_error(y, lgb_artifacts.oof_predictions)
    cat_mae = mean_absolute_error(y, cat_artifacts.oof_predictions)
    best_cat_weight, blend_oof_mae = search_blend_weight(y, lgb_artifacts.oof_predictions, cat_artifacts.oof_predictions)
    best_lgb_weight = 1.0 - best_cat_weight

    blended_test = best_lgb_weight * lgb_artifacts.test_predictions + best_cat_weight * cat_artifacts.test_predictions
    blended_test = np.clip(blended_test, 0, None)

    predictions = pd.DataFrame(
        {
            "SaleID": pd.to_numeric(test_raw["SaleID"], errors="coerce").astype("Int64"),
            "price": blended_test,
        }
    )
    predictions["price"] = predictions["price"].round(6)
    predictions.to_csv(PREDICTION_DIR / "predictions.csv", index=False)

    metrics = {
        "lightgbm_cv_scores": lgb_artifacts.cv_scores,
        "lightgbm_oof_mae": lgb_mae,
        "lightgbm_model_name": lgb_artifacts.model_name,
        "catboost_cv_scores": cat_artifacts.cv_scores,
        "catboost_oof_mae": cat_mae,
        "catboost_model_name": cat_artifacts.model_name,
        "best_lightgbm_weight": best_lgb_weight,
        "best_catboost_weight": best_cat_weight,
        "blend_oof_mae": blend_oof_mae,
        "feature_count": int(X.shape[1]),
        "train_rows": int(X.shape[0]),
        "test_rows": int(X_test.shape[0]),
    }
    (USER_DATA_DIR / "metrics.json").write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print(json.dumps(metrics, ensure_ascii=False, indent=2))
    print(f"Saved predictions to: {PREDICTION_DIR / 'predictions.csv'}")


if __name__ == "__main__":
    main()
