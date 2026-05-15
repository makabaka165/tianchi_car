from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import KFold

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from feature.loader import load_space_separated_table
from feature.preprocess import (
    TARGET_STAT_AGGS,
    TARGET_STAT_GROUPS,
    add_fold_target_stat_features,
    add_frequency_features,
    build_features,
    fill_missing_values,
)
from model.train import train_catboost, train_lightgbm


DATA_DIR = ROOT / "data"
USER_DATA_DIR = ROOT / "user_data"
PREDICTION_DIR = ROOT / "prediction_result"

N_SPLITS = 5
RANDOM_STATE = 42
LGB_CACHE_PRED = USER_DATA_DIR / "lgb_cache_predictions.npz"
LGB_CACHE_META = USER_DATA_DIR / "lgb_cache_meta.json"
METRICS_PATH = USER_DATA_DIR / "metrics.json"
PREDICTIONS_PATH = PREDICTION_DIR / "predictions.csv"


@dataclass
class CachedArtifacts:
    oof_predictions: np.ndarray
    test_predictions: np.ndarray
    cv_scores: list[float]
    model_name: str
    feature_count: int | None = None


CATBOOST_CANDIDATES = {
    "cfg_b_baseline": {
        "model_name": "cfg_b_baseline",
        "iterations": 2400,
        "learning_rate": 0.03,
        "depth": 8,
        "l2_leaf_reg": 7.0,
        "od_wait": 120,
    },
    "cfg_c_regularized": {
        "model_name": "cfg_c_regularized",
        "iterations": 2600,
        "learning_rate": 0.03,
        "depth": 8,
        "l2_leaf_reg": 9.0,
        "od_wait": 140,
    },
}


def search_blend_weight(
    y_true: pd.Series,
    lightgbm_oof: np.ndarray,
    catboost_oof: np.ndarray,
) -> tuple[float, float]:
    best_weight = 0.0
    best_mae = float("inf")
    for cat_weight in np.linspace(0.0, 1.0, 51):
        blended = (1.0 - cat_weight) * lightgbm_oof + cat_weight * catboost_oof
        score = mean_absolute_error(y_true, blended)
        if score < best_mae:
            best_mae = score
            best_weight = float(cat_weight)
    return best_weight, best_mae


def load_lgb_cache() -> CachedArtifacts | None:
    if not LGB_CACHE_PRED.exists() or not LGB_CACHE_META.exists():
        return None
    arrays = np.load(LGB_CACHE_PRED)
    meta = json.loads(LGB_CACHE_META.read_text(encoding="utf-8"))
    return CachedArtifacts(
        oof_predictions=arrays["oof_predictions"],
        test_predictions=arrays["test_predictions"],
        cv_scores=meta["cv_scores"],
        model_name=meta["model_name"],
        feature_count=meta.get("feature_count"),
    )


def save_lgb_cache(artifacts) -> None:
    np.savez_compressed(
        LGB_CACHE_PRED,
        oof_predictions=artifacts.oof_predictions,
        test_predictions=artifacts.test_predictions,
    )
    LGB_CACHE_META.write_text(
        json.dumps(
            {
                "cv_scores": artifacts.cv_scores,
                "model_name": artifacts.model_name,
                "feature_count": artifacts.feature_count,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Tianchi used car experiment entrypoint.")
    parser.add_argument(
        "mode",
        nargs="?",
        default="catboost_only_sweep",
        choices=["prepare_lgb_cache", "catboost_only_sweep", "full"],
        help="Execution mode.",
    )
    parser.add_argument(
        "--candidate",
        action="append",
        choices=sorted(CATBOOST_CANDIDATES),
        help="Run only the specified CatBoost candidate. Repeat to run multiple candidates.",
    )
    return parser.parse_args()


def prepare_datasets(include_lgb_branch: bool, include_cat_branch: bool):
    USER_DATA_DIR.mkdir(exist_ok=True)
    PREDICTION_DIR.mkdir(exist_ok=True)

    train_raw = load_space_separated_table(DATA_DIR / "used_car_train_20200313.csv")
    test_raw = load_space_separated_table(DATA_DIR / "used_car_testA_20200313.csv")

    train_df = build_features(train_raw, is_train=True)
    test_df = build_features(test_raw, is_train=False)

    base_train_df, base_test_df = add_frequency_features(train_df, test_df)
    folds = list(KFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE).split(train_df))

    datasets = {
        "train_raw": train_raw,
        "test_raw": test_raw,
        "folds": folds,
        "y": base_train_df["price"].astype(float),
        "train_rows": int(base_train_df.shape[0]),
        "test_rows": int(base_test_df.shape[0]),
    }

    if include_lgb_branch:
        lgb_train_df, lgb_test_df = add_fold_target_stat_features(base_train_df, base_test_df, folds=folds)
        lgb_train_df, lgb_test_df = fill_missing_values(lgb_train_df, lgb_test_df)
        X_lgb = lgb_train_df.drop(columns=["price"])
        datasets["X_lgb"] = X_lgb
        datasets["X_lgb_test"] = lgb_test_df.copy()
        datasets["lgb_categorical_features"] = [
            col
            for col in X_lgb.columns
            if pd.api.types.is_string_dtype(X_lgb[col]) or str(X_lgb[col].dtype) == "category"
        ]

    if include_cat_branch:
        cat_train_df, cat_test_df = fill_missing_values(base_train_df.copy(), base_test_df.copy())
        X_cat = cat_train_df.drop(columns=["price"])
        datasets["X_cat"] = X_cat
        datasets["X_cat_test"] = cat_test_df.copy()
        datasets["cat_categorical_features"] = [
            col
            for col in X_cat.columns
            if pd.api.types.is_string_dtype(X_cat[col]) or str(X_cat[col].dtype) == "category"
        ]

    return datasets


def resolve_catboost_candidates(selected_names: list[str] | None) -> list[dict]:
    if not selected_names:
        return [CATBOOST_CANDIDATES[name].copy() for name in CATBOOST_CANDIDATES]
    return [CATBOOST_CANDIDATES[name].copy() for name in selected_names]


def evaluate_catboost_candidates(
    y: pd.Series,
    lgb_artifacts,
    X_cat: pd.DataFrame,
    X_cat_test: pd.DataFrame,
    cat_categorical_features: list[str],
    folds,
    candidates: list[dict],
):
    results = []
    best = None

    for params in candidates:
        artifacts = train_catboost(
            X_cat.copy(),
            y,
            X_cat_test.copy(),
            cat_categorical_features,
            use_log_target=False,
            folds=folds,
            model_params=params,
        )
        cat_mae = mean_absolute_error(y, artifacts.oof_predictions)
        best_cat_weight, blend_oof_mae = search_blend_weight(
            y, lgb_artifacts.oof_predictions, artifacts.oof_predictions
        )
        result = {
            "params": params,
            "artifacts": artifacts,
            "catboost_oof_mae": cat_mae,
            "best_catboost_weight": best_cat_weight,
            "best_lightgbm_weight": 1.0 - best_cat_weight,
            "blend_oof_mae": blend_oof_mae,
        }
        results.append(result)
        if best is None or blend_oof_mae < best["blend_oof_mae"]:
            best = result

    return best, results


def build_lgb_cache(datasets) -> CachedArtifacts:
    lgb_artifacts = train_lightgbm(
        datasets["X_lgb"].copy(),
        datasets["y"],
        datasets["X_lgb_test"].copy(),
        datasets["lgb_categorical_features"],
        use_log_target=True,
        folds=datasets["folds"],
    )
    lgb_artifacts.feature_count = int(datasets["X_lgb"].shape[1])
    save_lgb_cache(lgb_artifacts)
    return lgb_artifacts


def run_catboost_sweep(
    mode: str,
    datasets,
    lgb_artifacts: CachedArtifacts,
    lgb_cache_used: bool,
    candidates: list[dict],
) -> dict:
    if lgb_artifacts.feature_count is None:
        lgb_artifacts.feature_count = int(
            datasets["X_cat"].shape[1] + len(TARGET_STAT_GROUPS) * len(TARGET_STAT_AGGS)
        )

    best_cat_result, cat_results = evaluate_catboost_candidates(
        y=datasets["y"],
        lgb_artifacts=lgb_artifacts,
        X_cat=datasets["X_cat"],
        X_cat_test=datasets["X_cat_test"],
        cat_categorical_features=datasets["cat_categorical_features"],
        folds=datasets["folds"],
        candidates=candidates,
    )
    cat_artifacts = best_cat_result["artifacts"]

    lgb_mae = mean_absolute_error(datasets["y"], lgb_artifacts.oof_predictions)
    cat_mae = best_cat_result["catboost_oof_mae"]
    best_cat_weight = best_cat_result["best_catboost_weight"]
    best_lgb_weight = best_cat_result["best_lightgbm_weight"]
    blend_oof_mae = best_cat_result["blend_oof_mae"]

    blended_test = best_lgb_weight * lgb_artifacts.test_predictions + best_cat_weight * cat_artifacts.test_predictions
    blended_test = np.clip(blended_test, 0, None)

    predictions = pd.DataFrame(
        {
            "SaleID": pd.to_numeric(datasets["test_raw"]["SaleID"], errors="coerce").astype("Int64"),
            "price": blended_test,
        }
    )
    predictions["price"] = predictions["price"].round(6)

    metrics = {
        "mode": mode,
        "lightgbm_cv_scores": lgb_artifacts.cv_scores,
        "lightgbm_oof_mae": lgb_mae,
        "lightgbm_model_name": lgb_artifacts.model_name,
        "lightgbm_cache_used": lgb_cache_used,
        "catboost_cv_scores": cat_artifacts.cv_scores,
        "catboost_oof_mae": cat_mae,
        "catboost_model_name": cat_artifacts.model_name,
        "selected_catboost_params": best_cat_result["params"],
        "best_lightgbm_weight": best_lgb_weight,
        "best_catboost_weight": best_cat_weight,
        "blend_oof_mae": blend_oof_mae,
        "lightgbm_feature_count": lgb_artifacts.feature_count,
        "catboost_feature_count": int(datasets["X_cat"].shape[1]),
        "catboost_candidates": [
            {
                "model_name": result["artifacts"].model_name,
                "catboost_oof_mae": result["catboost_oof_mae"],
                "best_lightgbm_weight": result["best_lightgbm_weight"],
                "best_catboost_weight": result["best_catboost_weight"],
                "blend_oof_mae": result["blend_oof_mae"],
                "params": result["params"],
            }
            for result in cat_results
        ],
        "train_rows": datasets["train_rows"],
        "test_rows": datasets["test_rows"],
    }

    metrics_tmp = METRICS_PATH.with_suffix(".json.tmp")
    predictions_tmp = PREDICTIONS_PATH.with_suffix(".csv.tmp")
    metrics_tmp.write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
    predictions.to_csv(predictions_tmp, index=False)
    metrics_tmp.replace(METRICS_PATH)
    predictions_tmp.replace(PREDICTIONS_PATH)
    return metrics


def main() -> None:
    args = parse_args()
    candidates = resolve_catboost_candidates(args.candidate)
    include_lgb_branch = args.mode == "prepare_lgb_cache"

    if args.mode == "full":
        include_lgb_branch = load_lgb_cache() is None

    datasets = prepare_datasets(
        include_lgb_branch=include_lgb_branch,
        include_cat_branch=args.mode != "prepare_lgb_cache",
    )

    if args.mode == "prepare_lgb_cache":
        build_lgb_cache(datasets)
        print(f"Prepared LightGBM cache at: {LGB_CACHE_PRED}")
        return

    lgb_cache = load_lgb_cache()
    if args.mode == "catboost_only_sweep":
        if lgb_cache is None:
            raise RuntimeError("LightGBM cache is missing. Run `python code/main.py prepare_lgb_cache` first.")
        lgb_artifacts = lgb_cache
        lgb_cache_used = True
    else:
        if lgb_cache is None:
            lgb_artifacts = build_lgb_cache(datasets)
            lgb_cache_used = False
        else:
            lgb_artifacts = lgb_cache
            lgb_cache_used = True

    metrics = run_catboost_sweep(args.mode, datasets, lgb_artifacts, lgb_cache_used, candidates)
    print(json.dumps(metrics, ensure_ascii=False, indent=2))
    print(f"Saved predictions to: {PREDICTIONS_PATH}")


if __name__ == "__main__":
    main()
