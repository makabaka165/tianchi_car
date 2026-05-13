from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from catboost import CatBoostRegressor
from lightgbm import LGBMRegressor, early_stopping, log_evaluation
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import KFold


@dataclass
class TrainingArtifacts:
    oof_predictions: np.ndarray
    test_predictions: np.ndarray
    cv_scores: list[float]
    model_name: str


def train_lightgbm(
    X: pd.DataFrame,
    y: pd.Series,
    X_test: pd.DataFrame,
    categorical_features: list[str],
    n_splits: int = 5,
    random_state: int = 42,
    use_log_target: bool = False,
    folds: list[tuple[np.ndarray, np.ndarray]] | None = None,
) -> TrainingArtifacts:
    fold_indices = folds or list(KFold(n_splits=n_splits, shuffle=True, random_state=random_state).split(X))
    oof = np.zeros(len(X), dtype=np.float64)
    test_preds = np.zeros(len(X_test), dtype=np.float64)
    scores: list[float] = []

    X_test_work = X_test.copy()

    for fold, (tr_idx, va_idx) in enumerate(fold_indices, start=1):
        X_tr, X_va = X.iloc[tr_idx].copy(), X.iloc[va_idx].copy()
        y_tr, y_va = y.iloc[tr_idx], y.iloc[va_idx]
        y_tr_fit = np.log1p(y_tr) if use_log_target else y_tr

        for col in categorical_features:
            if col in X_tr.columns:
                X_tr[col] = X_tr[col].astype("category")
                X_va[col] = X_va[col].astype("category")
                X_test_work[col] = X_test_work[col].astype("category")

        model = LGBMRegressor(
            objective="mae",
            n_estimators=4000,
            learning_rate=0.03,
            num_leaves=63,
            max_depth=-1,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_alpha=0.1,
            reg_lambda=0.1,
            random_state=random_state + fold,
            n_jobs=-1,
            force_col_wise=True,
        )
        model.fit(
            X_tr,
            y_tr_fit,
            eval_set=[(X_va, np.log1p(y_va) if use_log_target else y_va)],
            eval_metric="l1" if not use_log_target else "l2",
            categorical_feature=[c for c in categorical_features if c in X_tr.columns],
            callbacks=[early_stopping(200, verbose=False), log_evaluation(period=0)],
        )
        pred_va = model.predict(X_va)
        pred_test = model.predict(X_test_work)
        if use_log_target:
            pred_va = np.expm1(pred_va)
            pred_test = np.expm1(pred_test)
        score = mean_absolute_error(y_va, pred_va)
        scores.append(score)
        oof[va_idx] = pred_va
        test_preds += pred_test / len(fold_indices)

    suffix = "_log1p" if use_log_target else ""
    return TrainingArtifacts(oof, test_preds, scores, f"lightgbm{suffix}")


def train_catboost(
    X: pd.DataFrame,
    y: pd.Series,
    X_test: pd.DataFrame,
    categorical_features: list[str],
    n_splits: int = 5,
    random_state: int = 42,
    use_log_target: bool = False,
    folds: list[tuple[np.ndarray, np.ndarray]] | None = None,
) -> TrainingArtifacts:
    fold_indices = folds or list(KFold(n_splits=n_splits, shuffle=True, random_state=random_state).split(X))
    oof = np.zeros(len(X), dtype=np.float64)
    test_preds = np.zeros(len(X_test), dtype=np.float64)
    scores: list[float] = []

    cat_idx = [X.columns.get_loc(col) for col in categorical_features if col in X.columns]

    for fold, (tr_idx, va_idx) in enumerate(fold_indices, start=1):
        X_tr, X_va = X.iloc[tr_idx].copy(), X.iloc[va_idx].copy()
        y_tr, y_va = y.iloc[tr_idx], y.iloc[va_idx]
        y_tr_fit = np.log1p(y_tr) if use_log_target else y_tr

        model = CatBoostRegressor(
            loss_function="RMSE" if use_log_target else "MAE",
            eval_metric="RMSE" if use_log_target else "MAE",
            iterations=2200,
            learning_rate=0.03,
            depth=7,
            l2_leaf_reg=5.0,
            random_seed=random_state + fold,
            verbose=False,
            od_type="Iter",
            od_wait=100,
        )
        model.fit(
            X_tr,
            y_tr_fit,
            cat_features=cat_idx,
            eval_set=(X_va, np.log1p(y_va) if use_log_target else y_va),
            use_best_model=True,
        )
        pred_va = model.predict(X_va)
        pred_test = model.predict(X_test)
        if use_log_target:
            pred_va = np.expm1(pred_va)
            pred_test = np.expm1(pred_test)
        score = mean_absolute_error(y_va, pred_va)
        scores.append(score)
        oof[va_idx] = pred_va
        test_preds += pred_test / len(fold_indices)

    suffix = "_log1p" if use_log_target else ""
    return TrainingArtifacts(oof, test_preds, scores, f"catboost{suffix}")
