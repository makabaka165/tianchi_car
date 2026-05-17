from __future__ import annotations

import numpy as np
import pandas as pd


NUMERIC_COLUMNS = [
    "SaleID",
    "name",
    "model",
    "brand",
    "bodyType",
    "fuelType",
    "gearbox",
    "power",
    "kilometer",
    "notRepairedDamage",
    "regionCode",
    "seller",
    "offerType",
    "price",
    *[f"v_{i}" for i in range(15)],
]

CATEGORICAL_COLUMNS = [
    "name",
    "model",
    "brand",
    "bodyType",
    "fuelType",
    "gearbox",
    "notRepairedDamage",
    "regionCode",
    "seller",
]

TARGET_STAT_GROUPS = [
    "brand",
    "model",
    "bodyType",
    "fuelType",
    "gearbox",
    "regionCode",
    "brand_model",
    "brand_bodyType",
]

TARGET_STAT_AGGS = ["mean", "median", "min", "max", "count"]


def _parse_date_column(series: pd.Series) -> tuple[pd.Series, pd.Series]:
    raw = series.astype("string")
    dt = pd.to_datetime(raw, format="%Y%m%d", errors="coerce")
    invalid = dt.isna().astype("int8")
    year = pd.to_numeric(raw.str[:4], errors="coerce")
    month = pd.to_numeric(raw.str[4:6], errors="coerce")
    day = pd.to_numeric(raw.str[6:8], errors="coerce")

    invalid_month_mask = month.notna() & ((month < 1) | (month > 12))
    month_fixed = month.where(~invalid_month_mask, 1)
    day_fixed = day.clip(lower=1, upper=28)
    repaired = pd.to_datetime(
        pd.DataFrame({"year": year, "month": month_fixed, "day": day_fixed}),
        errors="coerce",
    )
    dt = dt.fillna(repaired)
    return dt, invalid


def convert_types(df: pd.DataFrame, is_train: bool) -> pd.DataFrame:
    out = df.copy()
    for col in NUMERIC_COLUMNS:
        if col not in out.columns:
            continue
        out[col] = pd.to_numeric(out[col], errors="coerce")
    if not is_train and "price" in out.columns:
        out = out.drop(columns=["price"])
    return out


def build_features(df: pd.DataFrame, is_train: bool) -> pd.DataFrame:
    out = convert_types(df, is_train=is_train)

    reg_dt, reg_invalid = _parse_date_column(out["regDate"])
    create_dt, create_invalid = _parse_date_column(out["creatDate"])
    out["reg_year"] = reg_dt.dt.year
    out["reg_month"] = reg_dt.dt.month
    out["reg_day"] = reg_dt.dt.day
    out["creat_year"] = create_dt.dt.year
    out["creat_month"] = create_dt.dt.month
    out["creat_day"] = create_dt.dt.day
    out["creat_dayofweek"] = create_dt.dt.dayofweek
    out["reg_invalid"] = reg_invalid
    out["creat_invalid"] = create_invalid

    car_age_days = (create_dt - reg_dt).dt.days
    out["car_age_days"] = car_age_days
    out["car_age_years"] = car_age_days / 365.25
    out["date_gap_abs"] = car_age_days.abs()
    out["date_gap_negative"] = (car_age_days < 0).astype("int8")

    out["power"] = out["power"].clip(lower=0, upper=600)
    out["power_is_zero"] = (out["power"] == 0).astype("int8")
    out["power_log1p"] = np.log1p(out["power"].clip(lower=0))

    out["kilometer"] = out["kilometer"].clip(lower=0)
    out["power_per_km"] = out["power"] / (out["kilometer"] + 1.0)
    out["kilometer_log1p"] = np.log1p(out["kilometer"])
    out["power_x_kilometer"] = out["power"] * out["kilometer"]
    out["power_per_age"] = out["power"] / (out["car_age_days"].abs() + 1.0)

    out["brand_model"] = (
        out["brand"].fillna(-1).astype("Int64").astype("string")
        + "_"
        + out["model"].fillna(-1).astype("Int64").astype("string")
    )
    out["brand_bodyType"] = (
        out["brand"].fillna(-1).astype("Int64").astype("string")
        + "_"
        + out["bodyType"].fillna(-1).astype("Int64").astype("string")
    )

    # Drop raw date strings after deriving stable features.
    out = out.drop(columns=["regDate", "creatDate"])

    drop_cols = ["offerType"]
    for col in drop_cols:
        if col in out.columns:
            out = out.drop(columns=[col])

    numeric_cols = [
        col
        for col in out.columns
        if pd.api.types.is_numeric_dtype(out[col]) and col != "price"
    ]
    out["missing_count"] = out[numeric_cols].isna().sum(axis=1).astype("float32")

    for col in CATEGORICAL_COLUMNS + ["brand_model", "brand_bodyType"]:
        if col in out.columns:
            out[col] = out[col].astype("string").fillna("missing")

    return out


def add_frequency_features(train_df: pd.DataFrame, test_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    cat_cols = [
        "name",
        "model",
        "brand",
        "bodyType",
        "fuelType",
        "gearbox",
        "notRepairedDamage",
        "regionCode",
        "brand_model",
        "brand_bodyType",
    ]
    train_out = train_df.copy()
    test_out = test_df.copy()
    full = pd.concat([train_out[cat_cols], test_out[cat_cols]], axis=0, ignore_index=True)

    for col in cat_cols:
        freq = full[col].value_counts(dropna=False)
        train_out[f"{col}_freq"] = train_out[col].map(freq).astype("float32")
        test_out[f"{col}_freq"] = test_out[col].map(freq).astype("float32")

    return train_out, test_out


def add_target_stat_features(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    target_col: str = "price",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    train_out = train_df.copy()
    test_out = test_df.copy()
    group_cols = TARGET_STAT_GROUPS

    for col in group_cols:
        stats = (
            train_out.groupby(col, dropna=False)[target_col]
            .agg(TARGET_STAT_AGGS)
            .rename(
                columns={
                    "mean": f"{col}_price_mean",
                    "median": f"{col}_price_median",
                    "min": f"{col}_price_min",
                    "max": f"{col}_price_max",
                    "count": f"{col}_price_count",
                }
            )
        )

        train_out = train_out.merge(stats, on=col, how="left")
        test_out = test_out.merge(stats, on=col, how="left")

    return train_out, test_out


def add_fold_target_stat_features(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    folds: list[tuple[np.ndarray, np.ndarray]],
    target_col: str = "price",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    train_out = train_df.copy()
    test_out = test_df.copy()

    for col in TARGET_STAT_GROUPS:
        feature_names = [f"{col}_price_{agg}" for agg in TARGET_STAT_AGGS]
        train_block = pd.DataFrame(index=train_out.index, columns=feature_names, dtype="float64")

        for tr_idx, va_idx in folds:
            source = train_out.iloc[tr_idx]
            valid = train_out.iloc[va_idx]
            stats = source.groupby(col, dropna=False)[target_col].agg(TARGET_STAT_AGGS)
            for agg in TARGET_STAT_AGGS:
                feature_name = f"{col}_price_{agg}"
                train_block.loc[train_block.index[va_idx], feature_name] = valid[col].map(stats[agg]).to_numpy()

        full_stats = train_out.groupby(col, dropna=False)[target_col].agg(TARGET_STAT_AGGS)
        test_block = pd.DataFrame(index=test_out.index, columns=feature_names, dtype="float64")
        for agg in TARGET_STAT_AGGS:
            feature_name = f"{col}_price_{agg}"
            test_block[feature_name] = test_out[col].map(full_stats[agg]).to_numpy()

        train_out[feature_names] = train_block
        test_out[feature_names] = test_block

    return train_out, test_out


def fill_missing_values(train_df: pd.DataFrame, test_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    train_out = train_df.copy()
    test_out = test_df.copy()

    for col in train_out.columns:
        if col == "price":
            continue
        if pd.api.types.is_string_dtype(train_out[col]):
            train_out[col] = train_out[col].fillna("missing")
            test_out[col] = test_out[col].fillna("missing")
        else:
            fill_value = train_out[col].median()
            if pd.isna(fill_value):
                fill_value = -1
            train_out[col] = train_out[col].fillna(fill_value)
            test_out[col] = test_out[col].fillna(fill_value)

    return train_out, test_out
