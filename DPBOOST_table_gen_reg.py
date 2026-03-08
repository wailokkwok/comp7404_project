#!/usr/bin/env python3
"""
DPBoost regression experiment — baseline (large budget) + DPBoost_2level
Parameter template taken from run_exp.py which is known to work.
"""

import numpy as np
import pandas as pd
import os
import warnings
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.metrics import mean_squared_error, r2_score
import lightgbm as lgb

warnings.filterwarnings("ignore")

# ======================================================================
#  CONFIGURATION
# ======================================================================

BASE_PATH = "/content"

N_SPLITS           = 25
EPSILONS           = [0.5, 1.0, 2.0, 4.0, 8.0]
N_TREES            = 50
INNER_BOOST_ROUND  = 50
BALANCE_PARTITION  = 0
BASELINE_BUDGET    = 1000.0        # large ε  ≈  non-private

# ======================================================================
#  DATA LOADING
# ======================================================================

def _encode_dataframe(df):
    """Ordinal-encode all object/category columns, return float array."""
    df = df.copy()
    for col in df.columns:
        if df[col].dtype == object or df[col].dtype.name == "category":
            df[col] = df[col].astype(str).str.strip()
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col])
        else:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df.values.astype(np.float64)


def load_cal_housing():
    path = os.path.join(BASE_PATH, "cal_housing.csv")
    df = pd.read_csv(path)
    target_col = df.columns[-1]
    y = df[target_col].values.astype(float)
    X_df = df.drop(columns=[target_col])
    X = _encode_dataframe(X_df)
    valid = ~(np.isnan(X).any(axis=1) | np.isnan(y))
    X, y = X[valid], y[valid]
    return X, y, "cal-housing"


def load_elevators():
    path = os.path.join(BASE_PATH, "elevators.csv")
    df = pd.read_csv(path)
    target_col = df.columns[-1]
    y = df[target_col].values.astype(float)
    X = df.drop(columns=[target_col]).values.astype(float)
    valid = ~(np.isnan(X).any(axis=1) | np.isnan(y))
    X, y = X[valid], y[valid]
    return X, y, "elevators"


def load_pol():
    path = os.path.join(BASE_PATH, "pol.csv")
    df = pd.read_csv(path)
    target_col = df.columns[-1]
    y = df[target_col].values.astype(float)
    X = df.drop(columns=[target_col]).values.astype(float)
    valid = ~(np.isnan(X).any(axis=1) | np.isnan(y))
    X, y = X[valid], y[valid]
    return X, y, "pol"


ALL_LOADERS = [
    load_cal_housing,
    load_elevators,
    load_pol,
]

# ======================================================================
#  HELPER — build DPBoost param dict  (mirrors run_exp.py exactly)
# ======================================================================

def _dpboost_params(total_budget):
    return {
        'boosting_type':          'gbdt',
        'objective':              'regression',
        'metric':                 'rmse',
        'num_leaves':             31,
        'max_depth':              6,
        'learning_rate':          0.1,
        'num_iterations':         N_TREES,
        'my_n_trees':             N_TREES,
        'lambda_l2':              0.1,
        'bagging_freq':           1,
        'bagging_fraction':       0.5,
        'max_bin':                255,
        'total_budget':           total_budget,
        'boost_method':           'DPBoost_2level',
        'high_level_boost_round': 1,
        'inner_boost_round':      INNER_BOOST_ROUND,
        'balance_partition':      BALANCE_PARTITION,
        'geo_clip':               1,
        'verbose':                -1,
    }

# ======================================================================
#  PER-DATASET SUMMARY PRINTER
# ======================================================================

def print_dataset_summary(results, dataset_name):
    df = pd.DataFrame(results)

    print(f"\n  {'─'*60}")
    print(f"  RESULTS — {dataset_name.upper()} (mean ± std over {N_SPLITS} splits)")
    print(f"  {'─'*60}")

    models = df["model"].unique()
    eps_order = sorted(
        df["epsilon"].unique(),
        key=lambda x: (0, float(x)) if x != "Non-Private" else (1, 0),
    )

    print(f"\n  RMSE (lower is better):")
    header = f"    {'epsilon':<14}"
    for m in models:
        header += f"{m:<24}"
    print(header)
    print(f"    {'─'*14}" + "─" * 24 * len(models))

    for eps_label in eps_order:
        line = f"    {eps_label:<14}"
        for m in models:
            subset = df[
                (df["model"] == m) & (df["epsilon"] == eps_label)
            ]["RMSE"]
            if len(subset) > 0:
                line += f"{subset.mean():.4f} ± {subset.std():.4f}       "
            else:
                line += f"{'—':^24}"
        print(line)

    print(f"\n  R² (higher is better):")
    header = f"    {'epsilon':<14}"
    for m in models:
        header += f"{m:<24}"
    print(header)
    print(f"    {'─'*14}" + "─" * 24 * len(models))

    for eps_label in eps_order:
        line = f"    {eps_label:<14}"
        for m in models:
            subset = df[
                (df["model"] == m) & (df["epsilon"] == eps_label)
            ]["R2"]
            if len(subset) > 0:
                line += f"{subset.mean():.4f} ± {subset.std():.4f}       "
            else:
                line += f"{'—':^24}"
        print(line)

    print()

# ======================================================================
#  REGRESSION EXPERIMENT
# ======================================================================

def run_regression_experiment(X, y, dataset_name):
    print(f"\n{'='*70}")
    print(f"  REGRESSION — {dataset_name}  (N={X.shape[0]}, K={X.shape[1]})")
    print(f"{'='*70}")

    results = []

    for split_idx in range(N_SPLITS):
        seed = 42 + split_idx
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=seed
        )

        # Scale y to [-1, 1]  (same as run_exp.py)
        scaler_y = MinMaxScaler(feature_range=(-1, 1))
        y_train_sc = scaler_y.fit_transform(y_train.reshape(-1, 1)).ravel()

        train_data = lgb.Dataset(X_train, y_train_sc)

        # ── Baseline (large budget ≈ non-private) ─────────────────
        params = _dpboost_params(BASELINE_BUDGET)
        model = lgb.train(params, train_data, num_boost_round=N_TREES)
        y_pred_sc = model.predict(X_test)
        y_pred = scaler_y.inverse_transform(y_pred_sc.reshape(-1, 1)).ravel()
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        r2   = r2_score(y_test, y_pred)
        results.append({
            "dataset": dataset_name, "model": "DPBoost-NonPrivate",
            "epsilon": "Non-Private", "split": split_idx + 1,
            "RMSE": round(rmse, 6), "R2": round(r2, 6),
        })

        # ── DPBoost at each epsilon ───────────────────────────────
        for eps in EPSILONS:
            params = _dpboost_params(eps)
            model = lgb.train(params, train_data, num_boost_round=N_TREES)
            y_pred_sc = model.predict(X_test)
            y_pred = scaler_y.inverse_transform(y_pred_sc.reshape(-1, 1)).ravel()
            rmse = np.sqrt(mean_squared_error(y_test, y_pred))
            r2   = r2_score(y_test, y_pred)
            results.append({
                "dataset": dataset_name, "model": "DPBoost",
                "epsilon": str(eps), "split": split_idx + 1,
                "RMSE": round(rmse, 6), "R2": round(r2, 6),
            })

        if (split_idx + 1) % 5 == 0:
            print(f"    Completed split {split_idx + 1}/{N_SPLITS}")

    print_dataset_summary(results, dataset_name)
    return results

# ======================================================================
#  GLOBAL SUMMARY
# ======================================================================

def print_global_summary(df, metric_col, direction):
    print(f"\n{'─'*70}")
    print(f"  {metric_col} ({direction})")
    print(f"{'─'*70}")

    for ds in df["dataset"].unique():
        ds_df = df[df["dataset"] == ds]
        print(f"\n  {ds.upper()}")

        models = ds_df["model"].unique()
        header = f"  {'epsilon':<14}"
        for m in models:
            header += f"{m:<24}"
        print(header)
        print(f"  {'─'*14}" + "─" * 24 * len(models))

        eps_order = sorted(
            ds_df["epsilon"].unique(),
            key=lambda x: (0, float(x)) if x != "Non-Private" else (1, 0),
        )
        for eps_label in eps_order:
            line = f"  {eps_label:<14}"
            for m in models:
                subset = ds_df[
                    (ds_df["model"] == m) & (ds_df["epsilon"] == eps_label)
                ][metric_col]
                if len(subset) > 0:
                    line += f"{subset.mean():.4f} ± {subset.std():.4f}       "
                else:
                    line += f"{'—':^24}"
            print(line)

# ======================================================================
#  MAIN
# ======================================================================

def main():
    all_results = []

    for loader in ALL_LOADERS:
        try:
            X, y, dataset_name = loader()
        except Exception as e:
            print(f"ERROR loading {loader.__name__}: {e}")
            continue

        results = run_regression_experiment(X, y, dataset_name)
        all_results.extend(results)

    if all_results:
        reg_df = pd.DataFrame(all_results)
        reg_path = os.path.join(BASE_PATH, "dpboost_regression_results.csv")
        reg_df.to_csv(reg_path, index=False)
        print(f"\nRegression results saved to: {reg_path}")
        print(f"\n{'='*70}")
        print(f"  GLOBAL REGRESSION SUMMARY")
        print(f"{'='*70}")
        print_global_summary(reg_df, "RMSE", "lower is better")
        print_global_summary(reg_df, "R2", "higher is better")

    print("\nDone.")


if __name__ == "__main__":
    main()