#!/usr/bin/env python3
"""
Train baseline (sklearn) and DP (diffprivlib) Linear Regression
using the same data loading / processing pipeline as the experiment runner.
"""

import numpy as np
import pandas as pd
import os
import warnings
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.metrics import mean_squared_error, r2_score, roc_auc_score
from sklearn.linear_model import LinearRegression as SKLearnLR
from sklearn.linear_model import LogisticRegression as SKLearnLogReg
from diffprivlib.models import LinearRegression as DPLR
from diffprivlib.models import LogisticRegression as DPLogReg

warnings.filterwarnings("ignore")

# ======================================================================
#  CONFIGURATION
# ======================================================================

BASE_PATH = "/Users/kwokwailok/Desktop/comp7404/datasets"

N_SPLITS = 25
EPSILONS = [0.5, 1.0, 2.0, 4.0, 8.0]
DELTA = 1e-6

# ======================================================================
#  DATA LOADING — identical to experiment runner
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


def load_adult_income():
    path = os.path.join(BASE_PATH, "adult_train.csv")
    col_names = [
        "age", "workclass", "fnlwgt", "education", "education_num",
        "marital_status", "occupation", "relationship", "race", "sex",
        "capital_gain", "capital_loss", "hours_per_week", "native_country",
        "income",
    ]
    df = pd.read_csv(path, skipinitialspace=True)
    if "income" not in df.columns and "label" not in df.columns:
        df = pd.read_csv(path, header=None, names=col_names, skipinitialspace=True)
    target_col = df.columns[-1]
    y_raw = df[target_col].astype(str).str.strip()
    y = (y_raw.str.contains(">50K") | y_raw.str.contains("1")).astype(float).values
    X_df = df.drop(columns=[target_col])
    X_df = X_df.replace("?", np.nan).replace(" ?", np.nan)
    mask = X_df.notna().all(axis=1)
    X_df, y = X_df[mask].reset_index(drop=True), y[mask]
    X = _encode_dataframe(X_df)
    valid = ~(np.isnan(X).any(axis=1) | np.isnan(y))
    X, y = X[valid], y[valid]
    return X, y, "classification", "adult-income"


def load_credit_fraud():
    path = os.path.join(BASE_PATH, "creditcard.csv")
    df = pd.read_csv(path)
    target_col = "Class" if "Class" in df.columns else df.columns[-1]
    y = df[target_col].values.astype(float)
    drop_cols = [target_col]
    if "Time" in df.columns:
        drop_cols.append("Time")
    X = df.drop(columns=drop_cols).values.astype(float)
    valid = ~(np.isnan(X).any(axis=1) | np.isnan(y))
    X, y = X[valid], y[valid]
    return X, y, "classification", "credit-fraud"


def load_telco_churn():
    path = os.path.join(BASE_PATH, "telco_churn.csv")
    df = pd.read_csv(path)
    target_col = df.columns[-1]
    y_raw = df[target_col].astype(str).str.strip()
    y = y_raw.isin(["Yes", "1", "True", "yes", "YES"]).astype(float).values
    X_df = df.drop(columns=[target_col])
    for c in ["customerID", "CustomerID", "customer_id"]:
        if c in X_df.columns:
            X_df = X_df.drop(columns=[c])
    if "TotalCharges" in X_df.columns:
        X_df["TotalCharges"] = pd.to_numeric(X_df["TotalCharges"], errors="coerce")
    mask = X_df.notna().all(axis=1)
    X_df, y = X_df[mask].reset_index(drop=True), y[mask]
    X = _encode_dataframe(X_df)
    valid = ~(np.isnan(X).any(axis=1) | np.isnan(y))
    X, y = X[valid], y[valid]
    return X, y, "classification", "telco-churn"


def load_cal_housing():
    path = os.path.join(BASE_PATH, "cal_housing.csv")
    df = pd.read_csv(path)
    target_col = df.columns[-1]
    y = df[target_col].values.astype(float)
    X_df = df.drop(columns=[target_col])
    X = _encode_dataframe(X_df)
    valid = ~(np.isnan(X).any(axis=1) | np.isnan(y))
    X, y = X[valid], y[valid]
    return X, y, "regression", "cal-housing"


def load_elevators():
    path = os.path.join(BASE_PATH, "elevators.csv")
    df = pd.read_csv(path)
    target_col = df.columns[-1]
    y = df[target_col].values.astype(float)
    X = df.drop(columns=[target_col]).values.astype(float)
    valid = ~(np.isnan(X).any(axis=1) | np.isnan(y))
    X, y = X[valid], y[valid]
    return X, y, "regression", "elevators"


def load_pol():
    path = os.path.join(BASE_PATH, "pol.csv")
    df = pd.read_csv(path)
    target_col = df.columns[-1]
    y = df[target_col].values.astype(float)
    X = df.drop(columns=[target_col]).values.astype(float)
    valid = ~(np.isnan(X).any(axis=1) | np.isnan(y))
    X, y = X[valid], y[valid]
    return X, y, "regression", "pol"


ALL_LOADERS = [
    load_cal_housing,
    load_elevators,
    load_pol,
    load_adult_income,
    load_credit_fraud,
    load_telco_churn
]

# ======================================================================
#  PER-DATASET SUMMARY PRINTER
# ======================================================================

def print_dataset_summary(results, task, dataset_name):
    """Print a summary table for one dataset immediately after it finishes."""
    df = pd.DataFrame(results)

    print(f"\n  {'─'*60}")
    print(f"  RESULTS — {dataset_name.upper()} (mean ± std over {N_SPLITS} splits)")
    print(f"  {'─'*60}")

    models = df["model"].unique()
    eps_order = sorted(
        df["epsilon"].unique(),
        key=lambda x: (0, float(x)) if x != "Non-Private" else (1, 0),
    )

    if task == "regression":
        # ── RMSE table ──
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

        # ── R2 table ──
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

    else:
        # ── AUROC table ──
        print(f"\n  AUROC (higher is better):")
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
                ]["AUROC"]
                if len(subset) > 0:
                    line += f"{subset.mean():.4f} ± {subset.std():.4f}       "
                else:
                    line += f"{'—':^24}"
            print(line)

    print()

# ======================================================================
#  EXPERIMENT LOOP
# ======================================================================

def run_regression_experiment(X, y, dataset_name):
    """Run 25 splits of baseline LR + DP-LR at each epsilon."""
    print(f"\n{'='*70}")
    print(f"  REGRESSION — {dataset_name}  (N={X.shape[0]}, K={X.shape[1]})")
    print(f"{'='*70}")

    results = []

    for split_idx in range(N_SPLITS):
        seed = 42 + split_idx
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=seed
        )

        # ── Baseline (non-private) — NO scaling needed ───────────
        baseline = SKLearnLR()
        baseline.fit(X_train, y_train)
        y_pred = baseline.predict(X_test)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        r2 = r2_score(y_test, y_pred)
        results.append({
            "dataset": dataset_name, "model": "LR-NonPrivate",
            "epsilon": "Non-Private", "split": split_idx + 1,
            "RMSE": round(rmse, 6), "R2": round(r2, 6),
        })

        # ── Scale features to [0, 1] (needed ONLY for diffprivlib) ──
        scaler_X = MinMaxScaler(feature_range=(0, 1))
        X_train_sc = scaler_X.fit_transform(X_train)
        X_test_sc = scaler_X.transform(X_test)

        scaler_y = MinMaxScaler(feature_range=(0, 1))
        y_train_sc = scaler_y.fit_transform(y_train.reshape(-1, 1)).ravel()

        # ── DP Linear Regression at each epsilon ─────────────────
        for eps in EPSILONS:
            dp_model = DPLR(
                epsilon=eps,
                bounds_X=(0, 1),
                bounds_y=(0, 1),
                fit_intercept=True,
            )
            dp_model.fit(X_train_sc, y_train_sc)
            y_pred_sc = dp_model.predict(X_test_sc)
            y_pred_sc = np.clip(y_pred_sc, 0, 1)
            y_pred = scaler_y.inverse_transform(
                y_pred_sc.reshape(-1, 1)
            ).ravel()
            rmse = np.sqrt(mean_squared_error(y_test, y_pred))
            r2 = r2_score(y_test, y_pred)
            results.append({
                "dataset": dataset_name, "model": "DPLR",
                "epsilon": str(eps), "split": split_idx + 1,
                "RMSE": round(rmse, 6), "R2": round(r2, 6),
            })

        if (split_idx + 1) % 5 == 0:
            print(f"    Completed split {split_idx + 1}/{N_SPLITS}")

    print_dataset_summary(results, "regression", dataset_name)
    return results


def run_classification_experiment(X, y, dataset_name):
    """Run 25 splits of baseline LogReg + DP-LogReg at each epsilon."""
    print(f"\n{'='*70}")
    print(f"  CLASSIFICATION — {dataset_name}  (N={X.shape[0]}, K={X.shape[1]})")
    print(f"{'='*70}")

    results = []

    for split_idx in range(N_SPLITS):
        seed = 42 + split_idx
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=seed
        )

        # ── Scale features to [0, 1] ─────────────────────────────
        scaler_X = MinMaxScaler(feature_range=(0, 1))
        X_train_sc = scaler_X.fit_transform(X_train)
        X_test_sc = scaler_X.transform(X_test)

        # ── Baseline (non-private) ────────────────────────────────
        baseline = SKLearnLogReg(max_iter=1000, solver="lbfgs")
        baseline.fit(X_train_sc, y_train)
        y_proba = baseline.predict_proba(X_test_sc)[:, 1]
        auroc = roc_auc_score(y_test, y_proba)
        results.append({
            "dataset": dataset_name, "model": "LogReg-NonPrivate",
            "epsilon": "Non-Private", "split": split_idx + 1,
            "AUROC": round(auroc, 6),
        })

        # ── DP Logistic Regression at each epsilon ────────────────
        for eps in EPSILONS:
            dp_model = DPLogReg(
                epsilon=eps,
                bounds_X=(0, 1),
                data_norm=np.sqrt(X_train_sc.shape[1]),
                max_iter=1000,
            )
            dp_model.fit(X_train_sc, y_train)
            y_proba = dp_model.predict_proba(X_test_sc)[:, 1]
            try:
                auroc = roc_auc_score(y_test, y_proba)
            except Exception:
                auroc = np.nan
            results.append({
                "dataset": dataset_name, "model": "DPLogReg",
                "epsilon": str(eps), "split": split_idx + 1,
                "AUROC": round(auroc, 6),
            })

        if (split_idx + 1) % 5 == 0:
            print(f"    Completed split {split_idx + 1}/{N_SPLITS}")

    # ── Print summary for this dataset ────────────────────────────
    print_dataset_summary(results, "classification", dataset_name)

    return results


# ======================================================================
#  GLOBAL SUMMARY (end of run)
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
    all_reg_results = []
    all_cls_results = []

    for loader in ALL_LOADERS:
        try:
            X, y, task, dataset_name = loader()
        except Exception as e:
            print(f"ERROR loading {loader.__name__}: {e}")
            continue

        if task == "regression":
            results = run_regression_experiment(X, y, dataset_name)
            all_reg_results.extend(results)
        else:
            results = run_classification_experiment(X, y, dataset_name)
            all_cls_results.extend(results)

    # ── Save & final global summary ───────────────────────────────
    output_dir = os.path.join(BASE_PATH, "..")

    if all_reg_results:
        reg_df = pd.DataFrame(all_reg_results)
        reg_path = os.path.join(output_dir, "lr_regression_results.csv")
        reg_df.to_csv(reg_path, index=False)
        print(f"\nRegression results saved to: {reg_path}")
        print(f"\n{'='*70}")
        print(f"  GLOBAL REGRESSION SUMMARY")
        print(f"{'='*70}")
        print_global_summary(reg_df, "RMSE", "lower is better")
        print_global_summary(reg_df, "R2", "higher is better")

    if all_cls_results:
        cls_df = pd.DataFrame(all_cls_results)
        cls_path = os.path.join(output_dir, "lr_classification_results.csv")
        cls_df.to_csv(cls_path, index=False)
        print(f"\nClassification results saved to: {cls_path}")
        print(f"\n{'='*70}")
        print(f"  GLOBAL CLASSIFICATION SUMMARY")
        print(f"{'='*70}")
        print_global_summary(cls_df, "AUROC", "higher is better")

    print("\nDone.")


if __name__ == "__main__":
    main()