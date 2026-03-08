#!/usr/bin/env python3
"""
Experiment runner for DP-EBM paper replication (Nori et al., 2021).

Models: DPEBM-GDP, DPEBM-Classic, EBM-NonPrivate
Datasets: adult-income, credit-fraud, telco-churn, cal-housing, elevators, pol
Epsilons: 0.5, 1.0, 2.0, 4.0, 8.0, Non-Private
Splits: 25 random 80/20 per (dataset, model, epsilon)

Output: CSV with columns [dataset, model, epsilon, split, metric_name, metric_value]
        + saved model weights (pickle) for every trained model
"""

import numpy as np
import pandas as pd
import os
import time
import pickle
import warnings
from datetime import timedelta
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, mean_squared_error
from sklearn.preprocessing import LabelEncoder

warnings.filterwarnings("ignore")

from DPEBM import ExplainableBoostingMachine, DPExplainableBoostingMachine


# ======================================================================
#  CONFIGURATION
# ======================================================================

BASE_PATH = "/Users/kwokwailok/Desktop/comp7404/datasets"
OUTPUT_DIR = os.path.join(BASE_PATH, "..")
OUTPUT_CSV = os.path.join(OUTPUT_DIR, "experiment_results4.csv")
MODEL_DIR = os.path.join(OUTPUT_DIR, "saved_models4")

N_SPLITS = 25
DELTA = 1e-6
EPSILONS = [0.5, 1.0, 2.0, 4.0, 8.0]

MODELS = [
    "DPEBM-GDP",
    "DPEBM-Classic",
    "EBM-NonPrivate",
]


# ======================================================================
#  DATASET LOADERS
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
        df = pd.read_csv(
            path, header=None, names=col_names, skipinitialspace=True
        )

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
        X_df["TotalCharges"] = pd.to_numeric(
            X_df["TotalCharges"], errors="coerce"
        )

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
    load_adult_income,
    load_credit_fraud,
    load_telco_churn,
    load_cal_housing,
    load_elevators,
    load_pol,
]


# ======================================================================
#  MODEL FACTORY
# ======================================================================

def build_model(model_name, task, epsilon, label_range, random_state):
    if model_name == "DPEBM-GDP":
        return DPExplainableBoostingMachine(
            n_bins=32,
            max_rounds=300,
            learning_rate=0.01,
            max_leaves=3,
            min_samples_leaf=2,
            task=task,
            n_bags=1,
            epsilon=epsilon,
            delta=DELTA,
            label_range=label_range,
            bin_budget_fraction=0.10,
            intercept_budget_fraction=0.01,
            composition="gdp",
            random_state=random_state,
            early_stopping_rounds=None,
        )

    elif model_name == "DPEBM-Classic":
        return DPExplainableBoostingMachine(
            n_bins=32,
            max_rounds=300,
            learning_rate=0.01,
            max_leaves=3,
            min_samples_leaf=2,
            task=task,
            n_bags=1,
            epsilon=epsilon,
            delta=DELTA,
            label_range=label_range,
            bin_budget_fraction=0.10,
            intercept_budget_fraction=0.01,
            composition="classic",
            random_state=random_state,
            early_stopping_rounds=None,
        )

    elif model_name == "EBM-NonPrivate":
        # ✅ FIX #3: n_bags=25 to match paper (was n_bags=1)
        return ExplainableBoostingMachine(
            n_bins=32,
            max_rounds=300,
            learning_rate=0.01,
            max_leaves=3,
            min_samples_leaf=2,
            task=task,
            n_bags=25,
            bag_fraction=0.8,
            random_state=random_state,
            early_stopping_rounds=50,
        )

    else:
        raise ValueError(f"Unknown model: {model_name}")


# ======================================================================
#  SAVE / LOAD MODEL WEIGHTS
# ======================================================================

def _safe_name(name):
    """Sanitise a string so it is safe for use as a directory/file name."""
    return name.replace(".", "p").replace(" ", "_")


def get_model_dir(dataset_name, model_name, eps_label):
    """Return the nested directory path for a given (dataset, model, epsilon)."""
    return os.path.join(
        MODEL_DIR,
        _safe_name(dataset_name),
        _safe_name(model_name),
        f"eps_{_safe_name(eps_label)}",
    )


def get_model_filepath(dataset_name, model_name, eps_label, split_idx):
    """Return the full file path for a saved model."""
    directory = get_model_dir(dataset_name, model_name, eps_label)
    filename = f"split_{split_idx + 1}.pkl"
    return os.path.join(directory, filename)


def save_model(model, dataset_name, model_name, eps_label, split_idx):
    """Pickle the entire model object to disk in a nested directory."""
    fpath = get_model_filepath(dataset_name, model_name, eps_label, split_idx)
    os.makedirs(os.path.dirname(fpath), exist_ok=True)
    with open(fpath, "wb") as f:
        pickle.dump(model, f, protocol=pickle.HIGHEST_PROTOCOL)
    return fpath


def model_already_saved(dataset_name, model_name, eps_label, split_idx):
    """Check if a model file already exists on disk."""
    fpath = get_model_filepath(dataset_name, model_name, eps_label, split_idx)
    return os.path.exists(fpath)


# ======================================================================
#  EVALUATION
# ======================================================================

def evaluate_model(model, X_test, y_test, task):
    if task == "classification":
        try:
            if hasattr(model, "predict_proba"):
                proba = model.predict_proba(X_test)
                if proba.ndim == 2:
                    proba = proba[:, 1]
            elif hasattr(model, "_raw_predict"):
                proba = model._raw_predict(X_test)
            else:
                proba = model.predict(X_test).astype(float)
            score = roc_auc_score(y_test, proba)
        except Exception:
            score = np.nan
        return "AUROC", score
    else:
        preds = model.predict(X_test)
        if isinstance(preds, pd.Series):
            preds = preds.values
        rmse = np.sqrt(mean_squared_error(y_test, preds))
        return "RMSE", rmse


def run_one(X, y, task, dataset_name, model_name, epsilon, eps_label,
            split_idx, split_seed, label_range):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=split_seed
    )

    lr = label_range
    if lr is None and task == "regression":
        lr = float(y_train.max() - y_train.min())
    if task == "classification":
        lr = 1.0

    model_seed = split_seed * 7 + 13
    model = build_model(model_name, task, epsilon, lr, model_seed)
    model.fit(X_train, y_train)

    # ── Save model weights ────────────────────────────────────
    saved_path = save_model(model, dataset_name, model_name, eps_label, split_idx)

    metric_name, metric_value = evaluate_model(model, X_test, y_test, task)
    return metric_name, metric_value, saved_path


# ======================================================================
#  MAIN LOOP
# ======================================================================

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(MODEL_DIR, exist_ok=True)
    all_results = []

    # Resume support
    if os.path.exists(OUTPUT_CSV):
        existing = pd.read_csv(OUTPUT_CSV)
        all_results = existing.to_dict("records")
        done_keys = set(
            (r["dataset"], r["model"], str(r["epsilon"]), int(r["split"]))
            for r in all_results
        )
        print(f"Resuming: found {len(all_results)} existing results.\n")
    else:
        done_keys = set()

    total_start = time.time()

    for loader in ALL_LOADERS:
        # ── Load dataset ──────────────────────────────────────────
        print(f"\n{'='*70}")
        ds_start = time.time()
        try:
            X, y, task, dataset_name = loader()
        except Exception as e:
            print(f"ERROR loading {loader.__name__}: {e}")
            continue

        n_samples, n_features = X.shape
        print(f"Dataset: {dataset_name}")
        print(f"  N={n_samples}, K={n_features}, task={task}")
        if task == "classification":
            print(f"  Positive rate: {y.mean():.3f}")

        label_range = None
        if task == "regression":
            label_range = float(y.max() - y.min())
            print(f"  Label range (R): {label_range:.4f}")

        # ── Build job list ────────────────────────────────────────
        jobs = []
        for model_name in MODELS:
            if model_name == "EBM-NonPrivate":
                eps_label = "Non-Private"
                for s in range(N_SPLITS):
                    key = (dataset_name, model_name, eps_label, s + 1)
                    if key not in done_keys:
                        jobs.append((model_name, None, s, eps_label))
            else:
                for eps in EPSILONS:
                    eps_label = str(eps)
                    for s in range(N_SPLITS):
                        key = (dataset_name, model_name, eps_label, s + 1)
                        if key not in done_keys:
                            jobs.append((model_name, eps, s, eps_label))

        if not jobs:
            print(f"  All experiments already completed. Skipping.")
            continue

        print(f"  Experiments to run: {len(jobs)}")
        print(f"  Models will be saved to: {MODEL_DIR}")
        print(f"{'─'*70}")

        # ── Execute jobs ──────────────────────────────────────────
        completed = 0
        last_model_eps = (None, None)

        for model_name, eps, split_idx, eps_label in jobs:
            split_seed = 42 + split_idx

            if (model_name, eps_label) != last_model_eps:
                last_model_eps = (model_name, eps_label)
                print(f"\n  ▸ {model_name}, ε={eps_label}")

            t0 = time.time()
            try:
                result = run_one(
                    X, y, task, dataset_name, model_name, eps, eps_label,
                    split_idx, split_seed, label_range
                )
            except Exception as e:
                print(f"    Split {split_idx+1:2d}: ERROR — {e}")
                result = None

            elapsed = time.time() - t0

            if result is None:
                continue

            metric_name, metric_value, saved_path = result
            row = {
                "dataset": dataset_name,
                "model": model_name,
                "epsilon": eps_label,
                "split": split_idx + 1,
                "metric_name": metric_name,
                "metric_value": round(metric_value, 6),
            }
            all_results.append(row)

            print(
                f"    Split {split_idx+1:2d}: {metric_name}={metric_value:.4f}"
                f"  ({elapsed:.1f}s)  [saved]"
            )

            completed += 1

            # Save CSV every 25 experiments
            if completed % N_SPLITS == 0:
                pd.DataFrame(all_results).to_csv(OUTPUT_CSV, index=False)

        # Save after each dataset
        pd.DataFrame(all_results).to_csv(OUTPUT_CSV, index=False)
        ds_elapsed = time.time() - ds_start
        print(
            f"\n  Dataset {dataset_name} done in "
            f"{timedelta(seconds=int(ds_elapsed))}"
        )

    # ── Final save ────────────────────────────────────────────────
    results_df = pd.DataFrame(all_results)
    results_df.to_csv(OUTPUT_CSV, index=False)

    total_elapsed = time.time() - total_start
    print(f"\n{'='*70}")
    print(f"ALL EXPERIMENTS COMPLETE")
    print(f"Total time: {timedelta(seconds=int(total_elapsed))}")
    print(f"Results saved to: {OUTPUT_CSV}")
    print(f"Models saved to:  {MODEL_DIR}")
    print(f"Total rows: {len(results_df)}")

    # Count saved models
    n_models = sum(
        len(files)
        for _, _, files in os.walk(MODEL_DIR)
        if files
    )
    print(f"Total saved models: {n_models}")

    print_summary(results_df)


# ======================================================================
#  SUMMARY PRINTER
# ======================================================================

def print_summary(df):
    print(f"\n{'='*70}")
    print("SUMMARY TABLES (mean ± std across 25 splits)")
    print(f"{'='*70}")

    for metric_name, direction in [("AUROC", "higher is better"), ("RMSE", "lower is better")]:
        task_df = df[df["metric_name"] == metric_name]
        if task_df.empty:
            continue

        print(f"\n{'─'*70}")
        print(f"  {metric_name} ({direction})")
        print(f"{'─'*70}")

        datasets = task_df["dataset"].unique()

        for ds in datasets:
            ds_df = task_df[task_df["dataset"] == ds]
            print(f"\n  {ds.upper()}")

            models_present = ds_df["model"].unique()
            header = f"  {'epsilon':<14}"
            for m in models_present:
                header += f"{m:<22}"
            print(header)

            separator = f"  {'─'*14}"
            for _ in models_present:
                separator += f"{'─'*22}"
            print(separator)

            epsilons_in_data = sorted(
                ds_df["epsilon"].unique(),
                key=lambda x: (0, float(x)) if x != "Non-Private" else (1, 0),
            )

            for eps_label in epsilons_in_data:
                line = f"  {eps_label:<14}"
                for m in models_present:
                    subset = ds_df[
                        (ds_df["model"] == m) & (ds_df["epsilon"] == eps_label)
                    ]["metric_value"]
                    if len(subset) > 0:
                        mean_val = subset.mean()
                        std_val = subset.std()
                        line += f"{mean_val:.3f} ± {std_val:.3f}      "
                    else:
                        line += f"{'—':^22}"
                print(line)

    print()


# ======================================================================
#  ENTRY POINT
# ======================================================================

if __name__ == "__main__":
    main()