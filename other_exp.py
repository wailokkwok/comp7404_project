import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import copy

# Import the classes from your provided file
from DPEBM import ExplainableBoostingMachine, DPExplainableBoostingMachine


def load_and_preprocess_adult(filepath):
    print(f"Loading dataset from {filepath}...")
    df = pd.read_csv(filepath)

    # Strip whitespace from column names just in case
    df.columns = df.columns.str.strip()

    # We only want the numeric features for these experiments
    numeric_features = [
        "age",
        "fnlwgt",
        "education-num",
        "capital-gain",
        "capital-loss",
        "hours-per-week",
    ]

    # Extract X
    X = df[numeric_features].values

    # Extract and binarize y (income)
    # The adult dataset often has leading spaces in strings, so we strip them
    y_raw = df["income"].astype(str).str.strip()
    y = (y_raw == ">50K").astype(int).values

    print(f"Dataset loaded. Shape: {X.shape}. Positive class ratio: {y.mean():.4f}")
    return X, y, numeric_features


def experiment_2(X, y, feature_names):
    print("\n--- Running Experiment 2: Shape Function Visualization ---")
    age_idx = feature_names.index("age")

    # Step 2: Train non-private EBM (using default 25 bags)
    print("Training standard EBM...")
    ebm_standard = ExplainableBoostingMachine(task="classification", n_bags=25)
    ebm_standard.fit(X, y)
    centers_std, sf_std = ebm_standard.explain_feature(age_idx)

    # Step 4: Train DPEBM with eps=4
    print("Training DPEBM (eps=4)...")
    ebm_dp4 = DPExplainableBoostingMachine(
        task="classification", epsilon=4.0, delta=1e-6
    )
    ebm_dp4.fit(X, y)
    centers_dp4, sf_dp4 = ebm_dp4.explain_feature(age_idx)

    # Step 6: Train DPEBM with eps=1
    print("Training DPEBM (eps=1)...")
    ebm_dp1 = DPExplainableBoostingMachine(
        task="classification", epsilon=1.0, delta=1e-6
    )
    ebm_dp1.fit(X, y)
    centers_dp1, sf_dp1 = ebm_dp1.explain_feature(age_idx)

    # Step 8: Plot Figure 3
    fig, axes = plt.subplots(1, 3, figsize=(15, 5), sharey=True)

    axes[0].step(centers_std, sf_std, where="mid", color="blue")
    axes[0].set_title("Standard EBM")
    axes[0].set_xlabel("Age")
    axes[0].set_ylabel("Contribution Score")

    axes[1].step(centers_dp4, sf_dp4, where="mid", color="green")
    axes[1].set_title("DP-EBM (ε=4)")
    axes[1].set_xlabel("Age")

    axes[2].step(centers_dp1, sf_dp1, where="mid", color="red")
    axes[2].set_title("DP-EBM (ε=1)")
    axes[2].set_xlabel("Age")

    plt.tight_layout()
    plt.savefig("Figure_3_Shape_Functions.png")
    plt.close(fig)  # Closes the figure to free up memory
    print("Saved Figure_3_Shape_Functions.png")

    return ebm_dp1, centers_dp1, sf_dp1


def experiment_3(ebm_dp1, centers_dp1, sf_dp1, feature_names):
    print("\n--- Running Experiment 3: Post-Processing Monotonicity ---")
    age_idx = feature_names.index("age")

    # Step 1: Take noisy model from Exp 2 and copy it
    # We copy it so we don't mutate the original model's state for future reference
    ebm_dp1_repaired = copy.deepcopy(ebm_dp1)

    # Step 2: Apply isotonic regression
    print("Enforcing monotonicity on the Age shape function...")
    ebm_dp1_repaired.enforce_monotonicity(age_idx, increasing=True)
    _, sf_repaired = ebm_dp1_repaired.explain_feature(age_idx)

    # Step 3: Plot Figure 4
    fig, axes = plt.subplots(1, 3, figsize=(15, 5), sharey=True)

    axes[0].step(centers_dp1, sf_dp1, where="mid", color="red")
    axes[0].set_title("Original Noisy (ε=1)")
    axes[0].set_xlabel("Age")
    axes[0].set_ylabel("Contribution Score")

    axes[1].step(centers_dp1, sf_repaired, where="mid", color="purple")
    axes[1].set_title("Repaired Monotonic")
    axes[1].set_xlabel("Age")

    axes[2].step(centers_dp1, sf_dp1, where="mid", color="red", alpha=0.4, label="Noisy")
    axes[2].step(
        centers_dp1, sf_repaired, where="mid", color="purple", linewidth=2, label="Repaired"
    )
    axes[2].set_title("Overlay")
    axes[2].set_xlabel("Age")
    axes[2].legend()

    plt.tight_layout()
    plt.savefig("Figure_4_Monotonicity.png")
    plt.close(fig)  # Closes the figure to free up memory
    print("Saved Figure_4_Monotonicity.png")


def experiment_4(X, y, feature_names):
    print("\n--- Running Experiment 4: Regularization Effect of DP Noise ---")

    # Train the 4 required models
    print("Training standard EBM (25 bags)...")
    ebm_bag25 = ExplainableBoostingMachine(task="classification", n_bags=25)
    ebm_bag25.fit(X, y)

    print("Training standard EBM (1 bag / no bagging)...")
    ebm_bag1 = ExplainableBoostingMachine(task="classification", n_bags=1)
    ebm_bag1.fit(X, y)

    print("Training DPEBM (ε=4)...")
    dp_eps4 = DPExplainableBoostingMachine(
        task="classification", epsilon=4.0, delta=1e-6
    )
    dp_eps4.fit(X, y)

    print("Training DPEBM (ε=1)...")
    dp_eps1 = DPExplainableBoostingMachine(
        task="classification", epsilon=1.0, delta=1e-6
    )
    dp_eps1.fit(X, y)

    models = {
        "EBM (25 Bags)": ebm_bag25,
        "EBM (No Bagging)": ebm_bag1,
        "DP-EBM (ε=4)": dp_eps4,
        "DP-EBM (ε=1)": dp_eps1,
    }

    # Setup grid plot: rows = features, cols = models
    n_features = len(feature_names)
    n_models = len(models)

    fig, axes = plt.subplots(n_features, n_models, figsize=(16, 3 * n_features))

    for row_idx, feature_name in enumerate(feature_names):
        for col_idx, (model_name, model) in enumerate(models.items()):
            ax = axes[row_idx, col_idx]
            centers, sf = model.explain_feature(row_idx)

            ax.step(centers, sf, where="mid", color="steelblue")

            # Format axes
            if row_idx == 0:
                ax.set_title(model_name, fontsize=12, fontweight="bold")
            if col_idx == 0:
                ax.set_ylabel(f"{feature_name}\nScore", fontweight="bold")
            if row_idx == n_features - 1:
                ax.set_xlabel(feature_name)

            # Keep Y-axis consistent across a single feature row for direct visual comparison
            if col_idx > 0:
                ax.sharey(axes[row_idx, 0])

    plt.tight_layout()
    plt.savefig("Figure_5_Regularization_Grid.png")
    plt.close(fig)  # Closes the figure to free up memory
    print("Saved Figure_5_Regularization_Grid.png")


if __name__ == "__main__":
    # Update this path if necessary
    FILE_PATH = "/Users/kwokwailok/Desktop/comp7404/datasets/adult_train.csv"

    # 1. Load Data
    X, y, features = load_and_preprocess_adult(FILE_PATH)

    # 2. Run Experiments
    dp_eps1_model, centers_dp1, sf_dp1 = experiment_2(X, y, features)
    experiment_3(dp_eps1_model, centers_dp1, sf_dp1, features)
    experiment_4(X, y, features)