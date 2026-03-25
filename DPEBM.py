# ebm_corrected.py


import numpy as np
from typing import Optional
from scipy.stats import norm as _norm




class ExplainableBoostingMachine:
    """
    Explainable Boosting Machine (EBM) from scratch.

    Follows Algorithm 1 from Nori et al. (2021):
      - Quantile binning (equal-density)
      - Cyclic gradient boosting, one feature at a time
      - Each boosting step fits a shallow 1-D decision tree (max_leaves)
        that groups bins into leaves → coarser, regularized updates
      - Bagging for smoother shape functions (paper default: 25 bags)

    g(E[y]) = β₀ + f₁(x₁) + f₂(x₂) + … + fₖ(xₖ)

    Parameters
    ----------
    n_bins : int
        Max quantile bins per feature.
    max_rounds : int
        Number of full epochs (cycles through all features).
    learning_rate : float
        Shrinkage applied to each tree's predictions.
    max_leaves : int
        Maximum leaf nodes per 1-D tree (paper default = 3).
    min_samples_leaf : int
        Minimum samples required in a tree leaf / bin.
    task : str
        'regression' or 'classification' (binary).
    n_bags : int
        Number of bagging iterations (paper default = 25).
    bag_fraction : float
        Fraction of data sampled per bag (with replacement).
    early_stopping_rounds : int | None
        Stop if loss hasn't improved for this many full rounds.
    random_state : int | None
        Random seed for reproducibility.
    """

    def __init__(
        self,
        n_bins: int = 32,
        max_rounds: int = 300,
        learning_rate: float = 0.01,
        max_leaves: int = 3,
        min_samples_leaf: int = 2,
        task: str = "regression",
        n_bags: int = 25,
        bag_fraction: float = 0.8,
        early_stopping_rounds: Optional[int] = 50,
        random_state: Optional[int] = None,
    ):
        self.n_bins = n_bins
        self.max_rounds = max_rounds
        self.learning_rate = learning_rate
        self.max_leaves = max_leaves
        self.min_samples_leaf = min_samples_leaf
        self.task = task
        self.n_bags = n_bags
        self.bag_fraction = bag_fraction
        self.early_stopping_rounds = early_stopping_rounds
        self.random_state = random_state

    # ================================================================== #
    #  Binning                                                             #
    # ================================================================== #

    @staticmethod
    def _quantile_bin_edges(x: np.ndarray, n_bins: int) -> np.ndarray:
        """Compute unique quantile-based bin edges."""
        percentiles = np.linspace(0, 100, n_bins + 1)
        edges = np.unique(np.percentile(x, percentiles))
        return edges

    @staticmethod
    def _assign_bins(x: np.ndarray, edges: np.ndarray) -> np.ndarray:
        """Map continuous values to bin indices."""
        idx = np.searchsorted(edges, x, side="right") - 1
        return np.clip(idx, 0, len(edges) - 2)

    # ================================================================== #
    #  Loss / gradient helpers                                             #
    # ================================================================== #

    @staticmethod
    def _sigmoid(z: np.ndarray) -> np.ndarray:
        z = np.clip(z, -500, 500)
        return 1.0 / (1.0 + np.exp(-z))

    def _negative_gradient(self, y: np.ndarray, raw: np.ndarray) -> np.ndarray:
        if self.task == "classification":
            return y - self._sigmoid(raw)
        return y - raw

    def _loss(self, y: np.ndarray, raw: np.ndarray) -> float:
        if self.task == "classification":
            p = np.clip(self._sigmoid(raw), 1e-15, 1 - 1e-15)
            return -float(np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))
        return float(np.mean((y - raw) ** 2))

    # ================================================================== #
    #  1-D greedy decision tree on histogram bins                          #
    # ================================================================== #

    def _fit_1d_tree(
        self,
        bins: np.ndarray,
        residuals: np.ndarray,
        n_bins: int,
    ) -> np.ndarray:
        """
        Fit a greedy 1-D decision tree over the histogram for one feature.

        The tree can only split along bin boundaries, grouping contiguous
        bins into at most `max_leaves` leaves.  Each leaf predicts the
        learning-rate-scaled mean residual of its data.

        Parameters
        ----------
        bins : (n_samples,) int array of bin indices
        residuals : (n_samples,) float array of current residuals
        n_bins : number of bins for this feature

        Returns
        -------
        output : (n_bins,) float array — the update δf_k to add to the
                 shape function for this feature.
        """
        # ── Aggregate per-bin statistics ──────────────────────────────
        bin_sum = np.bincount(bins, weights=residuals, minlength=n_bins).astype(np.float64)
        bin_cnt = np.bincount(bins, minlength=n_bins).astype(np.float64)

        # ── Initialise: every bin is in one big leaf ──────────────────
        leaves: list[tuple[int, int]] = [(0, n_bins - 1)]

        # ── Greedily split until we reach max_leaves ──────────────────
        for _ in range(self.max_leaves - 1):
            best_gain = -np.inf
            best_leaf_idx = -1
            best_split_pos = -1

            for li, (lo, hi) in enumerate(leaves):
                if lo == hi:
                    continue

                sums = bin_sum[lo: hi + 1]
                cnts = bin_cnt[lo: hi + 1]
                cum_sum = np.cumsum(sums)
                cum_cnt = np.cumsum(cnts)
                total_sum = cum_sum[-1]
                total_cnt = cum_cnt[-1]

                if total_cnt < 2 * self.min_samples_leaf:
                    continue

                for s in range(len(sums) - 1):
                    l_cnt = cum_cnt[s]
                    r_cnt = total_cnt - l_cnt
                    if l_cnt < self.min_samples_leaf or r_cnt < self.min_samples_leaf:
                        continue

                    l_mean = cum_sum[s] / l_cnt if l_cnt > 0 else 0.0
                    r_mean = (total_sum - cum_sum[s]) / r_cnt if r_cnt > 0 else 0.0
                    gain = l_cnt * l_mean ** 2 + r_cnt * r_mean ** 2

                    if gain > best_gain:
                        best_gain = gain
                        best_leaf_idx = li
                        best_split_pos = s

            if best_leaf_idx < 0:
                break

            lo, hi = leaves[best_leaf_idx]
            split_bin = lo + best_split_pos
            leaves[best_leaf_idx] = (lo, split_bin)
            leaves.insert(best_leaf_idx + 1, (split_bin + 1, hi))

        # ── Compute leaf predictions → map back to bins ───────────────
        output = np.zeros(n_bins, dtype=np.float64)

        for lo, hi in leaves:
            leaf_sum = bin_sum[lo: hi + 1].sum()
            leaf_cnt = bin_cnt[lo: hi + 1].sum()

            if leaf_cnt >= self.min_samples_leaf:
                val = self.learning_rate * leaf_sum / leaf_cnt
            else:
                val = 0.0

            output[lo: hi + 1] = val

        return output

    # ================================================================== #
    #  Single-bag training                                                 #
    # ================================================================== #

    def _fit_single(
        self,
        X: np.ndarray,
        y: np.ndarray,
        X_binned: np.ndarray,
        rng: np.random.Generator,
    ) -> tuple[list[np.ndarray], float]:
        """Train one EBM on (possibly a bootstrap sample of) the data."""
        n_samples, n_features = X.shape

        shape_fns = [np.zeros(nb) for nb in self.n_bins_actual_]
        raw = np.full(n_samples, self.intercept_)

        best_loss = np.inf
        stale = 0

        for rnd in range(self.max_rounds):
            for j in range(n_features):
                residuals = self._negative_gradient(y, raw)
                bins_j = X_binned[:, j]
                nb = self.n_bins_actual_[j]

                step = self._fit_1d_tree(bins_j, residuals, nb)

                shape_fns[j] += step
                raw += step[bins_j]

            cur_loss = self._loss(y, raw)
            if cur_loss < best_loss - 1e-10:
                best_loss = cur_loss
                stale = 0
            else:
                stale += 1

            if self.early_stopping_rounds and stale >= self.early_stopping_rounds:
                break

        return shape_fns, best_loss

    # ================================================================== #
    #  Fit (public)                                                        #
    # ================================================================== #

    def fit(self, X, y):
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64).ravel()
        n_samples, n_features = X.shape
        rng = np.random.default_rng(self.random_state)

        # ── 1. Quantile-bin every feature ─────────────────────────────
        self.bin_edges_: list[np.ndarray] = []
        self.n_bins_actual_: list[int] = []
        X_binned = np.empty((n_samples, n_features), dtype=np.intp)

        for j in range(n_features):
            edges = self._quantile_bin_edges(X[:, j], self.n_bins)
            self.bin_edges_.append(edges)
            nb = max(len(edges) - 1, 1)
            self.n_bins_actual_.append(nb)
            X_binned[:, j] = self._assign_bins(X[:, j], edges)

        # ── 2. Intercept ──────────────────────────────────────────────
        if self.task == "classification":
            p = np.clip(y.mean(), 1e-10, 1 - 1e-10)
            self.intercept_ = float(np.log(p / (1 - p)))
        else:
            self.intercept_ = float(y.mean())

        # ── 3. Train (with bagging) ───────────────────────────────────
        all_shape_fns: list[list[np.ndarray]] = []

        for bag in range(self.n_bags):
            if self.n_bags > 1:
                idx = rng.choice(n_samples, size=int(n_samples * self.bag_fraction), replace=True)
                X_bag = X[idx]
                y_bag = y[idx]
                Xb_bag = X_binned[idx]
            else:
                X_bag = X
                y_bag = y
                Xb_bag = X_binned

            shape_fns, _ = self._fit_single(X_bag, y_bag, Xb_bag, rng)
            all_shape_fns.append(shape_fns)

        # ── 4. Average shape functions across bags ────────────────────
        self.shape_functions_: list[np.ndarray] = []
        for j in range(n_features):
            avg = np.mean([sf[j] for sf in all_shape_fns], axis=0)
            self.shape_functions_.append(avg)

        self.n_features_ = n_features
        return self

    # ================================================================== #
    #  Predict                                                             #
    # ================================================================== #

    def _raw_predict(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X, dtype=np.float64)
        out = np.full(X.shape[0], self.intercept_)
        for j in range(self.n_features_):
            bins = self._assign_bins(X[:, j], self.bin_edges_[j])
            out += self.shape_functions_[j][bins]
        return out

    def predict(self, X):
        raw = self._raw_predict(X)
        if self.task == "classification":
            return (raw > 0.0).astype(int)
        return raw

    def predict_proba(self, X):
        """(n, 2) class probabilities for binary classification."""
        p1 = self._sigmoid(self._raw_predict(X))
        return np.column_stack([1 - p1, p1])

    # ================================================================== #
    #  Interpretability                                                    #
    # ================================================================== #

    def explain_feature(self, feature_idx: int):
        """Return (bin_centers, shape_scores) for one feature."""
        edges = self.bin_edges_[feature_idx]
        centers = (edges[:-1] + edges[1:]) / 2.0
        return centers, self.shape_functions_[feature_idx].copy()

    def feature_importances(self, feature_names=None):
        """Mean |score| per feature, sorted descending."""
        k = len(self.shape_functions_)
        names = feature_names or [f"x{j}" for j in range(k)]
        imp = {
            n: float(np.mean(np.abs(sf)))
            for n, sf in zip(names, self.shape_functions_)
        }
        return dict(sorted(imp.items(), key=lambda kv: -kv[1]))

    def explain_prediction(self, x_single, feature_names=None):
        """Per-feature contribution for one observation."""
        x = np.asarray(x_single, dtype=np.float64).ravel()
        k = len(self.shape_functions_)
        names = feature_names or [f"x{j}" for j in range(k)]
        contribs = {"intercept": self.intercept_}
        for j in range(k):
            b = self._assign_bins(x[j: j + 1], self.bin_edges_[j])[0]
            contribs[names[j]] = float(self.shape_functions_[j][b])
        return contribs

    # ================================================================== #
    #  Post-hoc smoothing (optional)                                       #
    # ================================================================== #

    def smooth_shape_functions(self, window: int = 5):
        """Moving-average smoothing on every shape function."""
        kernel = np.ones(window) / window
        for j in range(len(self.shape_functions_)):
            sf = self.shape_functions_[j]
            if len(sf) < window:
                continue
            padded = np.pad(sf, window // 2, mode="edge")
            self.shape_functions_[j] = np.convolve(padded, kernel, mode="valid")[: len(sf)]

    def enforce_monotonicity(self, feature_idx: int, increasing: bool = True):
        """Isotonic regression (PAV) on one shape function — no data needed."""
        from sklearn.isotonic import IsotonicRegression

        sf = self.shape_functions_[feature_idx]
        iso = IsotonicRegression(increasing=increasing, out_of_bounds="clip")
        self.shape_functions_[feature_idx] = iso.fit_transform(np.arange(len(sf)), sf)




class DPExplainableBoostingMachine(ExplainableBoostingMachine):
    """
    Differentially Private EBM (Algorithm 2, Nori et al. 2021).

    Privacy-critical design decisions:
      - Leaf denominators use the *noisy* histogram counts from DP binning
      - The intercept is privatised via the Gaussian mechanism
      - label_range (R) must be supplied by the user for regression
      - Early stopping is disabled by default
      - Bagging is not supported (raises an error if n_bags > 1)

    Parameters (additional to base EBM)
    ------------------------------------
    epsilon : float
        Total (ε, δ)-DP privacy budget.
    delta : float
        δ parameter for (ε, δ)-DP.
    label_range : float | None
        R = max(y) - min(y), the range of labels.
        For classification with y ∈ {0,1}, R = 1 (set automatically).
        For regression, must be supplied by user for strict DP.
    bin_budget_fraction : float
        Fraction of ε allocated to binning (paper default = 0.10).
    intercept_budget_fraction : float
        Fraction of ε allocated to privatising the intercept.
    composition : str
        'gdp' for Gaussian DP composition (tighter) or
        'classic' for strong composition (Kairouz et al. 2017).
    """

    def __init__(
        self,
        n_bins: int = 32,
        max_rounds: int = 300,
        learning_rate: float = 0.01,
        max_leaves: int = 3,
        min_samples_leaf: int = 2,
        task: str = "regression",
        n_bags: int = 1,
        bag_fraction: float = 0.8,
        early_stopping_rounds: int | None = None,
        random_state: int | None = None,
        # ── DP-specific ──────────────────
        epsilon: float = 1.0,
        delta: float = 1e-6,
        label_range: float | None = None,
        bin_budget_fraction: float = 0.10,
        intercept_budget_fraction: float = 0.01,
        composition: str = "gdp",
    ):
        super().__init__(
            n_bins=n_bins,
            max_rounds=max_rounds,
            learning_rate=learning_rate,
            max_leaves=max_leaves,
            min_samples_leaf=min_samples_leaf,
            task=task,
            n_bags=n_bags,
            bag_fraction=bag_fraction,
            early_stopping_rounds=early_stopping_rounds,
            random_state=random_state,
        )
        self.epsilon = epsilon
        self.delta = delta
        self.label_range = label_range
        self.bin_budget_fraction = bin_budget_fraction
        self.intercept_budget_fraction = intercept_budget_fraction
        self.composition = composition

    # ================================================================== #
    #  GDP ↔ (ε,δ)-DP conversion (Theorem 5)                              #
    # ================================================================== #

    @staticmethod
    def _gdp_delta(mu: float, eps: float) -> float:
        """Convert µ-GDP → δ for a given ε (Theorem 5)."""
        return (
            _norm.cdf(-eps / mu + mu / 2)
            - np.exp(eps) * _norm.cdf(-eps / mu - mu / 2)
        )

    @staticmethod
    def _gdp_mu_from_eps_delta(eps: float, delta: float, tol: float = 1e-8) -> float:
        """Binary search: find µ such that µ-GDP ⟹ (ε,δ)-DP."""
        lo, hi = 1e-6, 1000.0
        for _ in range(200):
            mid = (lo + hi) / 2
            d = DPExplainableBoostingMachine._gdp_delta(mid, eps)
            if d < delta:
                lo = mid
            else:
                hi = mid
            if hi - lo < tol:
                break
        return (lo + hi) / 2

    @staticmethod
    def _classic_sigma(eps: float, delta: float, k: int, sensitivity: float) -> float:
        """σ from strong composition (Theorem 2, Kairouz et al. 2017)."""
        variance = 8 * k * sensitivity ** 2 * np.log(np.e + eps / delta) / eps ** 2
        return np.sqrt(variance)

    # ================================================================== #
    #  DP Quantile Binning (Algorithm 3)                                   #
    # ================================================================== #

    def _dp_quantile_bin_edges(
        self,
        x: np.ndarray,
        n_bins: int,
        sigma_bin: float,
        rng: np.random.Generator,
        x_min: float,
        x_max: float,
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Algorithm 3: DP quantile binning.
        1. Create 2·m equal-width bins using user-supplied min/max
        2. Add Gaussian noise to counts
        3. Greedily collapse small bins → approximate quantiles

        Returns
        -------
        edges : np.ndarray
            Bin edges (length = n_final_bins + 1).
        noisy_counts : np.ndarray
            The published DP histogram counts (length = n_final_bins).
            These are used as PUBLIC denominators during training.
        """
        if x_min >= x_max:
            return np.array([x_min, x_min + 1e-10]), np.array([float(len(x))])

        n_init = 2 * n_bins
        init_edges = np.linspace(x_min, x_max, n_init + 1)

        # True histogram counts on the equal-width grid
        counts = np.histogram(x, bins=init_edges)[0].astype(np.float64)

        # Add Gaussian noise (sensitivity = 1 per bin count)
        counts += sigma_bin * rng.standard_normal(len(counts))
        counts = np.maximum(counts, 0)  # clip negatives

        # Target samples per bin (using total noisy count)
        noisy_total = counts.sum()
        target = max(noisy_total / n_bins, 1.0)

        # Greedy collapse: merge adjacent bins until each is large enough
        merged_edges = [init_edges[0]]
        merged_counts = []
        running_count = 0.0

        for i in range(len(counts)):
            running_count += counts[i]
            if running_count >= target:
                merged_edges.append(init_edges[i + 1])
                merged_counts.append(running_count)
                running_count = 0.0

        # Handle remainder: collapse into last bin
        if running_count > 0:
            if len(merged_counts) > 0:
                merged_counts[-1] += running_count
            else:
                merged_counts.append(running_count)
            # Ensure final edge is the max
            if merged_edges[-1] < init_edges[-1]:
                merged_edges[-1] = init_edges[-1]

        # Ensure we always have at least the max edge
        if merged_edges[-1] < init_edges[-1]:
            merged_edges.append(init_edges[-1])

        edges = np.array(merged_edges)
        noisy_counts = np.array(merged_counts, dtype=np.float64)

        # Fallback: if collapse produced only 1 edge pair or mismatched
        if len(edges) < 2:
            edges = np.array([x_min, x_max])
            noisy_counts = np.array([max(noisy_total, 1.0)])

        # Ensure counts length matches bins
        n_final = len(edges) - 1
        if len(noisy_counts) != n_final:
            noisy_counts = np.full(n_final, max(noisy_total / n_final, 1.0))

        # Ensure no zero counts (would cause division by zero)
        noisy_counts = np.maximum(noisy_counts, 1.0)

        return edges, noisy_counts

    # ================================================================== #
    #  DP 1-D tree: uniform random splits + noisy leaf values              #
    # ================================================================== #

    def _fit_1d_tree_dp(
        self,
        bins: np.ndarray,
        residuals: np.ndarray,
        n_bins: int,
        dp_counts: np.ndarray,
        sigma_leaf: float,
        rng: np.random.Generator,
    ) -> np.ndarray:
        """
        DP version of the 1-D tree (Algorithm 2, lines 14-22):
          - Splits are chosen UNIFORMLY at random (data-agnostic, zero privacy cost)
          - Leaf values use noisy residual sums / DP counts (public denominators)
        """
        # ── Aggregate per-bin statistics (true residual sums) ─────────
        bin_sum = np.bincount(bins, weights=residuals, minlength=n_bins).astype(np.float64)

        # ── ✅ FIX #2: Uniformly random splits (paper line 14) ────────
        # "by choosing the splitting thresholds at random, we can learn
        #  the entire structure of each tree without looking at any
        #  training data" — Section 3, page 4
        if n_bins <= 1:
            split_points = []
        else:
            n_splits = min(self.max_leaves - 1, n_bins - 1)
            split_points = sorted(
                rng.choice(
                    np.arange(1, n_bins),
                    size=n_splits,
                    replace=False,
                )
            )

        # Build leaf boundaries: each leaf = [lo, hi) in bin indices
        boundaries = [0] + list(split_points) + [n_bins]
        leaves = [
            (boundaries[i], boundaries[i + 1])
            for i in range(len(boundaries) - 1)
        ]

        # ── Noisy leaf predictions ────────────────────────────────────
        output = np.zeros(n_bins, dtype=np.float64)
        R = self._label_range

        for lo, hi in leaves:
            # True sum of residuals in this leaf (scaled by learning rate)
            T = self.learning_rate * bin_sum[lo:hi].sum()

            # Add calibrated Gaussian noise (line 17 of Algo 2)
            # Sensitivity of T is η·R; noise = σ · η · R · N(0,1)
            T_noisy = T + sigma_leaf * self.learning_rate * R * rng.standard_normal()

            # Divide by DP count from the PUBLISHED histogram
            # This is PUBLIC information — no privacy cost
            leaf_dp_cnt = dp_counts[lo:hi].sum()

            if leaf_dp_cnt > 0:
                val = T_noisy / leaf_dp_cnt
            else:
                val = 0.0

            output[lo:hi] = val

        return output

    # ================================================================== #
    #  Calibrate noise σ                                                   #
    # ================================================================== #

    def _calibrate_sigma(self, n_features: int) -> float:
        """
        Determine σ for the training phase.

        GDP path (Theorem 7):
            Total mechanism is √(E·K)/σ - GDP.
            We want this to satisfy (ε_train, δ)-DP.
            → find µ from (ε_train, δ), then σ = √(E·K) / µ.

        Classic path (Theorem 2):
            σ from strong composition over E·K iterations.
        """
        eps_train = self.epsilon * (1 - self.bin_budget_fraction - self.intercept_budget_fraction)
        total_iterations = self.max_rounds * n_features

        if self.composition == "gdp":
            mu = self._gdp_mu_from_eps_delta(eps_train, self.delta)
            sigma = np.sqrt(total_iterations) / mu
        else:
            sigma = self._classic_sigma(
                eps_train, self.delta, total_iterations, sensitivity=1.0
            )
        return sigma

    # ================================================================== #
    #  Private intercept                                                   #
    # ================================================================== #

    def _private_intercept(
        self,
        y: np.ndarray,
        rng: np.random.Generator,
    ) -> float:
        """
        Compute the intercept with Gaussian noise.

        For regression: intercept = mean(y) + noise
            sensitivity of mean(y) = R / n
        For classification: intercept = log(p/(1-p)) where p = mean(y)
            We privatise mean(y) first, then transform.
            sensitivity of mean(y) = 1/n (since y ∈ {0,1})
        """
        n = len(y)
        eps_intercept = self.epsilon * self.intercept_budget_fraction

        if self.task == "classification":
            sensitivity = 1.0 / n
        else:
            sensitivity = (y.max() - y.min()) / n

        # Calibrate noise for the intercept (single mechanism, not composed)
        if self.composition == "gdp":
            mu = self._gdp_mu_from_eps_delta(eps_intercept, self.delta)
            sigma_intercept = sensitivity / mu
        else:
            sigma_intercept = self._classic_sigma(
                eps_intercept, self.delta, k=1, sensitivity=sensitivity
            )

        noisy_mean = float(y.mean()) + sigma_intercept * rng.standard_normal()

        if self.task == "classification":
            p = np.clip(noisy_mean, 1e-10, 1 - 1e-10)
            return float(np.log(p / (1 - p)))
        else:
            return float(noisy_mean)

    # ================================================================== #
    #  Label range (sensitivity bound R)                                   #
    # ================================================================== #

    def _get_label_range(self, y: np.ndarray) -> float:
        """
        ✅ FIX #1: R is the range of y_i, NOT the range of residuals.

        From Theorem 6 proof (paper pages 4-5):
            T = η · (Σ y_i) - Z
            where Z is computed from publicly released f_k^{t-1} values.
            Z has zero sensitivity. Therefore sensitivity of T depends
            only on the range of y_i.

        For classification:
            y_i ∈ {0, 1} → R = 1
        For regression:
            y_i ∈ [y_min, y_max] → R = y_max - y_min
        """
        if self.task == "classification":
            return 1.0  # ✅ FIX: was 2.0, should be 1.0 per Theorem 6

        # For regression
        if self.label_range is not None:
            return float(self.label_range)  # ✅ FIX: was 2.0 * label_range

        # Fallback: compute from data (not strictly DP)
        import warnings
        warnings.warn(
            "label_range not provided for DP regression. "
            "For rigorous privacy, supply label_range explicitly. "
            "Using data range as approximation.",
            UserWarning,
        )
        return float(y.max() - y.min())

    # ================================================================== #
    #  Fit (override)                                                      #
    # ================================================================== #

    def fit(self, X, y):
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64).ravel()
        n_samples, n_features = X.shape
        rng = np.random.default_rng(self.random_state)

        # ── Validate: no bagging in DP mode ───────────────────────────
        if self.n_bags > 1:
            raise ValueError(
                "Bagging is not supported for DP-EBMs. Each bag would "
                "require its own privacy budget. Set n_bags=1."
            )

        # ── ✅ FIX #1: Label range R (for sensitivity calculation) ────
        self._label_range = self._get_label_range(y)

        if self.task == "classification":
            assert abs(self._label_range - 1.0) < 1e-6, \
                f"Classification label range should be 1.0, got {self._label_range}"

        # ── Budget split ──────────────────────────────────────────────
        eps_bin = self.epsilon * self.bin_budget_fraction

        # σ for binning noise (composed across K features via GDP)
        if self.composition == "gdp":
            mu_bin = self._gdp_mu_from_eps_delta(eps_bin, self.delta)
            sigma_bin = np.sqrt(n_features) / mu_bin
        else:
            sigma_bin = self._classic_sigma(eps_bin, self.delta, n_features, 1.0)

        # σ for training leaf noise
        sigma_leaf = self._calibrate_sigma(n_features)

        # ── Feature min/max (assumed public or user-supplied) ─────────
        feature_mins = X.min(axis=0)
        feature_maxs = X.max(axis=0)

        # ── 1. DP Quantile binning (Algorithm 3) ─────────────────────
        self.bin_edges_: list[np.ndarray] = []
        self.n_bins_actual_: list[int] = []
        self.dp_counts_: list[np.ndarray] = []
        X_binned = np.empty((n_samples, n_features), dtype=np.intp)

        for j in range(n_features):
            edges, noisy_counts = self._dp_quantile_bin_edges(
                X[:, j], self.n_bins, sigma_bin, rng,
                x_min=float(feature_mins[j]),
                x_max=float(feature_maxs[j]),
            )
            self.bin_edges_.append(edges)
            nb = max(len(edges) - 1, 1)
            self.n_bins_actual_.append(nb)
            X_binned[:, j] = self._assign_bins(X[:, j], edges)

            # Store the NOISY counts from DP binning as public information
            self.dp_counts_.append(noisy_counts)

        # ── 2. Private intercept ──────────────────────────────────────
        self.intercept_ = self._private_intercept(y, rng)

        # ── 3. DP Cyclic boosting ─────────────────────────────────────
        shape_fns = [np.zeros(nb) for nb in self.n_bins_actual_]
        raw = np.full(n_samples, self.intercept_)

        best_loss = np.inf
        stale = 0
        self.loss_history_: list[float] = []

        for rnd in range(self.max_rounds):
            for j in range(n_features):
                residuals = self._negative_gradient(y, raw)
                bins_j = X_binned[:, j]
                nb = self.n_bins_actual_[j]

                step = self._fit_1d_tree_dp(
                    bins_j, residuals, nb,
                    dp_counts=self.dp_counts_[j],
                    sigma_leaf=sigma_leaf,
                    rng=rng,
                )

                shape_fns[j] += step
                raw += step[bins_j]

            cur_loss = self._loss(y, raw)
            self.loss_history_.append(cur_loss)

            if cur_loss < best_loss - 1e-10:
                best_loss = cur_loss
                stale = 0
            else:
                stale += 1

            if self.early_stopping_rounds and stale >= self.early_stopping_rounds:
                break

        self.shape_functions_ = shape_fns
        self.n_features_ = n_features
        self.n_rounds_fitted_ = rnd + 1
        self.sigma_leaf_ = sigma_leaf
        self.sigma_bin_ = sigma_bin

        return self

    # ================================================================== #
    #  Privacy report                                                      #
    # ================================================================== #

    def privacy_report(self) -> dict:
        """Summary of the privacy guarantees."""
        eps_train = self.epsilon * (1 - self.bin_budget_fraction - self.intercept_budget_fraction)
        return {
            "epsilon": self.epsilon,
            "delta": self.delta,
            "composition": self.composition,
            "epsilon_binning": self.epsilon * self.bin_budget_fraction,
            "epsilon_intercept": self.epsilon * self.intercept_budget_fraction,
            "epsilon_training": eps_train,
            "sigma_binning": float(self.sigma_bin_),
            "sigma_training": float(self.sigma_leaf_),
            "total_iterations_budgeted": self.max_rounds * self.n_features_,
            "rounds_fitted": self.n_rounds_fitted_,
            "label_range_R": self._label_range,
        }