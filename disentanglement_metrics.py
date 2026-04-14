"""
Disentanglement metrics for evaluating latent representation quality.

Implements metrics from:
    "Challenging Common Assumptions in the Unsupervised Learning of
     Disentangled Representations" (Locatello et al., 2019)

Metrics:
    - FactorVAE metric  (Kim & Mnih 2018) – variance-based majority voting
    - DCI Disentanglement (Eastwood & Williams 2018) – gradient boosted trees
    - MIG (Chen et al. 2018) – Mutual Information Gap
    - Modularity (Ridgeway & Mozer 2018)
    - SAP score (Kumar et al. 2017) – linear SVM
    - Total Correlation
    - Latent Correlation Matrix (visual)

Usage example::

    from disentanglement_metrics import compute_all_metrics

    scores = compute_all_metrics(
        model=clap_model.pred_vae,        # or clap_model.cl_vae
        dataset=test_dataset,             # dataset with .factor_data attribute
        n_samples=10_000,
        use_mean=True,
        device="cpu",
    )
    print(scores)
"""

from __future__ import annotations

import logging
from typing import Any, Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Utility helpers
# ---------------------------------------------------------------------------


def _to_numpy(t: torch.Tensor) -> np.ndarray:
    return t.detach().cpu().numpy()


# ---------------------------------------------------------------------------
# Representation extraction
# ---------------------------------------------------------------------------


def _get_representations(
    model: nn.Module,
    dataloader: DataLoader,
    use_mean: bool,
    device: torch.device,
    model_key: Optional[str] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Encode a dataset and return (representations, factors).

    Parameters
    ----------
    model:
        A ``PredictionVAE`` or ``ConceptLearningVAE`` (or any module whose
        ``forward`` returns a dict with keys ``mean_core``, ``log_var_core``,
        ``mean_style``, ``log_var_style``).  If the full CLAP model is passed
        use *model_key* to select the sub-dict (``"pred"`` or ``"cl"``).
    dataloader:
        DataLoader that yields ``(x, factors)`` pairs.
    use_mean:
        When ``True`` the posterior mean is used as the representation;
        otherwise a single reparameterised sample is drawn.
    device:
        Torch device.
    model_key:
        ``"pred"`` or ``"cl"`` when the full CLAP model is passed.

    Returns
    -------
    representations : np.ndarray, shape ``(n, z_dim)``
    factors : np.ndarray, shape ``(n, n_factors)``
    """
    model.eval()
    all_reps: list[np.ndarray] = []
    all_factors: list[np.ndarray] = []

    with torch.no_grad():
        for batch in dataloader:
            x, fac = batch
            x = x.to(device)
            fac_np = fac.cpu().numpy() if isinstance(fac, torch.Tensor) else np.array(fac)

            # Support CLAP model (forward needs y) and single-VAE models.
            try:
                out = model(x)
            except TypeError:
                # ConceptLearningVAE needs y – pass zeros when factors are
                # not multi-label (evaluation only needs encoder outputs).
                y_dummy = torch.zeros(x.size(0), dtype=torch.float, device=device)
                out = model(x, y_dummy)

            # If this is the full CLAP model the output is nested.
            if model_key is not None:
                out = out[model_key]

            mean = torch.cat([out["mean_core"], out["mean_style"]], dim=-1)
            log_var = torch.cat([out["log_var_core"], out["log_var_style"]], dim=-1)

            if use_mean:
                rep = mean
            else:
                std = torch.exp(0.5 * log_var)
                rep = mean + std * torch.randn_like(std)

            all_reps.append(_to_numpy(rep))

            if fac_np.ndim == 1:
                fac_np = fac_np[:, np.newaxis]
            all_factors.append(fac_np)

    representations = np.concatenate(all_reps, axis=0)
    factors = np.concatenate(all_factors, axis=0)
    return representations, factors


# ---------------------------------------------------------------------------
# Mutual information helpers (binning-based estimator)
# ---------------------------------------------------------------------------


def _discrete_entropy(labels: np.ndarray) -> float:
    """Shannon entropy of a discrete distribution estimated from *labels*."""
    _, counts = np.unique(labels, return_counts=True)
    probs = counts / counts.sum()
    return float(-np.sum(probs * np.log(probs + 1e-10)))


def _discretize(values: np.ndarray, n_bins: int = 20) -> np.ndarray:
    """Bin continuous values into *n_bins* equal-width buckets."""
    mins = values.min(axis=0)
    maxs = values.max(axis=0)
    # Avoid degenerate bins for constant features.
    ranges = np.where(maxs - mins > 0, maxs - mins, 1.0)
    binned = np.floor((values - mins) / ranges * n_bins).astype(np.int32)
    binned = np.clip(binned, 0, n_bins - 1)
    return binned


def _mutual_information_matrix(
    representations: np.ndarray,
    factors: np.ndarray,
    n_bins: int = 20,
) -> np.ndarray:
    """Estimate mutual information I(z_i; v_k) via binning.

    Returns
    -------
    mi_matrix : np.ndarray, shape ``(z_dim, n_factors)``
    """
    n, z_dim = representations.shape
    n_factors = factors.shape[1]

    # Discretise continuous latents.
    z_disc = _discretize(representations, n_bins)

    mi_matrix = np.zeros((z_dim, n_factors))
    for j in range(z_dim):
        for k in range(n_factors):
            z_j = z_disc[:, j]
            v_k = factors[:, k].astype(np.int32)

            h_zj = _discrete_entropy(z_j)
            h_vk = _discrete_entropy(v_k)

            # Joint entropy H(z_j, v_k)
            joint = z_j * (v_k.max() + 1) + v_k
            h_joint = _discrete_entropy(joint)

            mi_matrix[j, k] = max(0.0, h_zj + h_vk - h_joint)

    return mi_matrix


# ---------------------------------------------------------------------------
# 1. MIG – Mutual Information Gap
# ---------------------------------------------------------------------------


def mig_score(
    representations: np.ndarray,
    factors: np.ndarray,
    n_bins: int = 20,
) -> float:
    """Compute the MIG score (Chen et al. 2018).

    .. math::
        \\text{MIG} = \\frac{1}{K} \\sum_k \\frac{1}{H(v_k)}
        \\left[ I(z_{j^*}; v_k) - \\max_{j \\neq j^*} I(z_j; v_k) \\right]

    where :math:`j^* = \\arg\\max_j I(z_j; v_k)`.

    Parameters
    ----------
    representations : np.ndarray, shape (n, z_dim)
    factors : np.ndarray, shape (n, n_factors)  – integer-valued
    n_bins : int
        Number of bins used when discretising continuous latent dimensions.

    Returns
    -------
    float in [0, 1]
    """
    mi = _mutual_information_matrix(representations, factors, n_bins)
    n_factors = factors.shape[1]
    gaps = []
    for k in range(n_factors):
        mi_k = mi[:, k]
        sorted_mi = np.sort(mi_k)[::-1]
        h_vk = _discrete_entropy(factors[:, k].astype(np.int32))
        if h_vk < 1e-8:
            continue
        gap = (sorted_mi[0] - sorted_mi[1]) / h_vk
        gaps.append(gap)
    return float(np.mean(gaps)) if gaps else 0.0


# ---------------------------------------------------------------------------
# 2. Modularity
# ---------------------------------------------------------------------------


def modularity_score(
    representations: np.ndarray,
    factors: np.ndarray,
    n_bins: int = 20,
) -> float:
    """Compute the Modularity score (Ridgeway & Mozer 2018).

    For each latent dimension :math:`z_j`, compute its MI with every factor
    and check whether it depends on at most one factor.

    Returns
    -------
    float in [0, 1]
    """
    mi = _mutual_information_matrix(representations, factors, n_bins)  # (z_dim, K)
    z_dim, n_factors = mi.shape

    scores = []
    for j in range(z_dim):
        mi_j = mi[j, :]
        total = mi_j.sum()
        if total < 1e-10:
            scores.append(1.0)
            continue
        mi_j_norm = mi_j / total  # normalise to sum to 1

        # Perfect modularity: exactly one factor has all the MI.
        # Measure deviation from that: Gini-like measure.
        sorted_mi = np.sort(mi_j_norm)[::-1]
        if n_factors == 1:
            scores.append(1.0)
            continue
        # Score = (max - mean_rest) / (max - 1/K) normalised
        max_mi = sorted_mi[0]
        # Ideal: max_mi = 1 → deviation = 0
        # Worst (uniform): max_mi = 1/K
        ideal_delta = 1.0 - 1.0 / n_factors
        actual_delta = max_mi - 1.0 / n_factors
        score = actual_delta / ideal_delta if ideal_delta > 0 else 1.0
        scores.append(max(0.0, min(1.0, score)))

    return float(np.mean(scores))


# ---------------------------------------------------------------------------
# 3. SAP score – Separated Attribute Predictability
# ---------------------------------------------------------------------------


def sap_score(
    representations: np.ndarray,
    factors: np.ndarray,
    continuous_factors: bool = False,
) -> float:
    """Compute the SAP score (Kumar et al. 2017).

    For each generative factor :math:`v_k`, trains a linear SVM (or linear
    regressor for continuous factors) using each single latent dimension and
    returns the average gap between the top-2 single-code predictors.

    Parameters
    ----------
    representations : np.ndarray, shape (n, z_dim)
    factors : np.ndarray, shape (n, n_factors)
    continuous_factors : bool
        If ``True`` use linear regression R² instead of SVM accuracy.

    Returns
    -------
    float
    """
    from sklearn.svm import LinearSVC
    from sklearn.linear_model import Ridge
    from sklearn.model_selection import cross_val_score
    from sklearn.preprocessing import label_binarize

    n, z_dim = representations.shape
    n_factors = factors.shape[1]

    score_matrix = np.zeros((z_dim, n_factors))
    for k in range(n_factors):
        v_k = factors[:, k]
        for j in range(z_dim):
            z_j = representations[:, j].reshape(-1, 1)
            if continuous_factors:
                reg = Ridge(alpha=0.01)
                cv_scores = cross_val_score(reg, z_j, v_k, cv=5, scoring="r2")
                score_matrix[j, k] = max(0.0, cv_scores.mean())
            else:
                n_classes = len(np.unique(v_k))
                if n_classes < 2:
                    score_matrix[j, k] = 1.0
                    continue
                clf = LinearSVC(C=0.01, max_iter=2000, dual=True)
                cv_scores = cross_val_score(clf, z_j, v_k, cv=5, scoring="accuracy")
                score_matrix[j, k] = max(0.0, cv_scores.mean())

    gaps = []
    for k in range(n_factors):
        col = np.sort(score_matrix[:, k])[::-1]
        if len(col) >= 2:
            gaps.append(col[0] - col[1])
    return float(np.mean(gaps)) if gaps else 0.0


# ---------------------------------------------------------------------------
# 4. FactorVAE metric
# ---------------------------------------------------------------------------


def factorvae_metric(
    representations: np.ndarray,
    factors: np.ndarray,
    n_bins: int = 20,
    n_train: int = 800,
) -> float:
    """Compute the FactorVAE metric (Kim & Mnih 2018).

    Trains a majority-vote classifier that predicts which latent dimension
    has the lowest normalised variance when one generative factor is fixed.

    The representation array is used as a pre-encoded dataset: for each
    factor k, we repeatedly pick two random factor values, find samples
    with those values, compute the variance of each latent across those
    samples (normalised by overall variance), and vote for the dimension
    with smallest variance.

    Parameters
    ----------
    representations : np.ndarray, shape (n, z_dim)
    factors : np.ndarray, shape (n, n_factors)  – integer-valued
    n_bins : int
        Ignored (kept for API consistency).
    n_train : int
        Number of training votes per factor.

    Returns
    -------
    float in [0, 1]
    """
    n, z_dim = representations.shape
    n_factors = factors.shape[1]

    # Global variance across all samples for normalisation.
    global_var = representations.var(axis=0) + 1e-10  # (z_dim,)

    votes = np.zeros((z_dim, n_factors), dtype=np.int32)  # vote[j, k]

    rng = np.random.default_rng(42)

    for k in range(n_factors):
        v_k = factors[:, k]
        unique_vals = np.unique(v_k)
        if len(unique_vals) < 2:
            continue
        for _ in range(n_train):
            # Pick a fixed value of factor k.
            fixed_val = rng.choice(unique_vals)
            idx = np.where(v_k == fixed_val)[0]
            if len(idx) < 2:
                continue
            # Sample a small batch from those indices.
            batch_size = min(64, len(idx))
            chosen = rng.choice(idx, size=batch_size, replace=False)
            batch = representations[chosen]
            # Normalised variance for each latent.
            norm_var = batch.var(axis=0) / global_var  # (z_dim,)
            winning_dim = int(np.argmin(norm_var))
            votes[winning_dim, k] += 1

    # For each factor, the winning latent dimension.
    # Accuracy = fraction of votes that match the majority.
    if votes.sum() == 0:
        return 0.0

    # Accuracy: fraction of training votes for the plurality winner.
    total_votes = votes.sum()
    majority_votes = votes.max(axis=0).sum()
    return float(majority_votes / total_votes)


# ---------------------------------------------------------------------------
# 5. DCI Disentanglement (Eastwood & Williams 2018)
# ---------------------------------------------------------------------------


def dci_scores(
    representations: np.ndarray,
    factors: np.ndarray,
) -> Dict[str, float]:
    """Compute DCI Disentanglement, Completeness, and Informativeness.

    Uses gradient-boosted trees (sklearn GBM) as the function class.

    Parameters
    ----------
    representations : np.ndarray, shape (n, z_dim)
    factors : np.ndarray, shape (n, n_factors) – integer-valued

    Returns
    -------
    dict with keys ``"disentanglement"``, ``"completeness"``,
    ``"informativeness_train"``, ``"informativeness_test"``
    """
    from sklearn.ensemble import GradientBoostingClassifier
    from sklearn.model_selection import train_test_split as tts

    n, z_dim = representations.shape
    n_factors = factors.shape[1]

    # Split.
    train_x, test_x, train_f, test_f = tts(
        representations, factors, test_size=0.2, random_state=42
    )

    # Importance matrix R[j, k] = importance of latent j for predicting factor k.
    importance_matrix = np.zeros((z_dim, n_factors))
    train_errors: list[float] = []
    test_errors: list[float] = []

    for k in range(n_factors):
        y_train = train_f[:, k]
        y_test = test_f[:, k]
        n_classes = len(np.unique(y_train))
        if n_classes < 2:
            importance_matrix[:, k] = 1.0 / z_dim
            train_errors.append(0.0)
            test_errors.append(0.0)
            continue
        clf = GradientBoostingClassifier(
            n_estimators=10,
            max_depth=2,
            random_state=42,
        )
        clf.fit(train_x, y_train)
        importance_matrix[:, k] = clf.feature_importances_  # (z_dim,)
        train_errors.append(1.0 - clf.score(train_x, y_train))
        test_errors.append(1.0 - clf.score(test_x, y_test))

    # --- Disentanglement D ---
    # For each latent j, how much does it encode *one* factor?
    # D_j = (1 - H(P_j)) where P_j is the probability distribution over factors
    # weighted by R[j, :].
    # H computed using the Gini coefficient equivalent from the paper.
    def _entropy_from_weights(w: np.ndarray) -> float:
        total = w.sum()
        if total < 1e-10:
            return 0.0
        p = w / total
        return float(-np.sum(p * np.log(p + 1e-10)))

    d_scores: list[float] = []
    for j in range(z_dim):
        w = importance_matrix[j, :]
        if w.sum() < 1e-10:
            d_scores.append(0.0)
            continue
        # Normalise by maximum possible entropy.
        h = _entropy_from_weights(w)
        h_max = np.log(n_factors) if n_factors > 1 else 1.0
        d_j = 1.0 - h / h_max if h_max > 0 else 1.0
        d_scores.append(max(0.0, d_j))

    # Weight D scores by importance of each latent (across all factors).
    latent_importance = importance_matrix.sum(axis=1)
    latent_importance_norm = latent_importance / (latent_importance.sum() + 1e-10)
    disentanglement = float(np.dot(d_scores, latent_importance_norm))

    # --- Completeness C ---
    # For each factor k, how much is it encoded in a *single* latent?
    c_scores: list[float] = []
    for k in range(n_factors):
        w = importance_matrix[:, k]
        if w.sum() < 1e-10:
            c_scores.append(0.0)
            continue
        h = _entropy_from_weights(w)
        h_max = np.log(z_dim) if z_dim > 1 else 1.0
        c_k = 1.0 - h / h_max if h_max > 0 else 1.0
        c_scores.append(max(0.0, c_k))

    completeness = float(np.mean(c_scores))

    # --- Informativeness I ---
    informativeness_train = float(1.0 - np.mean(train_errors))
    informativeness_test = float(1.0 - np.mean(test_errors))

    return {
        "disentanglement": disentanglement,
        "completeness": completeness,
        "informativeness_train": informativeness_train,
        "informativeness_test": informativeness_test,
    }


# ---------------------------------------------------------------------------
# 6. Total Correlation (estimated from VAE posterior parameters)
# ---------------------------------------------------------------------------


def total_correlation_from_params(
    mean: np.ndarray,
    log_var: np.ndarray,
) -> float:
    """Estimate Total Correlation from VAE posterior parameters.

    Uses the closed-form approximation from Burgess et al. / Chen et al.:

    .. math::
        TC(z) \\approx \\frac{1}{n} \\sum_i \\left[
            \\sum_j \\log q(z_{ij} | x_i) - \\log q(z_i | x_i)
        \\right]

    where :math:`q(z_i | x_i) \\approx \\frac{1}{n}\\sum_m q(z_i | x_m)`
    (minibatch-weighted estimate).

    For scalability this implementation uses a random sub-sample to
    estimate :math:`\\log q(\\mathbf{z})`.

    Parameters
    ----------
    mean : np.ndarray, shape (n, z_dim)
    log_var : np.ndarray, shape (n, z_dim)

    Returns
    -------
    float  (lower is more disentangled)
    """
    n, z_dim = mean.shape
    var = np.exp(log_var)

    # Sub-sample for computational efficiency.
    max_samples = min(n, 2000)
    rng = np.random.default_rng(0)
    idx = rng.choice(n, max_samples, replace=False)
    mean_s, var_s = mean[idx], var[idx]

    # log q(z_i | x_i) = sum_j log N(z_ij; mu_ij, sigma^2_ij)
    # evaluated at z_i = mu_i (mean representation).
    def _log_gaussian(z: np.ndarray, mu: np.ndarray, v: np.ndarray) -> np.ndarray:
        """Log N(z; mu, v) shape (batch_z, batch_mu, z_dim)."""
        return -0.5 * (np.log(2 * np.pi) + np.log(v) + (z - mu) ** 2 / v)

    # log q(z_i | x_i): for each sample i, sum over dims.
    log_qz_given_xi = _log_gaussian(mean_s, mean_s, var_s).sum(axis=-1)  # (m,)

    # Marginal log q(z_j): for each sample i, use all m samples to estimate.
    # log q(z_ij) ≈ log mean_m N(z_ij; mu_mj, var_mj)
    # Shape broadcasting: (m, 1, z_dim) vs (1, m, z_dim)
    log_qzij = _log_gaussian(
        mean_s[:, np.newaxis, :],  # (m, 1, z_dim)
        mean_s[np.newaxis, :, :],  # (1, m, z_dim)
        var_s[np.newaxis, :, :],   # (1, m, z_dim)
    )  # (m, m, z_dim)

    from scipy.special import logsumexp as _logsumexp

    # log mean over m  →  logsumexp(axis=1) - log(m)
    # Shape: (m, z_dim)
    log_qzij_marginal = (
        _logsumexp(log_qzij, axis=1) - np.log(max_samples)  # (m, z_dim)
    )

    # log q(z_i): joint across dims (assuming marginals factorised).
    # Aggregate over m for estimate of log q(z_i).
    # log_qzij.sum(axis=-1): (m, m) — sum over z_dims gives log joint
    log_qzi = (
        _logsumexp(log_qzij.sum(axis=-1), axis=1) - np.log(max_samples)  # (m,)
    )

    # TC = E[log q(z) - sum_j log q(z_j)]
    tc = np.mean(log_qzi - log_qzij_marginal.sum(axis=-1))
    return float(tc)


def total_correlation(
    representations: np.ndarray,
    factors: np.ndarray,
) -> float:
    """Estimate Total Correlation using the covariance structure of representations.

    This is a simpler approximation based on the Gaussian TC formula:

    .. math::
        TC \\approx \\frac{1}{2} \\left[
            \\sum_j \\log \\text{Var}(z_j) - \\log \\det \\Sigma
        \\right]

    where :math:`\\Sigma` is the full covariance of ``representations``.

    Parameters
    ----------
    representations : np.ndarray, shape (n, z_dim)
    factors : np.ndarray  – unused, kept for API consistency.

    Returns
    -------
    float  (lower is better)
    """
    cov = np.cov(representations.T)  # (z_dim, z_dim)
    if cov.ndim == 0:
        return 0.0
    marginal_entropies = 0.5 * np.sum(np.log(np.diag(cov) + 1e-10))
    sign, logdet = np.linalg.slogdet(cov + 1e-6 * np.eye(cov.shape[0]))
    if sign <= 0:
        return 0.0
    return float(marginal_entropies - 0.5 * logdet)


# ---------------------------------------------------------------------------
# 7. Latent Correlation Matrix
# ---------------------------------------------------------------------------


def latent_correlation_matrix(
    representations: np.ndarray,
) -> np.ndarray:
    """Compute Pearson correlation matrix of latent dimensions.

    Parameters
    ----------
    representations : np.ndarray, shape (n, z_dim)

    Returns
    -------
    corr : np.ndarray, shape (z_dim, z_dim)
    """
    return np.corrcoef(representations.T)


# ---------------------------------------------------------------------------
# Visualization
# ---------------------------------------------------------------------------


def plot_latent_correlation_matrix(
    corr: np.ndarray,
    title: str = "Latent Correlation Matrix",
    ax: Optional[Any] = None,
) -> Any:
    """Plot the latent correlation matrix as a heatmap.

    Parameters
    ----------
    corr : np.ndarray, shape (z_dim, z_dim)
    title : str
    ax : optional matplotlib Axes

    Returns
    -------
    The matplotlib Axes object.
    """
    import matplotlib.pyplot as plt

    if ax is None:
        _, ax = plt.subplots(figsize=(8, 6))

    im = ax.imshow(corr, vmin=-1, vmax=1, cmap="RdBu_r", aspect="auto")
    plt.colorbar(im, ax=ax)
    ax.set_title(title)
    ax.set_xlabel("Latent dimension")
    ax.set_ylabel("Latent dimension")
    return ax


def plot_mi_heatmap(
    representations: np.ndarray,
    factors: np.ndarray,
    factor_names: Optional[list[str]] = None,
    title: str = "Mutual Information (latent × factor)",
    ax: Optional[Any] = None,
    n_bins: int = 20,
) -> Any:
    """Plot mutual information between latents and factors.

    Parameters
    ----------
    representations : np.ndarray, shape (n, z_dim)
    factors : np.ndarray, shape (n, n_factors)
    factor_names : list of str, optional
    title : str
    ax : optional matplotlib Axes
    n_bins : int

    Returns
    -------
    The matplotlib Axes object.
    """
    import matplotlib.pyplot as plt

    mi = _mutual_information_matrix(representations, factors, n_bins)  # (z, K)

    if ax is None:
        _, ax = plt.subplots(figsize=(max(4, mi.shape[1]), max(4, mi.shape[0] // 2)))

    im = ax.imshow(mi, aspect="auto", cmap="Blues")
    plt.colorbar(im, ax=ax, label="MI (nats)")
    ax.set_title(title)
    ax.set_xlabel("Generative factor")
    ax.set_ylabel("Latent dimension")
    if factor_names is not None:
        ax.set_xticks(range(len(factor_names)))
        ax.set_xticklabels(factor_names, rotation=45, ha="right")
    return ax


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------


def compute_all_metrics(
    model: nn.Module,
    dataset: Dataset,
    n_samples: int = 10_000,
    use_mean: bool = True,
    device: Optional[str] = None,
    batch_size: int = 256,
    n_bins: int = 20,
    model_key: Optional[str] = None,
    continuous_factors: bool = False,
    plot: bool = False,
    factor_names: Optional[list[str]] = None,
) -> Dict[str, Any]:
    """Compute all disentanglement metrics for a model and dataset.

    Parameters
    ----------
    model : nn.Module
        Encoder model.  Should be a ``PredictionVAE``, ``ConceptLearningVAE``,
        or the full CLAP model.  The forward pass must return (or contain, when
        *model_key* is given) a dict with ``mean_core``, ``log_var_core``,
        ``mean_style``, ``log_var_style``.
    dataset : torch.utils.data.Dataset
        Dataset that yields ``(x, factors)`` tuples where ``factors`` are
        integer-valued generative factor indices.
    n_samples : int
        Number of samples to draw from the dataset.
    use_mean : bool
        Use posterior mean (``True``) or a single sample (``False``).
    device : str, optional
        Torch device string (e.g. ``"cpu"``, ``"cuda"``).  Defaults to CUDA
        if available, else CPU.
    batch_size : int
        Batch size for the encoding pass.
    n_bins : int
        Number of bins for MI estimation.
    model_key : str, optional
        ``"pred"`` or ``"cl"`` when passing the full CLAP model.
    continuous_factors : bool
        Pass ``True`` if generative factors are continuous (uses regression in
        SAP score instead of SVM classification).
    plot : bool
        If ``True``, display correlation matrix and MI heatmap.
    factor_names : list of str, optional
        Human-readable names for the generative factors (used in plots).

    Returns
    -------
    dict
        Keys: ``"mig"``, ``"modularity"``, ``"sap"``, ``"factorvae"``,
        ``"dci_disentanglement"``, ``"dci_completeness"``,
        ``"dci_informativeness_train"``, ``"dci_informativeness_test"``,
        ``"total_correlation"``, ``"latent_correlation_matrix"``
    """
    if device is None:
        device_obj = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device_obj = torch.device(device)

    model = model.to(device_obj)

    # Sub-sample the dataset.
    n_total = len(dataset)
    if n_samples < n_total:
        indices = np.random.default_rng(0).choice(n_total, n_samples, replace=False)
        from torch.utils.data import Subset as _Subset
        dataset = _Subset(dataset, indices.tolist())

    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)

    logger.info("Extracting representations (%s, use_mean=%s)…", device_obj, use_mean)
    representations, factors = _get_representations(
        model, loader, use_mean=use_mean, device=device_obj, model_key=model_key
    )

    logger.info("representations: %s, factors: %s", representations.shape, factors.shape)

    # Ensure factors are integer-valued for discrete metrics.
    factors_int = factors.astype(np.int32)

    results: Dict[str, Any] = {}

    logger.info("Computing MIG…")
    results["mig"] = mig_score(representations, factors_int, n_bins=n_bins)

    logger.info("Computing Modularity…")
    results["modularity"] = modularity_score(representations, factors_int, n_bins=n_bins)

    logger.info("Computing SAP score…")
    results["sap"] = sap_score(
        representations, factors_int, continuous_factors=continuous_factors
    )

    logger.info("Computing FactorVAE metric…")
    results["factorvae"] = factorvae_metric(representations, factors_int, n_bins=n_bins)

    logger.info("Computing DCI scores…")
    dci = dci_scores(representations, factors_int)
    results["dci_disentanglement"] = dci["disentanglement"]
    results["dci_completeness"] = dci["completeness"]
    results["dci_informativeness_train"] = dci["informativeness_train"]
    results["dci_informativeness_test"] = dci["informativeness_test"]

    logger.info("Computing Total Correlation…")
    results["total_correlation"] = total_correlation(representations, factors_int)

    logger.info("Computing Latent Correlation Matrix…")
    corr = latent_correlation_matrix(representations)
    results["latent_correlation_matrix"] = corr

    if plot:
        try:
            import matplotlib.pyplot as plt

            fig, axes = plt.subplots(1, 2, figsize=(16, 6))
            plot_latent_correlation_matrix(corr, ax=axes[0])
            plot_mi_heatmap(
                representations,
                factors_int,
                factor_names=factor_names,
                ax=axes[1],
                n_bins=n_bins,
            )
            plt.tight_layout()
            plt.show()
        except Exception as exc:
            logger.warning("Plotting failed: %s", exc)

    return results


# ---------------------------------------------------------------------------
# Multi-model comparison utility
# ---------------------------------------------------------------------------


def compare_models(
    models: Dict[str, nn.Module],
    dataset: Dataset,
    **kwargs: Any,
) -> Dict[str, Dict[str, Any]]:
    """Compute and compare disentanglement metrics across multiple models.

    Parameters
    ----------
    models : dict mapping name → model
        Example: ``{"CL-VAE": clap.cl_vae, "Pred-VAE": clap.pred_vae}``
    dataset : Dataset
    **kwargs
        Extra keyword arguments forwarded to :func:`compute_all_metrics`.

    Returns
    -------
    dict mapping model name → metrics dict
    """
    all_results: Dict[str, Dict[str, Any]] = {}
    for name, model in models.items():
        logger.info("Evaluating model '%s'…", name)
        all_results[name] = compute_all_metrics(model, dataset, **kwargs)

    _print_comparison_table(all_results)
    return all_results


def _print_comparison_table(results: Dict[str, Dict[str, Any]]) -> None:
    """Print a formatted comparison table of scalar metrics."""
    scalar_keys = [
        "mig",
        "modularity",
        "sap",
        "factorvae",
        "dci_disentanglement",
        "dci_completeness",
        "dci_informativeness_test",
        "total_correlation",
    ]
    model_names = list(results.keys())
    col_w = max(max(len(n) for n in model_names), 12)
    metric_w = 30

    header = f"{'Metric':<{metric_w}}" + "".join(f"{n:>{col_w}}" for n in model_names)
    print(header)
    print("-" * len(header))
    for k in scalar_keys:
        row = f"{k:<{metric_w}}"
        for name in model_names:
            val = results[name].get(k, float("nan"))
            row += f"{val:>{col_w}.4f}"
        print(row)
