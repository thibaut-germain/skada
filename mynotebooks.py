# %%
import numpy as np
import matplotlib.pyplot as plt
from typing import Callable,Union,Literal

from skada.datasets import make_shifted_datasets
from skada import source_target_split
from skada.utils import check_X_domain, check_X_y_domain, extract_source_indices
from skada._pipeline import make_da_pipeline

from sklearn.utils.validation import check_is_fitted
from sklearn.model_selection import ParameterGrid

from skada._reweight import BaseReweightAdapter
from sklearn.metrics import pairwise_kernels

# %%
class ULSIFAdapter(BaseReweightAdapter): 

    def __init__(self,
            # kernel:Union[Callable, Literal["chi2", "rbf", "laplacian"]]='rbf',
            kernel="rbf",
            kernel_grid:dict=None, 
            reg_grid:np.ndarray=None,
            n_jobs:int=1,
            n_kernels:int=100,
            random_seed=None
        ):
        super().__init__()
        self.kernel = kernel
        self.kernel_grid = kernel_grid
        self.reg_grid = reg_grid
        self.n_jobs = n_jobs
        self.n_kernels = n_kernels
        self.random_seed = random_seed

    def _choose_centers(self):
        nt = self.Xt_.shape[0]
        if self.n_kernels == -1:
            self.n_kernels = nt
        centers_id = self.rng_.choice(nt, size=min(self.n_kernels, nt), replace=False)
        return centers_id

    def _compute_H_and_h(self, Xs, Xt, Xc, n_jobs, **kwds):
        Ys = pairwise_kernels(Xs, Xc, metric=self.kernel, filter_params=False, n_jobs=n_jobs, **kwds)
        Yt = pairwise_kernels(Xt, Xc, metric=self.kernel, filter_params=False, n_jobs=n_jobs, **kwds)
        H = np.sum(Ys.T[:, None, :] * Ys.T[None, :, :], axis=-1)
        h = np.sum(Yt, axis=0)
        return Ys, Yt, H, h
    
    def _compute_LOOCV(self, lmbda, Ys, Yt, H, h):
        ns = Ys.shape[0]
        nt = Yt.shape[0]
        n = min(ns, nt)
        if ns < nt:
            Ys_tronc = Ys.T
            tronc_id = self.rng_.choice(nt, size=n, replace=False)
            Yt_tronc = Yt[tronc_id].T
        else:
            Yt_tronc = Yt.T
            tronc_id = self.rng_.choice(ns, size=n, replace=False)
            Ys_tronc = Ys[tronc_id].T
        b = H.shape[0]
        B = H + lmbda * (ns - 1) / ns * np.eye(b)
        B_inv = np.linalg.inv(B)
        denom = nt * np.ones(n) - np.sum(Ys_tronc * (B_inv @ Ys_tronc), axis=0)
        num0 = h @ B_inv @ Ys_tronc
        num1 = np.sum(Yt_tronc * (B_inv @ Ys_tronc), axis=0)
        B_inv_Y = B_inv @ Ys_tronc
        B0 = np.repeat(B_inv @ h[:, None], repeats=n, axis=1) + B_inv_Y * (num0 / denom)
        B1 = B_inv @ Yt_tronc + B_inv_Y * (num1 / denom)
        B2 = np.maximum(0, (ns - 1) / (ns * (nt - 1)) * (nt * B0 - B1))
        
        ws = np.sum(Ys_tronc * B2, axis=0)
        wt = np.sum(Yt_tronc * B2, axis=0)
        
        score = np.sum(ws ** 2) / (2 * n) - np.sum(wt) / n
        return score

    def _hyperparameter_selection(self):
        """Estimate the optimal hyperparameters 'kernel_parameters' and 'regularization'."""
        Xc = self.Xt_[self.centers_id_]
        outer_grid = ParameterGrid(self.kernel_grid)
        scores = np.zeros((len(outer_grid), len(self.reg_grid)))
        for i, kernel_parameters in enumerate(outer_grid):
            Ys, Yt, H, h = self._compute_H_and_h(self.Xs_, self.Xt_, Xc, self.n_jobs, **kernel_parameters)
            for j, lmbda in enumerate(self.reg_grid):
                scores[i, j] = self._compute_LOOCV(lmbda, Ys, Yt, H, h)
        best_kernel_parameters_id, best_regularization_id = np.unravel_index(scores.argmin(), scores.shape)
        best_kernel_parameters = outer_grid[best_kernel_parameters_id]
        best_regularization = self.reg_grid[best_regularization_id]
        return best_kernel_parameters, best_regularization

    def _compute_coefs_and_weights(self):
        Xc = self.Xt_[self.centers_id_]
        Ys, _, H, h = self._compute_H_and_h(self.Xs_, self.Xt_, Xc, self.n_jobs, **self.best_kernel_parameters_)
        alpha = np.maximum(0, np.linalg.inv(H + self.best_regularization_ * np.eye(H.shape[0])) @ h)
        weights = Ys @ alpha
        return weights, alpha

    def fit(self, X, y=None, sample_domain=None):
        X, sample_domain = check_X_domain(X, sample_domain)
        self.Xs_, self.Xt_, _, _ = source_target_split(X, y, sample_domain=sample_domain)
        self.rng_ = np.random.default_rng(seed=self.random_seed)
        
        self.centers_id_ = self._choose_centers()
        self.best_kernel_parameters_, self.best_regularization_ = self._hyperparameter_selection()
        self.source_weights_, self.alpha_ = self._compute_coefs_and_weights()

    def compute_weights(self, X, y=None, *, sample_domain=None):
        check_is_fitted(self, 'alpha_')
        X, y, sample_domain = check_X_y_domain(X, y, sample_domain=sample_domain)
        source_id = extract_source_indices(sample_domain)
        if np.array_equal(self.Xs_, X[source_id]):
            source_weights = self.source_weights_
        else:
            evals = pairwise_kernels(X[source_id], self.Xt_[self.centers_id_], metric=self.kernel, filter_params=False, n_jobs=self.n_jobs, **self.best_kernel_parameters_)
            source_weights = evals @ self.alpha_
        weights = np.zeros(X.shape[0], dtype=source_weights.dtype)
        weights[source_id] = source_weights
        return weights

def ULSIF(
    base_estimator=None,
    # kernel:Union[Callable, Literal["chi2", "rbf", "laplacian"]]='rbf',
    kernel="rbf",
    kernel_grid:dict=None, 
    reg_grid:np.ndarray=None,
    n_jobs:int=1,
    n_kernels:int=100,
    random_seed=None
):
    if base_estimator is None:
        base_estimator = LogisticRegression().set_fit_request(sample_weight=True)
    return make_da_pipeline(
        ULSIFAdapter(
            kernel=kernel,
            kernel_grid=kernel_grid,
            reg_grid=reg_grid,
            n_jobs=n_jobs,
            n_kernels=n_kernels,
            random_seed=random_seed,
        ),
        base_estimator,
    )

# %%
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap
from sklearn.inspection import DecisionBoundaryDisplay
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KernelDensity

from skada import (
    DensityReweight,
    DiscriminatorReweight,
    GaussianReweight,
    KLIEPReweight,
    KMMReweight,
    NearestNeighborReweight,
    source_target_split,
)
from skada.datasets import make_shifted_datasets
from skada.utils import extract_source_indices

RANDOM_SEED = 42

X, y, sample_domain = make_shifted_datasets(
    n_samples_source=20, n_samples_target=20, noise=0.1, random_state=RANDOM_SEED
)

Xs, Xt, ys, yt = source_target_split(X, y, sample_domain=sample_domain)

x_min, x_max = -2.5, 4.5
y_min, y_max = -1.5, 4.5


figsize = (8, 4)
figure, axes = plt.subplots(1, 2, figsize=figsize)

cm = plt.cm.RdBu
colormap = ListedColormap(["#FFA056", "#6C4C7C"])
ax = axes[0]
ax.set_title("Source data")
# Plot the source points:
ax.scatter(Xs[:, 0], Xs[:, 1], c=ys, cmap=colormap, alpha=0.7, s=[25])

ax.set_xticks(()), ax.set_yticks(())
ax.set_xlim(x_min, x_max), ax.set_ylim(y_min, y_max)

ax = axes[1]

ax.set_title("Target data")
# Plot the target points:
ax.scatter(Xt[:, 0], Xt[:, 1], c=ys, cmap=colormap, alpha=0.1, s=[25])
ax.scatter(Xt[:, 0], Xt[:, 1], c=yt, cmap=colormap, alpha=0.7, s=[25])
figure.suptitle("Plot of the dataset", fontsize=16, y=1)
ax.set_xticks(()), ax.set_yticks(())
ax.set_xlim(x_min, x_max), ax.set_ylim(y_min, y_max)

# %%
scores_dict = {}

def plot_weights_and_classifier(
    clf,
    weights,
    name="Without DA",
    suptitle=None,
):
    if suptitle is None:
        suptitle = f"Illustration of the {name} method"
    figure, axes = plt.subplots(1, 2, figsize=figsize)
    ax = axes[1]
    score = clf.score(Xt, yt)
    DecisionBoundaryDisplay.from_estimator(
        clf,
        Xs,
        cmap=ListedColormap(["w", "k"]),
        alpha=1,
        ax=ax,
        eps=0.5,
        response_method="predict",
        plot_method="contour",
    )

    size = 5 + 10 * weights

    # Plot the target points:
    ax.scatter(
        Xt[:, 0],
        Xt[:, 1],
        c=yt,
        cmap=colormap,
        alpha=0.7,
        s=[25],
    )

    ax.set_xticks(()), ax.set_yticks(())
    ax.set_xlim(x_min, x_max), ax.set_ylim(y_min, y_max)
    ax.set_title("Accuracy on target", fontsize=12)
    ax.text(
        x_max - 0.3,
        y_min + 0.3,
        ("%.2f" % score).lstrip("0"),
        size=15,
        horizontalalignment="right",
    )
    scores_dict[name] = score

    ax = axes[0]

    # Plot the source points:
    ax.scatter(Xs[:, 0], Xs[:, 1], c=ys, cmap=colormap, alpha=0.7, s=size)

    DecisionBoundaryDisplay.from_estimator(
        clf,
        Xs,
        cmap=ListedColormap(["w", "k"]),
        alpha=1,
        ax=ax,
        eps=0.5,
        response_method="predict",
        plot_method="contour",
    )

    ax.set_xticks(()), ax.set_yticks(())
    ax.set_xlim(x_min, x_max), ax.set_ylim(y_min, y_max)
    if name != "Without DA":
        ax.set_title("Training with reweighted data", fontsize=12)
    else:
        ax.set_title("Training data", fontsize=12)
    figure.suptitle(suptitle, fontsize=16, y=1)


base_classifier = LogisticRegression().set_fit_request(sample_weight=True)
clf = base_classifier
clf.fit(Xs, ys)
plot_weights_and_classifier(
    base_classifier,
    name="Without DA",
    weights=np.array([2] * Xs.shape[0]),
    suptitle="Illustration of the classifier with no DA",
)

# %%
kernel_grid = {"gamma": np.logspace(-2, 2, 5)}
reg_grid = np.logspace(-4, 0, 5)
ulsif = ULSIF(kernel_grid=kernel_grid, reg_grid=reg_grid, random_seed=42)

# %%
ulsif.fit(X, y, sample_domain=sample_domain)

# %%
# We define our classifier, `clf` is a da pipeline


# %%
# We get the weights:

# we first get the adapter which is estimating the weights
weight_estimator = ulsif[0].get_estimator()
idx = extract_source_indices(sample_domain)
weights = weight_estimator.compute_weights(X, sample_domain=sample_domain)[idx]

plot_weights_and_classifier(ulsif, weights=weights, name="Density Reweighting")

# %%
