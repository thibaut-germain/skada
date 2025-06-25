# %%
import numpy as np
import matplotlib.pyplot as plt
from typing import Callable,Union,Literal

from skada.datasets import make_shifted_datasets
from skada import source_target_split
from skada.utils import check_X_domain,check_X_y_domain

from sklearn.utils import check_is_fitted


# %%
# Get DA dataset
X, y, sample_domain = make_shifted_datasets(
    20, 20, shift="conditional_shift", random_state=42
)

# split source and target for visualization
Xs, Xt, ys, yt = source_target_split(X, y, sample_domain=sample_domain)
sample_domain_s = np.ones(Xs.shape[0])
sample_domain_t = -np.ones(Xt.shape[0]) * 2

# plot data
plt.figure(1, (10, 5))

plt.subplot(1, 2, 1)
plt.scatter(Xs[:, 0], Xs[:, 1], c=ys, cmap="tab10", vmax=9, label="Source")
plt.title("Source data")
ax = plt.axis()

plt.subplot(1, 2, 2)
plt.scatter(Xt[:, 0], Xt[:, 1], c=yt, cmap="tab10", vmax=9, label="Target")
plt.axis(ax)
plt.title("Target data")

# %%
from skada._reweight import BaseReweightAdapter
from sklearn.metrics import pairwise_kernels 

# %%
class KernelLSIF(BaseReweightAdapter): 

    def __init__(self,
            kernel:Union[Callable,Literal["chi2","rbf","laplacian"]]='rbf',
            kernel_grid:dict = None, 
            reg_grid:np.ndarray = None,
            njobs:int = 1,
            max_control_points:int = 100,
            random_seed = None
        ):
        super().__init__()
        self.kernel = kernel 
        self.kernel_grid = kernel_grid
        self.reg_grid = reg_grid
        self.njobs = njobs
        self.max_control_points = max_control_points
        self.random_seed = random_seed

    def fit(self,X,y=None,sample_domain=None): 
        # Preprocess data
        X, sample_domain = check_X_domain(X, sample_domain)
        Xs, Xt, _, _ = source_target_split(X, y, sample_domain=sample_domain)
        ns = Xs.shape[0]
        nt = Xt.shape[0]
        n = min(ns,nt)

        # set random generator
        rng  = np.random.default_rng(seed = self.random_seed)
        if self.max_control_points == -1:
            self.max_control_points = nt
        mask_id = rng.choice(min(self.max_control_points,nt),replace=False)
        self.control_mask_ = np.full(nt,False)
        self.control_mask_[mask_id] = True

        # self._weights_optimization(Xs,Xt,**self._get_fit_parameters(Xs,Xt))

        self.fitted_ = True

        
    def _compute_matrices(self,Xs,Xt,njobs,**kwds):
        # compute 
        Ys = pairwise_kernels(Xs,Xt)
        Yt = pairwise_kernels(Xt,None,self.kernel,False,njobs, **kwds)
        h = np.sum(Yt,axis=1)
        H = np.sum(Ys[:,None,:] * Ys[None,:,:],axis=-1)
        return Ys,Yt,H,h
    
    def _compute_final_ponderation(self,Xs,Xt,njobs):
        if (not self.best_kernel_parameters is None) * (not self.best_regularization is None):
            _, _, H, h = self._compute_matrices(Xs,Xt,njobs,**self.best_kernel_parameters)
            # compute kernels ponderation
            self.alpha_ = np.maximum(0,np.linalg.inv(H + self.best_regularization * np.eye(H.shape[0])) @ h)

            


    def _parameter_selection_(self): 
        # trouver kernel_parameters et regularisation
        ### Toutes les combinaisons de kernel_parameters et regularization
        self.best_kernel_parameters = None
        self.best_regularization = None

    def _unconstrained_fit(self,X,y=None,sample_domain=None): 

        self.kernel_parmaters,regularization = self._parameter_selection(...)

        # Get the importance parameter (alpha)
        pass

    def compute_weights(self,X,y=None,*,sample_domain=None, **kwds):
        check_is_fitted(self, 'alpha_')
        X,y,sample_domain = check_X_y_domain(X, y, sample_domain=sample_domain)

        return lambda x: np.sum(...) #w function





#[‘additive_chi2’, ‘chi2’, ‘linear’, ‘poly’, ‘polynomial’, ‘rbf’, ‘laplacian’, ‘sigmoid’, ‘cosine’]

# %%
X = np.arange(12).reshape(4,3)
Y = np.arange(9).reshape(3,3)
pairwise_kernels(X,Y).shape

# %%
