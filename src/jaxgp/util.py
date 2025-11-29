import jax 
import jax.numpy as jnp
from jax import vmap, value_and_grad, random, jit
from jax.scipy.linalg import cho_solve, cholesky

from copy import deepcopy
from tqdm import tqdm 
import numpy as np
import math
from jaxopt import LBFGS as JaxoptLBFGS
from scipy.stats import qmc


# 64-bit 
try:
    jax.config.update("jax_enable_x64", True)
except:
    print("64-bit Jax Computation is not available on your CPU.")


"""
gplib.util
-----------
Small utility helpers used across the GP wrappers. Functions include
activation/transform helpers (sigmoid, softplus), kernel batching
helpers and a KL divergence for multivariate Gaussians parameterized
by Cholesky factors.
"""

def sigmoid(x):
    """Sigmoid activation elementwise: 1 / (1 + exp(-x))."""
    return 1.0 / (1.0 + jnp.exp(-x))

def inv_sigmoid(y):
    """Inverse sigmoid (logit): maps (0,1) -> R."""
    return jnp.log(y/(1-y))

def softplus(x):
    """Softplus activation: log(1 + exp(x))."""
    return jnp.log(1.0 + jnp.exp(x))

def inv_softplus(y):
    """Inverse softplus: maps positive values back to unconstrained space."""
    return jnp.log(jnp.exp(y) - 1.0)


def K(X1, X2, kernel, kernel_params):
    """Compute the full kernel matrix between two point sets.

    Returns an array with shape (len(X1), len(X2)) where each entry is
    kernel.eval(x,y,kernel_params).
    """
    return vmap(lambda x: vmap(lambda y: kernel.eval(x, y, kernel_params))(X2))(X1)

# For batching the training data
def create_batches(X, Y, batch_size, shuffle=True):
    """Yield minibatches of (X, Y).

    If `shuffle` is True the dataset is shuffled before batching.
    """
    n_samples = X.shape[0]
    
    if shuffle:
        # Shuffle indices
        indices = np.random.permutation(n_samples)
        X = X[indices]
        Y = Y[indices]
    
    # Yield batches
    for i in range(0, n_samples, batch_size):
        X_batch = X[i:i + batch_size, :]
        Y_batch = Y[i:i + batch_size]
        yield X_batch, Y_batch

# Function for greedily choosing the number of inducing inputs 
def greedy_k_center(X, k, seed=42):
    """Greedy k-centers selection of inducing inputs.

    Returns a tuple (selected_points, selected_indices).
    """
    np.random.seed(seed)
    N = X.shape[0]
    selected_indices = []
    idx = np.random.randint(N)
    selected_indices.append(idx)

    distances = np.linalg.norm(X - X[idx], axis=1)

    for _ in range(1, k):
        idx = np.argmax(distances)
        selected_indices.append(idx)
        new_distances = np.linalg.norm(X - X[idx], axis=1)
        distances = np.minimum(distances, new_distances)

    return X[np.array(selected_indices)], selected_indices

# Special KL-divergence function
def KL_div(mu_q, L_q, mu_p, L_p):
    """KL divergence KL(q || p) for Gaussians parameterized by Cholesky factors.

    Arguments:
      mu_q, L_q : mean and lower-triangular Cholesky factor for q
      mu_p, L_p : mean and lower-triangular Cholesky factor for p
    """
    k = mu_q.shape[0]

    # Covariance matrices
    Sigma_q = L_q @ L_q.T
    Sigma_p = L_p @ L_p.T

    # Trace term: tr(Sigma_p^{-1} Sigma_q)
    # Solve instead of explicitly inverting
    Sigma_p_inv = jnp.linalg.inv(Sigma_p)
    Tr_q = jnp.trace(Sigma_p_inv @ Sigma_q)

    # Mean term: (mu_p - mu_q)^T Sigma_p^{-1} (mu_p - mu_q)
    diff = mu_q - mu_p
    mean_term = jnp.inner(diff, cho_solve((L_p, True), diff))

    # Log-determinant ratio
    logdet_q = 2.0 * jnp.sum(jnp.log(jnp.diag(L_q)))
    logdet_p = 2.0 * jnp.sum(jnp.log(jnp.diag(L_p)))
    logdet_ratio = logdet_p - logdet_q

    return 0.5 * (Tr_q + mean_term - k + logdet_ratio)