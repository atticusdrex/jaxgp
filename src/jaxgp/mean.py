from .util import * 

"""
gplib.mean
-----------
Simple mean function implementations used by GP wrappers.

The module defines a small family of mean functions expected to
implement two methods:
  - calibrate(X, Y) -> 1D array of parameters
  - eval(X, params) -> evaluations at inputs X
"""

# The base mean function 
class Mean:
    """Base class for mean functions.

    Subclasses should set `self.p_dim` to indicate the number of
    parameters and implement `calibrate` and `eval`.
    """
    def __init__(self, input_dim, epsilon=1e-8):
        self.input_dim = input_dim
        self.eps = epsilon

class Linear(Mean):
    """Linear mean: m(x) = b + w^T x.

    Parameters
    ----------
    p_dim : int
        1 + input_dim (bias + weights)
    """
    def __init__(self, *args, **kwargs):
        # Initializing parent class 
        super().__init__(*args, **kwargs)

        # Storing parameter dimension 
        self.p_dim = 1 + self.input_dim
    
    def calibrate(self, X, Y):
        """Fit linear mean by ordinary least squares with Tikhonov regularization."""
        Phi = jnp.hstack((jnp.ones((X.shape[0],1)), X))
        return jnp.linalg.solve(Phi.T @ Phi + self.eps * jnp.eye(Phi.shape[1]), Phi.T @ Y)
        
    def eval(self, X, params):
        """Evaluate linear mean at rows of X using parameter vector params."""
        return (params[0] * jnp.ones((X.shape[0], 1)) + X @ params[1:].reshape(-1,1)).ravel()

# Constant mean function
class Constant(Mean):
    """Constant mean: m(x) = c where c is a scalar parameter."""
    def __init__(self, *args, **kwargs):
        # Initializing parent class 
        super().__init__(*args, **kwargs)

        # Storing parameter dimension 
        self.p_dim = 1
    
    def calibrate(self, X, Y):
        """Return empirical mean of Y as the constant parameter."""
        return jnp.mean(Y[:]) * jnp.ones(self.p_dim)
    
    def eval(self, X, params):
        """Evaluate constant mean at inputs X."""
        return (params[0] * jnp.ones((X.shape[0], 1))).ravel()

# Zero mean function 
class Zero(Mean):
    """Zero mean function (no learnable parameters)."""
    def __init__(self, *args, **kwargs):
        # Initializing parent class 
        super().__init__(*args, **kwargs)

        # Storing parameter dimension 
        self.p_dim = self.input_dim
    
    def calibrate(self, X, Y):
        """Return zero parameters (no-op calibration)."""
        return jnp.zeros(self.p_dim)
    
    def eval(self, X, params):
        """Evaluate zero mean at inputs X."""
        return np.zeros(X.shape[0])