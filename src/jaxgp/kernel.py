from .util import * 

"""
gplib.kernel
-------------
Kernel covariance functions used by the GP wrappers.

Provided kernels implement a lightweight API:
  - `p_dim` integer (number of kernel parameters)
  - `calibrate(X, Y)` to produce a sensible initial parameter vector
  - `eval(x, y, params)` which evaluates the scalar kernel k(x,y)
"""

# The parent kernel class which the other kernels inherit 
class Kernel:
    """Base kernel class.

    Subclasses should set `self.p_dim` and implement `calibrate` and
    `eval`.
    """
    def __init__(self, input_dim, epsilon = 1e-8):
        self.input_dim = input_dim # Storing kernel input dimension 
        self.eps = epsilon # Storing kernel epsilon 

# Automatic Relevancy Determination kernel 
class RBF(Kernel):
    """Isotropic RBF kernel with per-dimension lengthscales.

    Parameterization (params) is expected to be a 1D array where
    params[0] is a signal variance and params[1:] are lengthscales for
    each input dimension (positive after softplus).
    """
    def __init__(self, *args, **kwargs):
        # Calling super class 
        super().__init__(*args, **kwargs)
        # Storing parameter dimension 
        self.p_dim = 1 + self.input_dim
    
    def calibrate(self, X, Y):        
        """Return a reasonable initial parameter vector from data."""
        return inv_softplus(self.eps + jnp.concat((jnp.var(Y.ravel()).reshape(1), jnp.var(jnp.diff(Y)) * jnp.ones(self.p_dim-1)), axis=0))
        
    # Evaluation function 
    def eval(self, x, y, params):
        """Evaluate RBF kernel k(x,y) given raw `params` (unconstrained).

        The function applies `softplus` to `params` to obtain positive
        variances/lengthscales.
        """
        h = (x-y).ravel()
        # Enforcing positivity and boundedness
        params = softplus(params)
        # computing the kernel eval
        return params[0]*jnp.exp(-jnp.sum(h**2 / params[1:]))
    
# Automatic Relevancy Determination kernel 
class Laplace(Kernel):
    """Isotropic RBF kernel with per-dimension lengthscales.

    Parameterization (params) is expected to be a 1D array where
    params[0] is a signal variance and params[1:] are lengthscales for
    each input dimension (positive after softplus).
    """
    def __init__(self, *args, **kwargs):
        # Calling super class 
        super().__init__(*args, **kwargs)
        # Storing parameter dimension 
        self.p_dim = 1 + self.input_dim
    
    def calibrate(self, X, Y):        
        """Return a reasonable initial parameter vector from data."""
        return inv_softplus(self.eps + jnp.concat((jnp.var(Y.ravel()).reshape(1), jnp.var(jnp.diff(Y)) * jnp.ones(self.p_dim-1)), axis=0))
        
    # Evaluation function 
    def eval(self, x, y, params):
        """Evaluate RBF kernel k(x,y) given raw `params` (unconstrained).

        The function applies `softplus` to `params` to obtain positive
        variances/lengthscales.
        """
        h = (x-y).ravel()
        # Enforcing positivity and boundedness
        params = softplus(params)
        # computing the kernel eval
        return params[0]*jnp.exp(-jnp.sum(jnp.abs(h) / params[1:]))
    

# Automatic Relevancy Determination kernel 
class NARGP_RBF(Kernel):
    """Composite kernel used by NARGP-style autoregressive GPs.

    This kernel composes several RBF kernels to operate over extended
    state vectors where the last element may represent an autoregressive
    component.
    """
    def __init__(self, *args, **kwargs):
        # Calling super class 
        super().__init__(*args, **kwargs)
        # Storing parameter dimension 
        self.p_dim = 2*(1 + self.input_dim)
        # Making a list of RBF kernels 
        self.kernels = [RBF(*args, **kwargs), RBF(*args, **kwargs), RBF(1, **kwargs)]
    
    def calibrate(self, X, Y):        
        return inv_softplus(self.eps + jnp.concat((jnp.var(Y.ravel()).reshape(1), jnp.var(jnp.diff(Y)) * jnp.ones(self.p_dim-1)), axis=0))
        
    # Evaluation function 
    def eval(self, x1, x2, params):
        """Evaluate composite NARGP kernel between `x1` and `x2`.

        The function expects `x1` and `x2` to be extended input vectors
        where the last entry corresponds to the autoregressive scalar
        and the remaining entries are the standard input coordinates.
        """
        # Extracting the coordinates 
        y, x = x1[-1], x1[:-1]
        yp, xp = x2[-1], x2[:-1]
        # Extracting parameters 
        d = len(x)
        kx, ky, kd = params[:d+1], params[d+1:d+3], params[d+3:]
        # Computing without constraints 
        return self.kernels[0].eval(x, xp, kx) * self.kernels[1].eval(y, yp, ky) + self.kernels[2].eval(x, xp, kd)
    

