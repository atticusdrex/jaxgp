from .util import * 


"""
gplib.likelihood
------------------
Negative log-likelihood objective functions used for hyperparameter
training. Each function follows the signature `obj(model, p, ...)` and
returns a scalar loss value.
"""

def neg_mll(model, p):
    """Negative marginal log-likelihood for a standard GP.

    Parameters
    ----------
    model : object
        GP-like object implementing `get_L`, `.Y`, `.X`, `.mean` and `.N`.
    p : dict
        Parameter dictionary used to construct the kernel/mean/noise.
    """
    # Getting cholesky factors and solve linear system
    L = model.get_L(p['k_param'], p['noise_var'])
    Ytilde = model.Y - model.mean.eval(model.X, p['m_param']) 
    quad_term = jnp.inner(Ytilde, cho_solve((L, True), Ytilde))
    logdet_term = 2.0*jnp.sum(jnp.log(jnp.diag(L)))
    constant_term = model.N * jnp.log(2*math.pi)
    # Return quadratic and log-determinant components
    return 0.5*(quad_term + logdet_term + constant_term)


def delta_neg_mll(model, p):
    """Negative marginal log-likelihood for DeltaGP (difference) models."""
    # Getting cholesky factors and solve linear system
    L = model.get_L(p['k_param'], p['noise_var'])
    # Center the Y vector 
    Ytilde = model.Y1 - p['rho'] * model.Y2 - model.mean.eval(model.X, p['m_param']) 
    quad_term = jnp.inner(Ytilde, cho_solve((L, True), Ytilde))
    logdet_term = 2.0*jnp.sum(jnp.log(jnp.diag(L)))
    constant_term = model.N * jnp.log(2*math.pi)
    # Return quadratic and log-determinant components
    return 0.5*(quad_term + logdet_term + constant_term)


def cokriging_neg_mll(model, p):
    """Negative marginal log-likelihood for block cokriging models."""
    L = model.get_L(p)
    # Centering the training data
    Ytilde = model.Yfull - model.mean_train(p)
    # Creating the likelihood terms
    quad_term = jnp.inner(Ytilde, cho_solve((L, True), Ytilde))
    logdet_term = 2.0 * jnp.sum(jnp.log(jnp.diag(L)))
    constant_term = model.N_total * jnp.log(2*math.pi)
    return 0.5 * (quad_term + logdet_term + constant_term)


def svgp_neg_mll(model, p, N_mc = 25, seed = 42):
    """Monte-Carlo estimate of the SVGP negative log-likelihood + KL term.

    This objective estimates the expected squared error under the
    variational posterior and adds the KL divergence between the
    variational and prior distributions of the inducing variables.
    """
    # Getting cholesky factors and solve linear system
    L = model.get_L(p['Z'], p['k_param'], p['noise_var'])
    # Forming testing kernel matrices 
    Ktest = K(model.X, p['Z'], model.kernel, p['k_param'])
    Kaux = K(model.X, model.X, model.kernel, p['k_param'])
    # computing posterior covariance cholesky factor
    L_pos = cholesky(Kaux + Ktest @ cho_solve((L, True), Ktest.T) + model.eps * jnp.eye(model.N))
    # computing the softplus noise variance 
    noise_var = softplus(p['noise_var'])

    # Function for computing the quadratic term from a single sample 
    def squared_error(key):
        # Splitting the rng keys into two 
        key1, key2 = random.split(key, num = 2)
        # sampling from variational distribution 
        u = p['q_mu'] + p['q_L'] @ random.normal(key1, shape = (model.M)) 
        # computing posterior distribution  
        mu_pos = (Ktest @ cho_solve((L, True), u)).ravel() + model.mean.eval(model.X, p['m_param'])
        # sampling from the posterior 
        Y_pos = mu_pos + L_pos @ random.normal(key2, shape = (model.N))
        # returning the squared error 
        return jnp.sum((model.Y - Y_pos)**2)

    # Generating RNG keys 
    keys = random.split(random.key(seed), num = N_mc)

    # Computing squared errors
    squared_errors = vmap(squared_error)(keys)
    squared_error = jnp.mean(squared_errors) / (noise_var)

    # computing the kl_divergence
    kl_divergence = KL_div(p['q_mu'], p['q_L'], model.mean.eval(p['Z'], p['m_param']), L)

    # computing the constant term 
    constant_term = model.N * jnp.log(2*jnp.pi*noise_var)

    # returning the sum 
    return 0.5*(squared_error + constant_term) + kl_divergence



