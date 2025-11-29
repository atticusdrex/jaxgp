from .likelihood import * 

"""
gplib.optim
-----------
Optimizers and initialization helpers for GP hyperparameter training.

Exposes three optimizer wrappers:
 - Momentum: SGD with momentum and several random initializers
 - ADAM: Adam optimizer wrapper
 - LBFGS: jaxopt-backed L-BFGS wrapper

Each optimizer expects a model with a `.p` dict of parameters and a
`.set_params()` method to apply new parameter values.
"""

class Momentum:
    """Momentum-based optimizer with convenient initialization helpers.

    Methods provided include several parameter initializers (latin
    hypercube, log-normal, Poisson) and a `run` method that performs
    gradient updates with momentum on the selected parameter keys.
    """
    def __init__(self, model, objective_func, beta = 0.9, constraints = None):
        # Storing model and objective function
        self.model, self.obj, self.beta = model, objective_func, beta

        # Storing loss function
        self.grad_fn = value_and_grad(jit(lambda p: self.obj(self.model, p)))

        # Compute initial loss
        self.best_loss, _ = self.grad_fn(model.p)
        assert not jnp.isnan(self.best_loss), "Initial Loss is NaN!"

        # Store best params
        self.best_p = deepcopy(model.p)

        # Store parameter constraints 
        self.constraints = constraints 
    
    def latin_hypercube_init(self, param, k, min=-50.0, max=50.0, seed = 42):
        """
        param: the parameter to optimize
        k: number of samples
        min: array of length d with min values
        max: array of length d with max values
        """
        # Getting kernel parameter dimension 
        d = self.model.kernel.p_dim 

        # Getting the dimension of the kernel parameters 
        sampler = qmc.LatinHypercube(d=d, seed = seed)

        # Generate samples in [0,1]^d
        X = sampler.random(n=k)

        # Scale to [mins, maxs]
        X_scaled = qmc.scale(X, min*np.ones(d), max*np.ones(d))

        def sample(this_sample):
            # Copy model parameters 
            p = deepcopy(self.model.p)

            # Replace kernel hyperparameters
            p[param] = jnp.array(this_sample)

            # Returning objective at this parameter value
            return self.obj(self.model, p)

        # Get the losses at each kernel parameter 
        losses = np.array(vmap(jit(sample))(X_scaled))
        losses[np.isnan(losses)] = 1e99
        best_loss = np.min(losses)

        # Pick the kernel parameters that incurred the lowest loss
        best_p = X_scaled[np.argmin(losses),:]

        # Printing the best loss 
        print("Best Objective Value: %.4e" % (best_loss))

        # Storing the best parameters
        if best_loss < self.best_loss: 
            p = deepcopy(self.model.p)
            p[param] = best_p 
            self.model.set_params(p)

            self.best_loss, self.best_p = best_loss, p

    def log_normal_init(self, param, k, mean=0.0, std = 1.0, seed = 42):
        """
        param: the parameter to optimize
        k: number of samples
        min: array of length d with min values
        max: array of length d with max values
        """
        # Getting kernel parameter dimension 
        d = self.model.kernel.p_dim 

        # getting mean and covariance matrices 
        mean = mean * jnp.ones(d)
        cov = std**2 * jnp.eye(d)

        # Sampling from log-normal distribution
        np.random.seed(seed)
        X = np.exp(np.random.multivariate_normal(mean, cov, k))

        def sample(this_sample):
            # Copy model parameters 
            p = deepcopy(self.model.p)

            # Replace kernel hyperparameters
            p[param] = inv_softplus(this_sample)

            # Evaluate objective function 
            return self.obj(self.model, p)

        # Get the losses at each kernel parameter 
        losses = np.array(vmap(jit(sample))(X))
        losses[np.isnan(losses)] = 1e99
        best_loss = np.min(losses)

        # Pick the kernel parameters that incurred the lowest loss
        best_p = X[np.argmin(losses),:]

        # Printing the best loss 
        print("Best Objective Value: %.4e" % (best_loss))

        # Storing the best parameters
        if best_loss < self.best_loss: 
            p = deepcopy(self.model.p)
            p[param] = best_p 
            self.model.set_params(p)
            self.best_loss, self.best_p = best_loss, p

    def poisson_init(self, param, k, lam=0.1, seed = 42):
        """
        param: the parameter to optimize
        k: number of samples
        min: array of length d with min values
        max: array of length d with max values
        """
        # Getting kernel parameter dimension 
        d = self.model.kernel.p_dim 

        # Sampling from log-normal distribution
        np.random.seed(seed)
        X = np.exp(np.random.poisson(lam=lam, size=(k, d)))

        def sample(this_sample):
            # Copy model parameters 
            p = deepcopy(self.model.p)

            # Replace kernel hyperparameters
            p[param] = inv_softplus(this_sample)

            # Evaluate objective function 
            return self.obj(self.model, p)

        # Get the losses at each kernel parameter 
        losses = np.array(vmap(jit(sample))(X))
        losses[np.isnan(losses)] = 1e99
        best_loss = np.min(losses)

        # Pick the kernel parameters that incurred the lowest loss
        best_p = X[np.argmin(losses),:]

        # Printing the best loss 
        print("Best Objective Value: %.4e" % (best_loss))

        # Storing the best parameters
        if best_loss < self.best_loss: 
            p = deepcopy(self.model.p)
            p[param] = best_p 
            self.model.set_params(p)
            self.best_loss, self.best_p = best_loss, p

    def run(self, lr, epochs, params, p_init = None, verbose=True):
        """Run momentum-based gradient updates.

        Parameters
        ----------
        lr : float
            Learning rate.
        epochs : int
            Number of iterations.
        params : list
            List of parameter keys in the model.p dict to update.
        p_init : dict or None
            Optional starting parameters.
        verbose : bool
            Whether to show a progress bar.
        Returns
        -------
        best_p : dict
            Best parameter dictionary seen during optimization.
        """
        # Initializing the velocity dictionary 
        v = {} 
        if p_init is not None: 
            p = deepcopy(p_init)
        else:
            p = deepcopy(self.model.p)
        for param in p.keys():
            v[param] = jnp.zeros_like(p[param])

        # Declaring iterator 
        if verbose:
            iterator = tqdm(range(epochs))
        else:
            iterator = range(epochs)

        # Storing best loss
        best_loss, best_p = self.best_loss, self.best_p 

        for _ in iterator: 
            # Get value and grad 
            loss, grad = self.grad_fn(p)


            # Gradient-descent with momentum 
            for param in params:
                v[param] = self.beta * v[param] - lr * grad[param]
                p[param] += v[param]

            # Enforce constraints 
            if self.constraints is not None:
                for param in self.constraints.keys():
                    if param in params:
                        p[param] = self.constraints[param](p[param])

            # Display the current loss 
            if verbose:
                iterator.set_postfix_str(f'Loss: {loss:.4e}')

            # Store the best loss 
            if loss < best_loss:
                best_loss = best_loss 
                best_p = deepcopy(p) 
        
        # Store the best p as the model aprameters 
        self.model.set_params(best_p) 
        self.best_p = best_p 
        return best_p 
            
            
            
class ADAM:
    """ADAM optimizer wrapper for model parameter dictionaries.

    Provides a `.run()` method that updates the selected keys in the
    model's `.p` dictionary using bias-corrected ADAM.
    """
    def __init__(self, model, objective_func, beta1 = 0.9, beta2 = 0.999, constraints = None):
        # Storing model and objective function
        self.model, self.obj, self.beta1, self.beta2 = model, objective_func, beta1, beta2 

        # Storing loss function
        self.grad_fn = value_and_grad(jit(lambda p: self.obj(self.model, p)))

        # Compute initial loss
        self.best_loss, _ = self.grad_fn(model.p)
        assert not jnp.isnan(self.best_loss), "Initial Loss is NaN!"

        # Store best params
        self.best_p = deepcopy(model.p)

        # Store parameter constraints 
        self.constraints = constraints 

    def run(self, lr, epochs, params, p_init=None, verbose=True):
        """Run ADAM updates on selected parameter keys.

        Parameters are the same as `Momentum.run` but this method uses the
        ADAM update rule (with bias correction).
        """
        # Initialize moment estimates
        m, s = {}, {}
        if p_init is not None:
            p = deepcopy(p_init)
        else:
            p = deepcopy(self.model.p)
        for param in p.keys():
            m[param] = jnp.zeros_like(p[param])  # first moment
            s[param] = jnp.zeros_like(p[param])  # second moment

        # Declaring iterator
        if verbose:
            iterator = tqdm(range(epochs))
        else:
            iterator = range(epochs)

        # Storing best loss
        best_loss, best_p = self.best_loss, self.best_p

        t = 0  # timestep counter
        
        for _ in iterator:
            t += 1

            # Get value and grad
            loss, grad = self.grad_fn(p)

            # ADAM update
            for param in params:
                # Update biased first moment
                m[param] = self.beta1 * m[param] + (1 - self.beta1) * grad[param]
                # Update biased second raw moment
                s[param] = self.beta2 * s[param] + (1 - self.beta2) * (grad[param] ** 2)

                # Bias correction
                m_hat = m[param] / (1 - self.beta1 ** t)
                s_hat = s[param] / (1 - self.beta2 ** t)

                # Parameter update
                p[param] -= lr * m_hat / (jnp.sqrt(s_hat) + self.model.eps)

            # Enforce constraints
            if self.constraints is not None:
                for param in self.constraints.keys():
                    if param in params:
                        p[param] = self.constraints[param](p[param])

            # Display the current loss
            if verbose:
                iterator.set_postfix_str(f'Loss: {loss:.4e}')

            # Store the best loss
            if loss < best_loss:
                best_loss = loss
                best_p = deepcopy(p)

        # Store the best p as the model parameters
        self.model.set_params(best_p)
        self.best_p = best_p
        return best_p

        

    

class LBFGS:
    """L-BFGS optimizer wrapper backed by jaxopt.LBFGS.

    This wrapper minimizes the provided objective function with
    respect to the model parameter dictionary. The `k_steps` method
    runs the solver and updates the model on completion.
    """
    def __init__(self, model, objective_func, max_iter=10000, tol=1e-6, max_stepsize=1e-1, verbose = False):
        """Create an L-BFGS solver instance.

        Parameters
        ----------
        model : object
            Object exposing `.p` and `.set_params()`.
        objective_func : callable
            Function (model, p) -> scalar loss.
        max_iter, tol, max_stepsize, verbose : passed to jaxopt.LBFGS
        """
        self.model, self.obj = model, objective_func

        # Define scalar loss function in terms of parameters only
        self.loss_fn = lambda p: self.obj(self.model, p)

        # Build JAXOPT solver (handles value_and_grad internally)
        self.solver = JaxoptLBFGS(fun=self.loss_fn, maxiter=max_iter, tol=tol, max_stepsize=max_stepsize, verbose=verbose, history_size = 25, increase_factor = 1.1)

        # Compute initial loss
        self.best_loss = self.loss_fn(model.p)
        assert not jnp.isnan(self.best_loss), "Initial Loss is NaN!"

        # Store best params
        self.best_p = deepcopy(model.p)

    def k_steps(self):
        """
        Run full L-BFGS optimization until convergence.
        Unlike momentum/SGD, we don’t need an epoch loop here:
        jaxopt.LBFGS does the whole solve.
        """
        # Run solver
        res = self.solver.run(self.best_p)

        # Extract results if they're better than what exists
        if res.state.value < self.best_loss:
            best_p, state = res.params, res.state
            best_loss = state.value

            # Save into object
            self.best_p, self.best_loss = best_p, best_loss

        # Print out best log-likelihood
        print("Best objective function eval: %.5f" % (self.best_loss))

        # Update model
        self.model.set_params(self.best_p)


            



        


            
            



            
            
        
                


            

        
