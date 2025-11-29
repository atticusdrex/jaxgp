from .gp import * 

"""
gplib.mf
-----------
Multi-fidelity regression wrappers.

This module provides thin wrappers that compose `GP`, `DeltaGP` and
`SVGP` objects into multi-fidelity models. Supported multi-fidelity
approaches include Hyperkriging, Kennedy-O'Hagan co-kriging,
NARGP and a block Cokriging implementation.

Each class expects a `data_dict` where each fidelity level is a dict
containing keys like `'X'`, `'Y'`, and optional `'noise_var'`.
"""

# Creating a parent class for the multi-fidelity regressor objects 
class MFRegressor:
    """Base class for multi-fidelity regressors.

    Parameters
    ----------
    data_dict : dict
        Mapping fidelity level -> dict with at least 'X' and 'Y'.
    kernel : class
        Kernel class to instantiate for internal GPs.
    mean_func : class
        Mean-function class to instantiate for internal GPs.
    epsilon : float
        Jitter added to kernel diagonals for numerical stability.
    """
    def __init__(self, data_dict, kernel, mean_func, epsilon = 1e-8):
        # Storing data dictionary, kernel and kernel base dimension
        self.d, self.kernel, self.mean = deepcopy(data_dict), kernel, mean_func
        # Storing keyword arguments 
        self.eps = epsilon
        # Number of levels of fidelity
        self.K = len(self.d) 


class Hyperkriging(MFRegressor):
    def __init__(self, *args, max_cond = 1e3, **kwargs):
        # Initializing the parent class 
        super().__init__(*args, **kwargs)

        # Initializing level zero model
        self.d[0]['model'] = GP(self.d[0]['X'], self.d[0]['Y'], self.kernel, self.mean, noise_var = self.d[0]['noise_var'], epsilon = self.eps, max_cond = max_cond, calibrate=True)

        # Initializing models 
        for level in range(1, self.K):
            # Initializing the features 
            features = self.d[level]['X']
            for sublevel in range(level):
                # Getting the mean function from the sublevel immediately under
                mean, _ = self.d[sublevel]['model'].predict(features)

                # Horizontally concatenating the mean function to the existing features 
                features = jnp.hstack((features, mean.reshape(-1,1)))
            
            # Creating a model trained on this set of features 
            self.d[level]['model'] = GP(
                features,
                self.d[level]['Y'],
                self.kernel,
                self.mean,
                noise_var = self.d[level]['noise_var'], epsilon = self.eps, max_cond = max_cond, calibrate=True
            )

    def predict(self, Xtest, level, full_cov = True):
        test_features = Xtest
        # Initializing the features 
        for sublevel in range(level):
            # Getting the mean function from the sublevel immediately under
            mean, _ = self.d[sublevel]['model'].predict(test_features, full_cov = full_cov)

            # Horizontally concatenating the mean function to the existing features 
            test_features = jnp.hstack((test_features, mean.reshape(-1,1)))
        
        return self.d[level]['model'].predict(test_features, full_cov = full_cov)
    
    def optimize(self, level, params = ['k_param', 'm_param', 'noise_var'], lr=1e-3, epochs = 1000, beta1 = 0.9, beta2 = 0.999):
        
        # Optimizing lowest-fidelity model 
        if level == 0:
            # Creating a model trained on this set of features 
            optimizer = ADAM(self.d[0]['model'], neg_mll, beta1=beta1, beta2=beta2)
            optimizer.run(lr, epochs, params)
        else:
            features = self.d[level]['X']
            for sublevel in range(level):
                # Getting the mean function from the sublevel immediately under
                mean, _ = self.d[sublevel]['model'].predict(features)

                # Horizontally concatenating the mean function to the existing features 
                features = jnp.hstack((features, mean.reshape(-1,1)))

            # Updating the features at this fidelity-level 
            self.d[level]['model'].X = deepcopy(features)
            
            # Creating a model trained on this set of features 
            optimizer = ADAM(self.d[level]['model'], neg_mll, beta1=beta1, beta2=beta2)
            optimizer.run(lr, epochs, params)


'''
Kennedy O'Hagan Co-Kriging 
---------------------------------
NOTE: Requires training data to be nested! 
'''
class KennedyOHagan(MFRegressor):
    def __init__(self, *args, max_cond = 1e3, **kwargs):
        # Initializing the parent class 
        super().__init__(*args, **kwargs)

        # Initializing level zero as a simple GP 
        # Initializing level zero model
        self.d[0]['model'] = GP(self.d[0]['X'], self.d[0]['Y'], self.kernel, self.mean, noise_var = self.d[0]['noise_var'], epsilon = self.eps, max_cond = max_cond, calibrate=True)

        # Iterating through the levels of fidelity
        for level in range(1, self.K):
            mean, _ = self.predict(self.d[level]['X'], level-1)
            self.d[level]['model'] = DeltaGP(
                self.d[level]['X'], self.d[level]['Y'], mean.ravel(), self.kernel, self.mean, noise_var = self.d[level]['noise_var'], epsilon = self.eps, max_cond = max_cond, calibrate=True)

    # Update the y2 predictions for each model
    def update(self):
        for level in range(1, self.K):
            mean, _ = self.predict(self.d[level]['X'], level-1)
            self.d[level]['model'].Y2 = mean.ravel()
    
    def predict(self, Xtest, level, full_cov = True):
        # Predicting lowest level of fidelity 
        Ymean, Ycov = self.d[0]['model'].predict(Xtest, full_cov = full_cov)

        # Predicting up to the specified level of fidelity
        for sublevel in range(1, level+1):
            # Getting rho 
            rho = self.d[sublevel]['model'].p['rho']
            # Getting the delta predictions
            delta_mean, delta_cov = self.d[sublevel]['model'].predict(Xtest, full_cov = full_cov)

            # Getting this level's mean and variance 
            Ymean = rho * Ymean + delta_mean
            Ycov = rho**2 * Ycov + delta_cov 

        return Ymean, Ycov 

    def optimize(self, level, params = ['k_param', 'm_param', 'rho', 'noise_var'], lr = 1e-3, epochs = 1000, beta1 = 0.9, beta2 = 0.999):
        
        # Optimizing lowest-fidelity model 
        if level == 0:
            # Creating a model trained on this set of features 
            optimizer = ADAM(self.d[0]['model'], neg_mll, beta1=0.9, beta2=0.999)
            params.remove("rho")
            optimizer.run(lr, epochs, params)
        else:
            # Updating the levels to approximate 
            for sublevel in range(1, level+1):
                mean, _ = self.predict(self.d[sublevel]['X'], sublevel-1)
                self.d[sublevel]['model'].Y2 = mean.ravel()

            # Creating a model trained on this set of features 
            optimizer = ADAM(self.d[level]['model'], delta_neg_mll, beta1=0.9, beta2=0.999)
            optimizer.run(lr, epochs, params)

class NARGP(MFRegressor):
    def __init__(self, *args, max_cond = 1e3, **kwargs):
        # Initializing the parent class 
        super().__init__(*args, **kwargs)

        # Initializing level zero model
        self.d[0]['model'] = GP(self.d[0]['X'], self.d[0]['Y'], RBF, self.mean, noise_var = self.d[0]['noise_var'], epsilon = self.eps, max_cond = max_cond, calibrate=True)

        # Initializing models 
        for level in range(1, self.K):
            # Initializing the features 
            features = self.d[level]['X']
            for sublevel in range(level):
                # Getting the mean function from the sublevel immediately under
                mean, _ = self.d[sublevel]['model'].predict(features)

                # Horizontally concatenating the mean function to the existing features 
                features = jnp.hstack((self.d[level]['X'], mean.reshape(-1,1)))
            
            # Creating a model trained on this set of features 
            self.d[level]['model'] = GP(
                features,
                self.d[level]['Y'],
                self.kernel,
                self.mean,
                noise_var = self.d[level]['noise_var'], epsilon = self.eps, max_cond = max_cond, calibrate=True
            )

    def predict(self, Xtest, level, full_cov = True):
        test_features = Xtest
        # Initializing the features 
        for sublevel in range(level):
            # Getting the mean function from the sublevel immediately under
            mean, _ = self.d[sublevel]['model'].predict(test_features, full_cov = full_cov)

            # Horizontally concatenating the mean function to the existing features 
            test_features = jnp.hstack((Xtest, mean.reshape(-1,1)))
        
        return self.d[level]['model'].predict(test_features, full_cov = full_cov)
    
    def optimize(self, level, params = ['k_param', 'm_param', 'noise_var'], lr = 1e-3, epochs = 1000, beta1 = 0.9, beta2 = 0.999):
        
        # Optimizing lowest-fidelity model 
        if level == 0:
            # Creating a model trained on this set of features 
            optimizer = ADAM(self.d[level]['model'], neg_mll, beta1=beta1, beta2=beta2)
            optimizer.run(lr, epochs, params)
        else:
            features = self.d[level]['X']
            for sublevel in range(level):
                # Getting the mean function from the sublevel immediately under
                mean, _ = self.d[sublevel]['model'].predict(features)

                # Horizontally concatenating the mean function to the existing features 
                features = jnp.hstack((self.d[level]['X'], mean.reshape(-1,1)))

            # Updating the features at this fidelity-level 
            self.d[level]['model'].X = deepcopy(features)
            
            # Creating a model trained on this set of features 
            optimizer = ADAM(self.d[level]['model'], neg_mll, beta1=beta1, beta2=beta2)
            optimizer.run(lr, epochs, params)


class Cokriging(MFRegressor):
    def __init__(self, *args, max_cond = 1e3, **kwargs):
        # Initializing the parent class 
        super().__init__(*args, **kwargs)

        # Initializing the parameter vector with zeros initially 
        self.p = {}

        # Storing dimensions 
        self.N_total = 0.0 
        for level in range(self.K):
            self.N_total += len(self.d[level]['Y'])

        # Storing full Y matrix 
        Y_list = [] 
        for level in range(self.K):
            Y_list.append(self.d[level]['Y'])
        self.Yfull = jnp.concatenate(Y_list)

        # Initializing kernel and mean functions 
        self.input_dim = self.d[0]['X'].shape[1]
        self.kernel = self.kernel(self.input_dim, epsilon=self.eps)
        self.mean = self.mean(self.input_dim, epsilon = self.eps)

        # Initialize kernel and mean function parameters via calibration  
        for level in range(self.K):
            # Initializing Linear Model of Coregionalization coefficients for each level 
            self.p['B_%d' % level] = inv_softplus(self.eps * jnp.ones((self.K, self.K)))
            # Calibrating kernel hyperparameters  
            self.p['k_param_%d' % level] = self.kernel.calibrate(self.d[level]['X'], self.d[level]['Y'])
            # Calibrating mean function hyperparameters 
            self.p['m_param_%d' % level] = self.mean.calibrate(self.d[level]['X'], self.d[level]['Y'])
            # Storing noise variances 
            self.p['noise_var_%d' % level] = inv_softplus(self.d[level]['noise_var'])
        
        # Initializing L and alpha matrices 
        self.L = self.get_L(self.p)
        self.alpha = self.get_alpha(self.L, self.p)
        
    def Ktrain(self, p):
        kernel_matrices = [] 
        for level1 in range(self.K):
            mat_list = []
            for level2 in range(self.K):
                # Initializing a kernel matrix 
                Kmat = jnp.zeros((self.d[level1]['X'].shape[0], self.d[level2]['X'].shape[0]))
                # Looping through the LMC kernels 
                
                # Adding noise if necessary 
                if level1 == level2:
                    Kmat += (softplus(p['noise_var_%d' % level1]) + self.eps) * jnp.eye(Kmat.shape[0])

                for i in range(self.K):
                    Kmat += self.get_B(p['B_%d' % i], i)[level1, level2] * K(self.d[level1]['X'], self.d[level2]['X'], self.kernel, p['k_param_%d' % i])
                # Appending this kernel matrix onto the mat_list 
                mat_list.append(Kmat)
            # Appending this row to the list of kernel matrices 
            kernel_matrices.append(mat_list)
        # Returning the block matrix of the combined kernel matrices
        return jnp.block(kernel_matrices)
    
    def get_B(self, L, level):
        b = jnp.array([1 if x == level else 0 for x in range(self.K)])
        B = softplus(jnp.tril(L) + jnp.tril(L).T)
        return B - jnp.diag(jnp.diag(B)) + jnp.diag(b)

    def Ktest(self, level1, Xtest, p):
        kernel_matrices = []
        mat_list = []
        B_list = [self.get_B(p['B_%d' % i], i) for i in range(self.K)] 
        for level2 in range(self.K):
            # Initializing a kernel matrix 
            Kmat = jnp.zeros((Xtest.shape[0], self.d[level2]['X'].shape[0]))
            # Looping through the LMC kernels 
            for i in range(self.K):
                Kmat += B_list[i][level1, level2] * K(Xtest, self.d[level2]['X'], self.kernel, p['k_param_%d' % i])
            # Appending this kernel matrix onto the mat_list 
            mat_list.append(Kmat)
        # Appending this row to the list of kernel matrices 
        kernel_matrices.append(mat_list)
        return jnp.block(kernel_matrices)

    def mean_train(self, p):
        mean_evals = []
        # Looping through the levels of fidelity and evaluating the mean 
        for level in range(self.K):
            mean_evals.append(self.mean.eval(self.d[level]['X'], p['m_param_%d' % level]))
        # Returning the concatenated stack of mean function evaluations 
        return jnp.concat(mean_evals)

    def get_L(self, p):
        # Form kernel matrix 
        Ktrain = self.Ktrain(p)
        # Take cholesky factorization 
        return cholesky(Ktrain, lower=True)

    def get_alpha(self, L, p):
        # Solving the linear system 
        return cho_solve((L, True), self.Yfull - self.mean_train(p))
    
    def predict(self, Xtest, level, full_cov = True):
        # Creating the test kernel matrix 
        Ktest = self.Ktest(level, Xtest, self.p)
        # Now we create the K(Xtest, Xtest) matrix (with the LMC structure)
        Kaux = jnp.zeros((Xtest.shape[0], Xtest.shape[0]))
        # Kaux += K(Xtest, Xtest, self.kernel, self.p['k_param_%d' % level])
        for sublevel in range(self.K):
            Kaux += self.get_B(self.p['B_%d' % sublevel], sublevel)[level, level] * K(Xtest, Xtest, self.kernel, self.p['k_param_%d' % sublevel])
        # We assume the L and alpha have already been formed 
        mu = Ktest @ self.alpha + self.mean.eval(Xtest, self.p['m_param_%d' % level])
        cov = Kaux - Ktest @ cho_solve((self.L, True), Ktest.T)
        # Returning full covariance or diagonal 
        if full_cov:
            return mu, cov 
        else:
            return mu, jnp.diag(cov)
        
    def set_params(self, p):
        self.p = deepcopy(p)
        self.L = self.get_L(p)
        self.alpha = self.get_alpha(self.L, p)


