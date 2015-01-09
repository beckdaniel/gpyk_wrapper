
import numpy as np
from GPy.kern import SymbolAwareSubsetTreeKernel as SASSTK
from sklearn.svm import SVC, SVR
from sklearn.grid_search import GridSearchCV
from sklearn.base import BaseEstimator, RegressorMixin

class SVMWrapper(BaseEstimator, RegressorMixin):
    """
    Wrapper for SVMs. WARNING: BROKEN...
    """
    def __init__(self, kernel, estimator, **params):
        self.estimator = estimator
        self.estimator.kernel = "precomputed"
        self.kernel = kernel
        self.set_params(**params)
            
    def fit(self, X, y):
        self.Xtrain = X
        gram = self.kernel.K(X)
        self.estimator.fit(gram, y.flatten())
        return self

    def predict(self, X):
        kernel_evals = self.kernel.K(X, self.Xtrain)
        return self.estimator.predict(kernel_evals)

    def set_params(self, **params):
        for param in params:
            self.__setattr__(param, params[param])
            if param == 'C':
                self.estimator.C = params[param]
            elif param == 'epsilon':
                self.estimator.epsilon = params[param]
            elif param == '_lambda':
                self.kernel['lambda'] = params[param]
            else:
                self.kernel[param] = params[param]
        for parameter, value in params.items():
            self.__setattr__(parameter, value)
        return self



        
class SVRTKWrapper(BaseEstimator, RegressorMixin):
    """
    Wrapper for SVMTKs
    """
    def __init__(self, C=1.0, epsilon=0.1, gamma=0.0, _lambda=0.5, _sigma=1):
        self.C = C
        self.epsilon = epsilon
        self.gamma = gamma
        self._lambda = _lambda
        self._sigma = _sigma
        self.estimator = SVR(C=C, epsilon=epsilon, gamma=gamma, kernel="precomputed")
        self.kernel = SASSTK(_lambda=np.array([_lambda]), _sigma=np.array([_sigma]), num_threads=7)
            
    def fit(self, X, y):
        self.Xtrain = X
        gram = self.kernel.K(X)
        self.estimator.fit(gram, y.flatten())
        return self

    def predict(self, X):
        kernel_evals = self.kernel.K(X, self.Xtrain)
        return self.estimator.predict(kernel_evals)

    #def set_params(self, **parameters):
    #    for parameter, value in parameters.items():
    #        self.setattr(parameter, value)

    def set_params(self, **params):
        self.estimator.C = params['C']
        self.estimator.epsilon = params['epsilon']
        #self.estimator.gamma = params['gamma']
        self.kernel['lambda'] = np.array([params['_lambda']])
        self.kernel['sigma'] = np.array([params['_sigma']])
        for parameter, value in params.items():
            self.__setattr__(parameter, value)
        return self

    def get_params(self, deep=True):
        return {'C': self.C,
                'epsilon': self.epsilon,
                'gamma': self.gamma,
                '_lambda': self._lambda,
                '_sigma': self._sigma}
