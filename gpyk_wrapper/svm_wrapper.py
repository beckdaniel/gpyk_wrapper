
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

