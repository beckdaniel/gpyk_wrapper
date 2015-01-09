
import unittest
from gpyk_wrapper.svm_wrapper import SVMWrapper
import GPy
import numpy as np
from sklearn.svm import SVR
from sklearn.grid_search import GridSearchCV

class tests(unittest.TestCase):
    def setUp(self):
        self.X = np.array([['(S (NP ns) (VP v))'],
                           ['(S (NP n) (VP v))'],
                           ['(S (NP (N a)) (VP (V c)))'],
                           ['(S (NP (Det a) (N b)) (VP (V c)))'],
                           ['(S (NP (ADJ colorless) (N ideas)) (VP (V sleep) (ADV furiously)))']],
                          dtype=object)
        self.X2 = np.ones(5)[:,None]
        self.X3 = np.concatenate((self.X, self.X2), axis=1)
        self.Y = np.array([[(a+10)*5] for a in range(5)])

    def test_cv(self):
        svr = SVR()
        kernel = GPy.kern.SymbolAwareSubsetTreeKernel()
        model = SVMWrapper(kernel, svr)#, C=1, epsilon=1e-2, _lambda=1e-1, sigma=1)
        tuned_params = {'C': [0.1, 0.5, 1, 5, 10],
                        'epsilon': [1e-4, 1e-3, 1e-2, 1e-1, 1],
                        '_lambda': [1e-3, 1e-2, 1e-1, 0.5, 1],
                        'sigma': [1e-10, 0.5, 1, 2],
                        }
        cv_model = GridSearchCV(model, tuned_params)
        cv_model.fit(self.X, self.Y)
        print cv_model.predict(self.X)
        print cv_model.best_estimator_
        print cv_model.best_estimator_.kernel

    def test_cv2(self):
        svr = SVR()
        kernel = GPy.kern.SymbolAwareSubsetTreeKernel(active_dims=[0]) + GPy.kern.RBF(1, active_dims=[1])
        print kernel
        model = SVMWrapper(kernel, svr)
        tuned_params = {'C': [0.1, 0.5, 1, 5, 10],
                        'epsilon': [1e-4, 1e-3, 1e-2, 1e-1, 1],
                        'sasstk.lambda': [1e-3, 1e-2, 1e-1, 0.5, 1],
                        'sasstk.sigma': [1e-10, 1, 2],
                        'rbf.variance': [0.5, 2],
                        'rbf.lengthscale': [0.5, 2]
                        }
        cv_model = GridSearchCV(model, tuned_params)
        cv_model.fit(self.X3, self.Y)
        print cv_model.predict(self.X3)
        print cv_model.best_estimator_
        print cv_model.best_estimator_.kernel

if __name__ == "__main__":
    unittest.main()
