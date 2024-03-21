import numpy as np

from sklearn.base import BaseEstimator, RegressorMixin

class VotingModel(BaseEstimator, RegressorMixin):
    def __init__(self, estimators, w = None):
        super().__init__()
        self.estimators = np.array(estimators)
        if w is None:
            self.w = np.ones(self.estimators.shape)
        else:
            self.w = w
        self.w = self.w / np.sum(self.w)
    
    def predict(self, X):
        y_preds = np.array([estimator.predict(X) for estimator in self.estimators])
        return np.sum(y_preds * self.w, axis=0)
  