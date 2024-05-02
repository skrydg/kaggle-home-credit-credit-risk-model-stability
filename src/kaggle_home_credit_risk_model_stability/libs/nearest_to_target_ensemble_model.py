import numpy as np

class NearestToTargetEnsembleModel:
    def __init__(self, target):
        self.target = target

    def predict(self, X):
        assert(X.shape[0] == self.target.shape[0])
        nearest_to_target_index = np.absolute((X - self.target[:, np.newaxis]).transpose()).argmin(axis=0)
        return X[np.arange(nearest_to_target_index.shape[0]), nearest_to_target_index]