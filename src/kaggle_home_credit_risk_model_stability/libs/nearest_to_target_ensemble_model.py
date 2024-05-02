import numpy as np

class NearestToTargetEnsembleModel:
    def __init__(self, target):
        self.target = target

    def predict(self, dataframe):
        assert(dataframe.shape[0] == self.target.shape[0])
        np_array = dataframe
        np_array = np_array - self.target[:, np.newaxis]
        nearest_to_target_index = np_array.transpose().argmin(axis=0)
        return np_array[np.arange(nearest_to_target_index.shape[0]), nearest_to_target_index]