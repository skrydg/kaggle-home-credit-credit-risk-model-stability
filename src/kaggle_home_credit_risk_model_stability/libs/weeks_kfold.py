import numpy as np

from sklearn.model_selection import StratifiedKFold, StratifiedGroupKFold

class WeeksKFold:
    def __init__(self, n_splits, random_state=42):
        self.n_splits = n_splits
        self.random_state = random_state
        self.train_folds = [[] for i in range(n_splits)]
        self.test_folds = [[] for i in range(n_splits)]
        
    def split(self, X, Y, groups=None):
        weeks = X["WEEK_NUM"].unique()
        
        for week in weeks:
            week_mask = (X["WEEK_NUM"] == week)
            week_index,  = np.where(week_mask.to_numpy())
            week_X = X.filter(week_mask)
            week_Y = Y.filter(week_mask)
            for index, (idx_train, idx_test) in enumerate(StratifiedKFold(self.n_splits, shuffle=True, random_state=self.random_state).split(week_X, week_Y)):
                self.train_folds[index].append(week_index[idx_train])
                self.test_folds[index].append(week_index[idx_test])
        for i in range(self.n_splits):
            yield np.concatenate(self.train_folds[i]), np.concatenate(self.test_folds[i])
