import polars as pl

import random
import time
import gc
import numpy as np
import lightgbm as lgb

from kaggle_home_credit_risk_model_stability.libs.lightgbm.to_pandas import to_pandas
from kaggle_home_credit_risk_model_stability.libs.lightgbm.dataset_creator import LightGbmDatasetCreator
from kaggle_home_credit_risk_model_stability.libs.model.voting_model import VotingModel
from kaggle_home_credit_risk_model_stability.libs.env import Env

from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import roc_auc_score

class LightGbmModel:
    def __init__(self, env: Env, features):
        self.env = env

        self.features = features

        self.default_model_params = {
            "boosting_type": "gbdt",
            "objective": "binary",
            "metric": "auc",
            "max_depth": 10,
            "max_bin": 250,
            "learning_rate": 0.05,
            "n_estimators": 1000,
            "colsample_bytree": 0.8,
            "colsample_bynode": 0.8,
            "verbose": -1,
            "random_state": 42,
            "device": "gpu",
            "n_jobs": -1
        }

        self.model = None
        self.train_data = None
        self.serializer = None

    def _get_dataset(self, dataframe, dataset_params):
        return LightGbmDatasetCreator(dataset_params).create(dataframe)

    def train(self, train_dataframe, test_dataframe, model_params = None):
        print("Start train for LightGbmModel")
        if model_params is None:
            model_params = self.default_model_params

        dataframe = train_dataframe.vstack(test_dataframe)
        
        dataset = self._get_dataset(dataframe, {"max_bin": model_params["max_bin"]})
        train_subset = np.arange(0, train_dataframe.shape[0])
        test_subset = np.arange(train_dataframe.shape[0], train_dataframe.shape[0] + test_dataframe.shape[0])
        
        start = time.time()
        model = lgb.train(
            model_params,
            dataset.subset(train_subset),
            valid_sets=[dataset.subset(test_subset)],
            callbacks=[lgb.log_evaluation(100), lgb.early_stopping(100)]
        )

        finish = time.time()
        print("Fit time: {}".format(finish - start))
        gc.collect()

        self.model = VotingModel([model])

        train_Y_predicted = self.predict(train_dataframe)
        test_Y_predicted = self.predict(test_dataframe)

        self.train_data = {
            "train_roc_auc": roc_auc_score(train_dataframe["target"], train_Y_predicted),
            "train_y_predicted": train_Y_predicted,
            "test_roc_auc": roc_auc_score(test_dataframe["target"], test_Y_predicted),
            "test_y_predicted": test_Y_predicted 
        }

        print("Finish train_cv for LightGbmModel")
        return self.train_data
  
    def train_cv(self, dataframe, model_params = None, n_splits = 5):
        print("Start train_cv for LightGbmModel")
        if model_params is None:
            model_params = self.default_model_params

        dataset = self._get_dataset(dataframe, {"max_bin": model_params["max_bin"]})

        weeks = dataframe["WEEK_NUM"]
        oof_predicted = np.zeros(weeks.shape[0])

        fitted_models = []
        cv = StratifiedGroupKFold(n_splits=n_splits, shuffle=False)
        for idx_train, idx_test in cv.split(dataframe[self.features], dataframe["target"], groups=weeks):    
            start = time.time()
            model = lgb.train(
              model_params,
              dataset.subset(idx_train),
              valid_sets=[dataset.subset(idx_test)],
              callbacks=[lgb.log_evaluation(100), lgb.early_stopping(100)]
            )

            finish = time.time()
            print("Fit time: {}".format(finish - start))

            fitted_models.append(model)

            test_pred = model.predict(to_pandas(dataframe[self.features][idx_test]))
            oof_predicted[idx_test] = test_pred
            gc.collect()

        self.model = VotingModel(fitted_models)
        self.serializer.clear()

        self.train_data = {
          "roc_auc_oof": roc_auc_score(dataframe["target"], oof_predicted),
          "oof_predicted": oof_predicted
        }

        print("Finish train_cv for LightGbmModel")
        return self.train_data

    def predict(self, dataframe, chunk_size = 100000):
        assert(self.model is not None)

        Y_predicted = None

        for start_position in range(0, dataframe.shape[0], chunk_size):
            X_pandas = to_pandas(dataframe[self.features][start_position:start_position + chunk_size])
            current_Y_predicted = self.model.predict(X_pandas)

            if Y_predicted is None:
                Y_predicted = current_Y_predicted
            else:
                Y_predicted = np.concatenate([Y_predicted, current_Y_predicted])
            gc.collect()

        return Y_predicted

    def get_train_data(self):
        return self.train_data