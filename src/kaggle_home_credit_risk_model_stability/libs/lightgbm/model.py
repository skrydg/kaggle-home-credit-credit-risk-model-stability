import polars as pl

import random
import time
import gc
import numpy as np
import lightgbm as lgb

from kaggle_home_credit_risk_model_stability.libs.model.voting_model import VotingModel
from kaggle_home_credit_risk_model_stability.libs.lightgbm import LightGbmDatasetSerializer
from kaggle_home_credit_risk_model_stability.libs.env import Env

from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import roc_auc_score

class LightGbmModel:
    def __init__(self, env: Env, features, model_params = None):
        self.env = env
        self.features = features

        if model_params is None:
            self.model_params = {
              "boosting_type": "gbdt",
              "objective": "binary",
              "metric": "auc",
              "max_depth": 8,
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
        else:
            self.model_params = model_params

        self.model = None
        self.train_data = None

                
    def train(self, train_dataframe, test_dataframe):
        print("Start train for LightGbmModel")

        train_dataset_serializer = LightGbmDatasetSerializer(self.env.output_directory / "train_datasert", {"max_bin": self.model_params["max_bin"]})
        test_dataset_serializer = LightGbmDatasetSerializer(self.env.output_directory / "test_datasert", {"max_bin": self.model_params["max_bin"]})

        train_dataset_serializer.serialize(train_dataframe)
        train_dataset = train_dataset_serializer.deserialize()

        test_dataset_serializer.serialize(test_dataframe)
        test_dataset = test_dataset_serializer.deserialize()


        start = time.time()
        model = lgb.train(
            self.model_params,
            train_dataset,
            valid_sets=[test_dataset],
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

        train_dataset_serializer.clear()
        test_dataset_serializer.clear()
        print("Finish train_cv for LightGbmModel")
        return self.train_data
  
    def train_cv(self, dataframe, n_splits = 5):
        print("Start train_cv for LightGbmModel")
        weeks = dataframe["WEEK_NUM"]
        oof_predicted = np.zeros(weeks.shape[0])

        fitted_models = []
        cv = StratifiedGroupKFold(n_splits=n_splits, shuffle=False)
        for idx_train, idx_test in cv.split(dataframe[self.features], dataframe["target"], groups=weeks):   
            train_dataset_serializer = LightGbmDatasetSerializer(self.env.output_directory / "train_datasert", {"max_bin": self.model_params["max_bin"]})
            test_dataset_serializer = LightGbmDatasetSerializer(self.env.output_directory / "test_datasert", {"max_bin": self.model_params["max_bin"]})

            train_dataset_serializer.serialize(dataframe[idx_train])
            train_dataset = train_dataset_serializer.deserialize()

            test_dataset_serializer.serialize(dataframe[idx_test])
            test_dataset = test_dataset_serializer.deserialize()
            
            start = time.time()
            model = lgb.train(
              self.model_params,
              train_dataset,
              valid_sets=[test_dataset],
              callbacks=[lgb.log_evaluation(100), lgb.early_stopping(100)]
            )

            finish = time.time()
            print("Fit time: {}".format(finish - start))

            fitted_models.append(model)

            test_pred = self.predict_with_model(dataframe[idx_test], model)
            oof_predicted[idx_test] = test_pred

            train_dataset_serializer.clear()
            test_dataset_serializer.clear()
            gc.collect()

        self.model = VotingModel(fitted_models)

        self.train_data = {
          "roc_auc_oof": roc_auc_score(dataframe["target"], oof_predicted),
          "oof_predicted": oof_predicted
        }

        print("Finish train_cv for LightGbmModel")
        return self.train_data

    def predict(self, dataframe, chunk_size = 100000):
        assert(self.model is not None)
        return self.predict_with_model(dataframe, self.model, chunk_size)

    def predict_with_model(self, dataframe, model, chunk_size = 100000):
        Y_predicted = None

        for start_position in range(0, dataframe.shape[0], chunk_size):
            X = dataframe[start_position:start_position + chunk_size]
            physical_X = X[self.features].with_columns(*[
                pl.col(column).to_physical()
                for column in self.features
                if X[column].dtype == pl.Enum
            ])
            current_Y_predicted = model.predict(physical_X)

            if Y_predicted is None:
                Y_predicted = current_Y_predicted
            else:
                Y_predicted = np.concatenate([Y_predicted, current_Y_predicted])
            gc.collect()

        return Y_predicted
    def get_train_data(self):
        return self.train_data