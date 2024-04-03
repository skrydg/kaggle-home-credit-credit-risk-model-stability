import polars as pl

import random
import time
import gc
import numpy as np
import lightgbm as lgb
import pandas as pd

from kaggle_home_credit_risk_model_stability.libs.model.voting_model import VotingModel
from kaggle_home_credit_risk_model_stability.libs.lightgbm import LightGbmDatasetSerializer
from kaggle_home_credit_risk_model_stability.libs.env import Env
from kaggle_home_credit_risk_model_stability.libs.metric import calculate_gini_stability_metric

from sklearn.metrics import roc_auc_score

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

def dataframe_enums_to_physycal(dataframe):
    return dataframe.with_columns(*[
        pl.col(column).to_physical()
        for column in dataframe.columns
        if dataframe[column].dtype == pl.Enum
    ])

class LightGbmModel:
    def __init__(self, env: Env, features, model_params = None, metrics=None):
        self.env = env
        self.features = features
        self.features_with_target = self.features + ["target"]

        if model_params is None:
            self.model_params = {
              "boosting_type": "gbdt",
              "objective": "binary",
              "metric": "None",
              "max_depth": 8,
              "max_bin": 250,
              "learning_rate": 0.05,
              "n_estimators": 1000,
              "colsample_bytree": 0.8,
              "colsample_bynode": 0.8,
              "verbose": -1,
              "random_state": 42,
              "n_jobs": -1
          }
        else:
            self.model_params = model_params

        self.model = None
        self.train_data = None
        self.metrics = metrics

    def train_without_validation(self, train_dataframe):
        print("Start train for LightGbmModel")

        print("Start data serialization")
        start = time.time()

        train_dataset_serializer = LightGbmDatasetSerializer(self.env.output_directory / "train_datasert", {"max_bin": self.model_params["max_bin"]})
        train_dataset_serializer.serialize(train_dataframe[self.features_with_target])
        train_dataset = train_dataset_serializer.deserialize()

        finish = time.time()
        print(f"Finish data serialization, time={finish - start}")

        start = time.time()
        model = lgb.train(
            self.model_params,
            train_dataset,
            callbacks=[lgb.log_evaluation(100)],
            feval=self.feval_metrics
        )

        finish = time.time()
        print("Fit time: {}".format(finish - start))
        gc.collect()

        self.model = VotingModel([model])

        train_Y_predicted = self.predict_chunked(dataframe_enums_to_physycal(train_dataframe))

        self.train_data = {
            "train_roc_auc": roc_auc_score(train_dataframe["target"], train_Y_predicted),
            "train_y_predicted": train_Y_predicted
        }

        train_dataset_serializer.clear()
        print("Finish train_cv for LightGbmModel")
        return self.train_data
            
    def train(self, train_dataframe, test_dataframe = None):
        if test_dataframe is None:
            return self.train_without_validation(train_dataframe)

        print("Start train for LightGbmModel")

        print("Start data serialization")
        start = time.time()
        train_dataset_serializer = LightGbmDatasetSerializer(self.env.output_directory / "train_datasert", {"max_bin": self.model_params["max_bin"]})
        train_dataset_serializer.serialize(train_dataframe[self.features_with_target])
        train_dataset = train_dataset_serializer.deserialize()

        test_dataset_serializer = LightGbmDatasetSerializer(self.env.output_directory / "test_datasert", {"max_bin": self.model_params["max_bin"]})
        test_dataset_serializer.serialize(test_dataframe[self.features_with_target])
        test_dataset = test_dataset_serializer.deserialize()

        finish = time.time()
        print(f"Finish data serialization, time={finish - start}")

        gc.collect()
        start = time.time()
        model = lgb.train(
            self.model_params,
            train_dataset,
            valid_sets=[test_dataset],
            callbacks=[lgb.log_evaluation(100), lgb.early_stopping(100, first_metric_only=True)],
            feval=self.feval_metrics
        )

        finish = time.time()
        print("Fit time: {}".format(finish - start))
        gc.collect()

        self.model = VotingModel([model])

        train_Y_predicted = self.predict_chunked(dataframe_enums_to_physycal(train_dataframe))
        test_Y_predicted = self.predict_chunked(dataframe_enums_to_physycal(test_dataframe))

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
  
    def train_cv(self, dataframe, n_splits = 5, KFold = WeeksKFold):
        print("Start train_cv for LightGbmModel")
        weeks = dataframe["WEEK_NUM"]
        oof_predicted = np.zeros(weeks.shape[0])
        
        fitted_models = []
        cv = KFold(n_splits=n_splits)
        for idx_train, idx_test in cv.split(dataframe, dataframe["target"], groups=weeks):
            print("Start data serialization")
            start = time.time()

            train_dataset_serializer = LightGbmDatasetSerializer(self.env.output_directory / "train_datasert", {"max_bin": self.model_params["max_bin"]})
            test_dataset_serializer = LightGbmDatasetSerializer(self.env.output_directory / "test_datasert", {"max_bin": self.model_params["max_bin"]})

            train_dataset_serializer.serialize(dataframe[self.features_with_target][idx_train])
            train_dataset = train_dataset_serializer.deserialize()
            train_dataset.week_num = dataframe["WEEK_NUM"][idx_train]

            test_dataset_serializer.serialize(dataframe[self.features_with_target][idx_test])
            test_dataset = test_dataset_serializer.deserialize()
            test_dataset.week_num = dataframe["WEEK_NUM"][idx_test]

            finish = time.time()
            print(f"Finish data serialization, time={finish - start}")

            start = time.time()
            model = lgb.train(
              self.model_params,
              train_dataset,
              valid_sets=[test_dataset],
              callbacks=[lgb.log_evaluation(100), lgb.early_stopping(100, first_metric_only=True)],
              feval=self.feval_metrics
            )

            finish = time.time()
            print("Fit time: {}".format(finish - start))

            fitted_models.append(model)

            test_pred = self.predict_with_model(dataframe_enums_to_physycal(dataframe[idx_test]), model)
            oof_predicted[idx_test] = test_pred

            current_result_df = pd.DataFrame({
              "WEEK_NUM": dataframe[idx_test]["WEEK_NUM"],
              "true": dataframe[idx_test]["target"],
              "predicted": oof_predicted[idx_test]
            })
            gini_stability_metric = calculate_gini_stability_metric(current_result_df)
            roc_auc_oof = roc_auc_score(current_result_df["true"], current_result_df["predicted"])
            print(f"gini_stability_metric: {gini_stability_metric}, roc_auc_oof: {roc_auc_oof}")

            train_dataset_serializer.clear()
            test_dataset_serializer.clear()
            gc.collect()

        self.model = VotingModel(fitted_models)

        result_df = pd.DataFrame({
          "WEEK_NUM": dataframe["WEEK_NUM"],
          "true": dataframe["target"],
          "predicted": oof_predicted,
        })
        
        self.train_data = {
          "roc_auc_oof": roc_auc_score(result_df["true"], result_df["predicted"]),
          "gini_stability_metric": calculate_gini_stability_metric(result_df),
          "oof_predicted": result_df["predicted"].to_numpy()
        }

        print("Finish train_cv for LightGbmModel")
        return self.train_data

    def predict(self, dataframe, **kwargs):
        assert(self.model is not None)
        return self.predict_with_model(dataframe, self.model, **kwargs)

    def predict_with_model(self, dataframe, model, **kwargs):
        return model.predict(dataframe[self.features], **kwargs)

    def predict_chunked(self, dataframe, chunk_size=300000, **kwargs):
        assert(self.model is not None)
        Y_predicted = None
        

        for start_position in range(0, dataframe.shape[0], chunk_size):
            X = dataframe[self.features][start_position:start_position + chunk_size]
            current_Y_predicted = self.model.predict(dataframe_enums_to_physycal(X))

            if Y_predicted is None:
                Y_predicted = current_Y_predicted
            else:
                Y_predicted = np.concatenate([Y_predicted, current_Y_predicted])
            gc.collect()
        return Y_predicted
    
    def get_train_data(self):
        return self.train_data
    
    def feval_metrics(self, preds: np.ndarray, data: lgb.Dataset):
        ret = []
        if "roc_auc" in self.metrics:
            ret.append(LightGbmModel.roc_auc_for_lgbm(preds, data))

        if "gini_stability_metric" in self.metrics:
            ret.append(LightGbmModel.gini_stability_metric_for_lgbm(preds, data))

        return ret

    @staticmethod
    def gini_stability_metric_for_lgbm(preds: np.ndarray, data: lgb.Dataset):
        df = pd.DataFrame({
            "WEEK_NUM": data.week_num,
            "true": data.get_label(),
            "predicted": preds,
        })

        return 'gini_stability_metric', calculate_gini_stability_metric(df), True

    @staticmethod
    def roc_auc_for_lgbm(preds: np.ndarray, data: lgb.Dataset):
        return 'roc_auc', roc_auc_score(data.get_label(), preds), True
