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
from kaggle_home_credit_risk_model_stability.libs.weeks_kfold import WeeksKFold

from .pre_trained_model import PreTrainedLightGbmModel
from sklearn.metrics import roc_auc_score

def dataframe_enums_to_physycal(dataframe):
    return dataframe.with_columns(*[
        pl.col(column).to_physical()
        for column in dataframe.columns
        if dataframe[column].dtype == pl.Enum
    ])

class KFoldLightGbmModel:
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

    def train(self, dataframe, n_splits = 10, KFold = WeeksKFold):
        print("Start train for KFoldLightGbmModel")
        weeks = dataframe["WEEK_NUM"]
        oof_predicted = np.zeros(weeks.shape[0])
        
        fitted_models = []
        cv = KFold(n_splits=n_splits)
        for iteration, (idx_train, idx_test) in enumerate(cv.split(dataframe, dataframe["target"], groups=weeks)):
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
            model = PreTrainedLightGbmModel(model)
            finish = time.time()
            print(f"Fit time: {finish - start}, iteration={iteration}")

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
            print(f"gini_stability_metric: {gini_stability_metric}, roc_auc_oof: {roc_auc_oof}", flush=True)

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

        print("Finish train for KFoldLightGbmModel")
        return self.train_data

    def predict(self, dataframe, **kwargs):
        assert(self.model is not None)
        return self.predict_with_model(dataframe, self.model, **kwargs)

    def predict_with_model(self, dataframe, model, **kwargs):
        return model.predict(dataframe_enums_to_physycal(dataframe[self.features]), **kwargs)

    def predict_chunked(self, dataframe, chunk_size=300000, **kwargs):
        assert(self.model is not None)
        Y_predicted = None
        
        for start_position in range(0, dataframe.shape[0], chunk_size):
            X = dataframe[self.features][start_position:start_position + chunk_size]
            current_Y_predicted = self.predict(X, **kwargs)

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
            ret.append(KFoldLightGbmModel.roc_auc_for_lgbm(preds, data))

        if "gini_stability_metric" in self.metrics:
            ret.append(KFoldLightGbmModel.gini_stability_metric_for_lgbm(preds, data))

        return ret

    def get_feature_importance(self, type = "split"):
        fi = np.zeros(self.model.estimators[0].model.feature_importance(type).shape)
        fn = self.model.estimators[0].model.feature_name()
        for estimator in self.model.estimators:
            fi = fi + estimator.model.feature_importance(type)
        sorted_by_importance_features = list(reversed(sorted(list(zip(fi, fn)))))
        sorted_by_importance_features = [(fi, fn) for fi, fn in sorted_by_importance_features]
        return sorted_by_importance_features

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
