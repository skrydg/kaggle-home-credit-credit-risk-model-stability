import polars as pl

import random
import time
import gc
import numpy as np
from catboost import CatBoostClassifier, Pool, FeaturesData, sum_models
import pandas as pd

from kaggle_home_credit_risk_model_stability.libs.model.voting_model import VotingModel
from kaggle_home_credit_risk_model_stability.libs.env import Env
from kaggle_home_credit_risk_model_stability.libs.metric import calculate_gini_stability_metric

from sklearn.metrics import roc_auc_score

from sklearn.model_selection import StratifiedKFold, StratifiedGroupKFold
from kaggle_home_credit_risk_model_stability.libs.weeks_kfold import WeeksKFold
from .pre_trained_model import PreTrainedCatboostModel

class KFoldCatboostModel:
    def __init__(self, env: Env, features, model_params = None):
        self.env = env
        self.features = features
        self.features_with_target = self.features + ["target"]
        self.batch_count = 5

        if model_params is None:
            self.model_params = {
                "eval_metric": "AUC",
                "iterations": 1000,
                "random_seed": 42,
                "colsample_bylevel": 0.8,
                "max_depth": 10,
                "max_bin": 200,
            }
        else:
            self.model_params = model_params

        self.model = None
        self.train_data = None
        
    def train(self, dataframe, n_splits = 10, KFold = WeeksKFold):
        print("Start train for KFoldCatboostModel")
        weeks = dataframe["WEEK_NUM"]
        oof_predicted = np.zeros(weeks.shape[0])
        
        fitted_models = []
        cv = KFold(n_splits=n_splits)
        for iteration, (idx_train, idx_test) in enumerate(cv.split(dataframe, dataframe["target"], groups=weeks)):
            print(f"Start iteration: {iteration}", flush=True)
            start = time.time()

            categorical_features, numerical_features = self.get_features(dataframe)
            
            batched_models = []
            for i in range(self.batch_count):
                print(f"Start batch training, batch_index: {i}", flush=True)
                current_idx_train = idx_train[i::3]
                current_idx_test = idx_test

                train_pool = Pool(
                    data = FeaturesData(
                        num_feature_data = dataframe[numerical_features][current_idx_train].to_numpy().astype(np.float32),
                        cat_feature_data = dataframe[categorical_features][current_idx_train].to_numpy().astype("object")
                    ),
                    label = dataframe["target"][current_idx_train].to_numpy()
                )
                
                test_pool = Pool(
                    data = FeaturesData(
                        num_feature_data = dataframe[numerical_features][current_idx_test].to_numpy().astype(np.float32),
                        cat_feature_data = dataframe[categorical_features][current_idx_test].to_numpy().astype("object")
                    ),
                    label = dataframe["target"][current_idx_test].to_numpy()
                )
                
                if i > 0:
                    train_pool.set_baseline(batched_models[-1].predict_proba(train_pool)[:, 1])
                    test_pool.set_baseline(batched_models[-1].predict_proba(test_pool)[:, 1])
                
                gc.collect()
                model = CatBoostClassifier(**self.model_params)
                model.fit(train_pool, eval_set=test_pool, verbose=100)

                batched_models.append(model)
                del train_pool
                del test_pool
                gc.collect()
                print(f"Finish batch training, batch_index: {i}", flush=True)
            print(f"Finish iteration: {iteration}", flush=True)
            
            model = sum_models(batched_models)
            model = PreTrainedCatboostModel(model)

            finish = time.time()
            print(f"Fit time: {finish - start}, iteration={iteration}")

            fitted_models.append(model)

            test_pred = self.predict_with_model(dataframe[idx_test], model)
            oof_predicted[idx_test] = test_pred

            current_result_df = pd.DataFrame({
              "WEEK_NUM": dataframe[idx_test]["WEEK_NUM"],
              "true": dataframe[idx_test]["target"],
              "predicted": oof_predicted[idx_test]
            })
            gini_stability_metric = calculate_gini_stability_metric(current_result_df)
            roc_auc_oof = roc_auc_score(current_result_df["true"], current_result_df["predicted"])
            print(f"gini_stability_metric: {gini_stability_metric}, roc_auc_oof: {roc_auc_oof}", flush=True)

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

        print("Finish train for KFoldCatboostModel", flush=True)
        return self.train_data

    def predict(self, dataframe, **kwargs):
        assert(self.model is not None)
        return self.predict_with_model(dataframe, self.model, **kwargs)

    def predict_with_model(self, dataframe, model, **kwargs):
        categorical_features, numerical_features = self.get_features(dataframe)
        pool = Pool(
            data = FeaturesData(
                num_feature_data = dataframe[numerical_features].to_numpy().astype(np.float32),
                cat_feature_data = dataframe[categorical_features].to_numpy().astype("object")
            ),
        )
        return model.predict(pool, **kwargs)

    def predict_chunked(self, dataframe, chunk_size=300000, **kwargs):
        assert(self.model is not None)
        Y_predicted = None
        
        for start_position in range(0, dataframe.shape[0], chunk_size):
            X = dataframe[self.features][start_position:start_position + chunk_size]
            current_Y_predicted = self.predict(X)

            if Y_predicted is None:
                Y_predicted = current_Y_predicted
            else:
                Y_predicted = np.concatenate([Y_predicted, current_Y_predicted])
            gc.collect()
        return Y_predicted
    

    def get_features(self, dataframe):
        categorical_features = [column for column in self.features if dataframe[column].dtype == pl.Enum]
        numerical_features = [column for column in self.features if dataframe[column].dtype != pl.Enum]
        return categorical_features, numerical_features
    
    def get_train_data(self):
        return self.train_data
