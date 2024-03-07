import polars as pl

import random
import time
import gc
import numpy as np
import lightgbm as lgb

from kaggle_home_credit_risk_model_stability.libs.lightgbm.serializer import LightGbmDatasetSerializer
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
  
  def _get_dataset(self, dataframe, dataset_params):
    random_str = ''.join(random.choice('0123456789ABCDEF') for i in range(8))
    serializer = LightGbmDatasetSerializer(
      self.env.output_directory / "tmp" / f"lightgbm_dataset_{random_str}", 
      dataset_params
    )
    serializer.serialize(dataframe[self.features], dataframe["target"])
    dataset = serializer.deserialize()

    return dataset
  
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
          callbacks=[lgb.log_evaluation(100), lgb.early_stopping(100)])

        finish = time.time()
        print("Fit time: {}".format(finish - start))

        fitted_models.append(model)
        
        test_pred = model.predict(dataframe[self.features][idx_test].to_pandas())
        oof_predicted[idx_test] = test_pred
        gc.collect()

    self.model = VotingModel(fitted_models)
    print("Finish train_cv for LightGbmModel")

    return {
      "roc_auc_oof": roc_auc_score(dataframe["target"], oof_predicted)
    }

  def predict(self, dataframe):
    pass