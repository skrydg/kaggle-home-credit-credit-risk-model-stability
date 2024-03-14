import polars as pl

from pathlib import Path

from kaggle_home_credit_risk_model_stability.libs.env import Env

class FreatureDescriptionGetter:
  def __init__(self, env: Env):
    self.fd = pl.read_csv(env.input_directory / "home-credit-credit-risk-model-stability" / "feature_definitions.csv")

  def get(self, feature_name):
    return self.fd.filter(self.fd["Variable"] == feature_name)["Description"][0]
