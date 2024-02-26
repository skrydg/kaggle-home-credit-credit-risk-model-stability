from setuptools import setup, find_packages

setup(
  name='kaggle_home_credit_risk_model_stability',
  version='0.3',
  packages=find_packages(where="src"),
  package_dir={"": "src"},
)