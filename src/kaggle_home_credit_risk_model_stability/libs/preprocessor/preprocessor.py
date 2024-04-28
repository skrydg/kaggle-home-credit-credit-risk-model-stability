import time
import gc
import copy
import hashlib
import polars as pl

from kaggle_home_credit_risk_model_stability.libs.input.dataset import Dataset

class Preprocessor:
    def __init__(self, steps):
        self.steps = steps
    
    def process_train_dataset(self, train_dataset_generator):
        #print("Dataset hash='{}'".format(self._get_dataset_hash(train_dataset)))
        for name, step in self.steps.items():
            start = time.time()
            train_dataset_generator = self.collect_garbage(step.process_train_dataset(train_dataset_generator))
            gc.collect()
            finish = time.time()
            print("Step: {}".format(name), flush=True)
            #print("Dataset hash='{}' after step: {}".format(self._get_dataset_hash(train_dataset), name))
        
        train_dataset = next(train_dataset_generator)
        return train_dataset
    
    def process_test_dataset(self, test_dataset_generator):
        #print("Dataset hash='{}'".format(self._get_dataset_hash(test_dataset)))
        for name, step in self.steps.items():
            start = time.time()
            test_dataset_generator = self.collect_garbage(step.process_test_dataset(test_dataset_generator))
            gc.collect()
            finish = time.time()
            print("Step: {}".format(name), flush=True)
            #print("Dataset hash='{}' after step: {}".format(self._get_dataset_hash(test_dataset), name))
        
        test_dataset = next(test_dataset_generator)
        return test_dataset
            
    def collect_garbage(self, test_dataset_generator):
        gc.collect()
        yield from test_dataset_generator

    def _get_dataset_hash(self, dataset):
        if type(dataset) is Dataset:
            dataset_hash = hashlib.new('sha256')
            for name, table in sorted(dataset.get_tables()):
                dataset_hash.update(self._get_table_hash(table).encode('utf-8'))
            return dataset_hash.hexdigest()
        else:
            return self._get_table_hash(dataset)
      

    def _get_table_hash(self, table):
        hash_table = table.select(~pl.selectors.by_dtype(pl.Null))
        if len(hash_table.columns) == 0:
            return hashlib.new('sha256').hexdigest()
        
        dataframe_hash = hashlib.sha256(hash_table.hash_rows().to_numpy())
        dataframe_hash.update(str(hash_table.dtypes).encode('utf-8'))
        return dataframe_hash.hexdigest()