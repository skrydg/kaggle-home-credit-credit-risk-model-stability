import time
import gc
import copy

class Preprocessor:
    def __init__(self, steps):
      self.steps = steps
    
    def process_train_dataset(self, train_dataset, columns_info):
      for name, step in self.steps.items():
        start = time.time()
        train_dataset, columns_info = step.process_train_dataset(train_dataset, columns_info)
        gc.collect()
        finish = time.time()
        print("Step: {}, execution_time: {}".format(name, finish - start), flush=True)
      return train_dataset, columns_info
    
    def process_test_dataset(self, test_dataset, columns_info):
      for name, step in self.steps.items():
        start = time.time()
        test_dataset, _ = step.process_test_dataset(test_dataset, copy.deepcopy(columns_info))
        gc.collect()
        finish = time.time()
        print("Step: {}, execution_time: {}".format(name, finish - start), flush=True)
        
      return test_dataset, columns_info