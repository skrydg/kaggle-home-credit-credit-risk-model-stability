import time

class Preprocessor:
    def __init__(self, steps):
      self.steps = steps
    
    def process_train_dataset(self, train_dataset):
      for name, step in self.steps.items():
        start = time.time()
        train_dataset = step.process_train_dataset(train_dataset)
        finish = time.time()
        print("Step: {}, execution_time: {}".format(name, finish - start), flush=True)
      return train_dataset
    
    def process_test_dataset(self, test_dataset):
      for name, step in self.steps.items():
        start = time.time()
        test_dataset = step.process_test_dataset(test_dataset)
        finish = time.time()
        print("Step: {}, execution_time: {}".format(name, finish - start), flush=True)
      return test_dataset