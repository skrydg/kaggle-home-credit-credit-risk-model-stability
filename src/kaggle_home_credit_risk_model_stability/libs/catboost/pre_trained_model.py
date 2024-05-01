class PreTrainedCatboostModel:
    def __init__(self, model):
        self.model = model

    def predict(self, *args, **kwargs):
        return self.model.predict_proba(*args, **kwargs)[:, 1]
