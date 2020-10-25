from base_kam_model import BaseModel, BaseEvaluation
import torch
import numpy as np


class TianModel(BaseModel):
    @staticmethod
    def preprocessing(train_data, test_data):
        return train_data, test_data

    @staticmethod
    def train_model(X_train, Y_train):
        x_train_tensor = torch.from_numpy(X_train)
        x=1

    @staticmethod
    def predict(model, X_test):
        raise RuntimeError('Method not implemented')




if __name__ == "__main__":
    model = TianModel(BaseEvaluation)
    model.param_tuning([0, 1, 2])
