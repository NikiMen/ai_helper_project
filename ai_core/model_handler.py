from sklearn.linear_model import LinearRegression
import joblib
import os

MODEL_PATH = "../models/ai_model.pkl"

class ModelHandler:
    def __init__(self):
        self.model = None

    def train(self, X, y):
        self.model = LinearRegression()
        self.model.fit(X, y)
        self.save_model()
        return self.model.coef_, self.model.intercept_

    def predict(self, input_data):
        if self.model is None:
            raise ValueError("Модель не обучена.")
        return self.model.predict(input_data)

    def save_model(self):
        joblib.dump(self.model, MODEL_PATH)

    def load_model(self):
        if not os.path.exists(MODEL_PATH):
            return False
        self.model = joblib.load(MODEL_PATH)
        return True