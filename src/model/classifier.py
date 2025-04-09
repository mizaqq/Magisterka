import xgboost as xgb


class XGBoost:
    def __init__(self, **config):
        self.model = xgb.XGBClassifier(**config)

    def fit(self, X, y):
        self.model.fit(X, y)

    def predict(self, X):
        return self.model.predict(X)

    def predict_proba(self, X):
        return self.model.predict_proba(X)

    def save_model(self, path):
        self.model.save_model(path)

    def load_model(self, path):
        self.model.load_model(path)

    def get_feature_importance(self):
        return self.model.get_booster().get_score(importance_type='weight')

    def get_params(self):
        return self.model.get_params()
