import xgboost as xgb
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report


class XGBoost:
    def __init__(self, **config):
        self.model = xgb.XGBClassifier(
            **config,
        )

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

    def test_model(self, X, y):
        predictions = self.predict(X)
        accuracy = accuracy_score(y, predictions)
        precision = precision_score(y, predictions, average='weighted', zero_division=0)
        recall = recall_score(y, predictions, average='weighted', zero_division=0)
        f1 = f1_score(y, predictions, average='weighted', zero_division=0)
        cr = classification_report(y, predictions, zero_division=0)
        return {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'classification_report': cr,
        }
