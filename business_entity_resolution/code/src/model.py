import numpy as np
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier
from sklearn.base import BaseEstimator, ClassifierMixin

class EnsembleModel(BaseEstimator, ClassifierMixin):
    def __init__(self):
        self.lgb = LGBMClassifier(n_estimators=100, learning_rate=0.1, max_depth=6, random_state=42)
        self.xgb = XGBClassifier(n_estimators=100, learning_rate=0.1, max_depth=6, random_state=42, eval_metric='logloss')
        
    def fit(self, X, y, groups=None):
        self.lgb.fit(X, y)
        self.xgb.fit(X, y)
        return self
        
    def predict_proba(self, X):
        p_lgb = self.lgb.predict_proba(X)
        p_xgb = self.xgb.predict_proba(X)
        return (p_lgb + p_xgb) / 2.0
        
    def predict(self, X):
        return (self.predict_proba(X)[:, 1] > 0.5).astype(int)
