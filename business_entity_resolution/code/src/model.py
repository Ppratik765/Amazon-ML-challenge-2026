import numpy as np
from lightgbm import LGBMClassifier, early_stopping, log_evaluation
from xgboost import XGBClassifier
try:
    from catboost import CatBoostClassifier
except ImportError:
    CatBoostClassifier = None
from sklearn.isotonic import IsotonicRegression


def _clean(X):
    return np.nan_to_num(np.asarray(X, dtype='float64'), nan=-999.0, posinf=1e6, neginf=-1e6)


class EnsembleModel:
    """
    LightGBM + XGBoost (+ CatBoost if installed) ensemble, each trained once with
    early stopping against a held-out validation split, then calibrated with a
    single-pass isotonic fit on that same validation split's predictions.

    The original version wrapped each of three 1500-tree boosters in a 5-fold
    CalibratedClassifierCV, then stacked all three in a 5-fold StackingClassifier --
    on the order of 75 full booster trainings for one fit() call. That is not a
    reasonable use of limited AWS compute for a marginal accuracy gain over this.
    This version trains each booster exactly once (early stopping picks the tree
    count) and calibrates once, so the cost is roughly the cost of training the
    three boosters a single time each.

    scale_pos_weight is derived from the actual class balance in the training data
    rather than a hand-picked constant. The precision/recall trade-off for F0.5
    should be controlled by the decision threshold (tuned directly against the
    F0.5 metric in train.py), not by distorting the probabilities the threshold is
    applied to -- doing both at once (as the original scale_pos_weight=0.6 +
    isotonic calibration + threshold search did) fights itself, since calibration
    is trying to undo exactly the distortion the reweighting introduced.
    """

    def __init__(self, n_estimators=2000, learning_rate=0.03, max_depth=8,
                 use_catboost=True, early_stopping_rounds=100, random_state=42):
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.use_catboost = use_catboost
        self.early_stopping_rounds = early_stopping_rounds
        self.random_state = random_state
        self.models = {}
        self.calibrators = {}
        self.feature_importances_ = None

    def fit(self, X_train, y_train, X_val, y_val, feature_names=None):
        X_train = _clean(X_train)
        X_val = _clean(X_val)
        y_train = np.asarray(y_train)
        y_val = np.asarray(y_val)

        n_pos = max(1, int(y_train.sum()))
        n_neg = max(1, int(len(y_train) - n_pos))
        scale_pos_weight = n_neg / n_pos

        lgb = LGBMClassifier(
            n_estimators=self.n_estimators,
            learning_rate=self.learning_rate,
            max_depth=self.max_depth,
            num_leaves=min(255, 2 ** self.max_depth - 1),
            subsample=0.8,
            colsample_bytree=0.8,
            reg_alpha=0.1,
            reg_lambda=1.0,
            scale_pos_weight=scale_pos_weight,
            random_state=self.random_state,
            n_jobs=-1,
            verbosity=-1,
        )
        lgb.fit(X_train, y_train, eval_set=[(X_val, y_val)], eval_metric='average_precision',
                callbacks=[early_stopping(self.early_stopping_rounds, verbose=False), log_evaluation(0)])
        self.models['lgb'] = lgb

        xgb = XGBClassifier(
            n_estimators=self.n_estimators,
            learning_rate=self.learning_rate,
            max_depth=self.max_depth,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_alpha=0.1,
            reg_lambda=1.0,
            scale_pos_weight=scale_pos_weight,
            random_state=self.random_state,
            eval_metric='aucpr',
            early_stopping_rounds=self.early_stopping_rounds,
            n_jobs=-1,
        )
        xgb.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
        self.models['xgb'] = xgb

        if self.use_catboost and CatBoostClassifier is not None:
            cat = CatBoostClassifier(
                iterations=self.n_estimators,
                learning_rate=self.learning_rate,
                depth=min(self.max_depth, 10),
                subsample=0.8,
                colsample_bylevel=0.8,
                l2_leaf_reg=3,
                scale_pos_weight=scale_pos_weight,
                random_seed=self.random_state,
                early_stopping_rounds=self.early_stopping_rounds,
                verbose=False,
                thread_count=-1,
            )
            cat.fit(X_train, y_train, eval_set=(X_val, y_val), verbose=False)
            self.models['cat'] = cat

        # Single-pass isotonic calibration per model, fit on the same validation
        # split used for early stopping. (A fully independent third split would be
        # more principled still; given limited compute this is the practical
        # middle ground -- see train.py for the separate, never-touched holdout
        # used for threshold tuning and the final reported F0.5.)
        for name, m in self.models.items():
            raw_val_pred = m.predict_proba(X_val)[:, 1]
            iso = IsotonicRegression(out_of_bounds='clip')
            iso.fit(raw_val_pred, y_val)
            self.calibrators[name] = iso

        if feature_names is not None and 'lgb' in self.models:
            self.feature_importances_ = dict(zip(feature_names, self.models['lgb'].feature_importances_))

        return self

    def predict_proba(self, X):
        X = _clean(X)
        probs = []
        for name, m in self.models.items():
            raw = m.predict_proba(X)[:, 1]
            probs.append(self.calibrators[name].predict(raw))
        avg = np.mean(probs, axis=0)
        return np.stack([1 - avg, avg], axis=1)
