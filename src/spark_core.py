"""SPARK version 2, core functions for Mathematics only training."""
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, average_precision_score, f1_score,
                             precision_recall_curve, precision_score, recall_score)
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import SVC
from xgboost import XGBClassifier

# ===== Settings =====
RANDOM_STATE = 2026
TARGET_RECALL = 0.90
RISK_ORDER = ["High", "Medium", "Low"]
STAGE_COLS = {"stage1": ["G1"], "stage2": ["G1", "G2"]}

ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = ROOT / "data" / "raw"
OVERLAP_KEYS = ["school", "sex", "age", "address", "famsize", "Pstatus",
                "Medu", "Fedu", "Mjob", "Fjob", "reason", "nursery", "internet"]


# ===== Risk bands and data loading =====
def assign_risk(g3):
    if g3 <= 9:
        return "High"
    elif g3 <= 13:
        return "Medium"
    return "Low"


def load_math():
    df = pd.read_csv(RAW_DIR / "student-mat.csv", sep=";")
    df["risk"] = df["G3"].apply(assign_risk)
    return df


def load_portuguese():
    """Return the full Portuguese file and a copy without students found in the Mathematics file."""
    por = pd.read_csv(RAW_DIR / "student-por.csv", sep=";")
    por["risk"] = por["G3"].apply(assign_risk)

    mat = pd.read_csv(RAW_DIR / "student-mat.csv", sep=";")
    flagged = por.merge(mat[OVERLAP_KEYS].drop_duplicates(),
                        on=OVERLAP_KEYS, how="left", indicator=True)
    clean = (flagged[flagged["_merge"] == "left_only"]
             .drop(columns="_merge").reset_index(drop=True))
    return por, clean


def split_dev_holdout(df, test_size=0.2):
    dev, holdout = train_test_split(df, test_size=test_size, stratify=df["risk"],
                                    random_state=RANDOM_STATE)
    return dev.reset_index(drop=True), holdout.reset_index(drop=True)


# ===== Models and tuning grids =====
MODEL_TEMPLATES = {
    "Decision Tree": DecisionTreeClassifier(class_weight="balanced", random_state=RANDOM_STATE),
    "Random Forest": RandomForestClassifier(n_estimators=300, class_weight="balanced",
                                            random_state=RANDOM_STATE, n_jobs=-1),
    "KNN": KNeighborsClassifier(),
    "Logistic regression": LogisticRegression(max_iter=2000, class_weight="balanced"),
}

TUNING_GRIDS = {
    "Decision Tree": {"clf__max_depth": [2, 3, 4, 5, 6],
                      "clf__min_samples_leaf": [3, 5, 10, 20]},
    "Random Forest": {"clf__max_depth": [3, 5, 8, None],
                      "clf__min_samples_leaf": [1, 3, 5, 10]},
    "KNN": {"clf__n_neighbors": [5, 9, 15, 25, 35],
            "clf__weights": ["uniform", "distance"]},
    "Logistic regression": {"clf__C": [0.1, 0.3, 1, 3, 10, 30, 100]},
}

# ===== Additional baseline models, added after the first hold out look =====
class EncodedLabelClassifier(ClassifierMixin, BaseEstimator):
    """Lets estimators that need integer labels, such as XGBoost, accept text labels."""

    def __init__(self, estimator=None):
        self.estimator = estimator

    def fit(self, X, y):
        self.encoder_ = LabelEncoder().fit(y)
        self.estimator_ = clone(self.estimator).fit(X, self.encoder_.transform(y))
        self.classes_ = self.encoder_.classes_
        return self

    def predict_proba(self, X):
        return self.estimator_.predict_proba(X)

    def predict(self, X):
        return self.encoder_.inverse_transform(self.estimator_.predict(X))


EXTRA_MODEL_TEMPLATES = {
    "SVM": SVC(kernel="rbf", probability=True, class_weight="balanced",
               random_state=RANDOM_STATE),
    "XGBoost": EncodedLabelClassifier(XGBClassifier(
        n_estimators=200, learning_rate=0.05, max_depth=3,
        eval_metric="mlogloss", random_state=RANDOM_STATE, n_jobs=1)),
}

EXTRA_TUNING_GRIDS = {
    "SVM": {"clf__C": [0.1, 1, 10, 100], "clf__gamma": ["scale", 0.01, 0.1, 1]},
    "XGBoost": {"clf__estimator__max_depth": [2, 3, 4],
                "clf__estimator__n_estimators": [100, 200],
                "clf__estimator__learning_rate": [0.03, 0.1]},
}

ALL_TEMPLATES = {**MODEL_TEMPLATES, **EXTRA_MODEL_TEMPLATES}

def make_pipeline(name, **params):
    pipe = Pipeline([("scale", StandardScaler()), ("clf", clone(ALL_TEMPLATES[name]))])
    return pipe.set_params(**params) if params else pipe


def high_avg_precision(estimator, X, y_true):
    """Scorer for GridSearchCV, average precision for the High class."""
    proba = estimator.predict_proba(X)
    col = list(estimator.classes_).index("High")
    return average_precision_score((y_true == "High").astype(int), proba[:, col])


# ===== Out of fold scores and the threshold =====
def oof_high_proba(pipe, X, y, n_repeats=3, n_splits=5):
    """Out of fold probability of High, one array per repeat."""
    X, y = X.reset_index(drop=True), y.reset_index(drop=True)
    runs = []
    for repeat in range(n_repeats):
        cv = StratifiedKFold(n_splits=n_splits, shuffle=True,
                             random_state=RANDOM_STATE + repeat)
        scores = np.zeros(len(y))
        for train_idx, test_idx in cv.split(X, y):
            model = clone(pipe).fit(X.iloc[train_idx], y.iloc[train_idx])
            col = list(model.classes_).index("High")
            scores[test_idx] = model.predict_proba(X.iloc[test_idx])[:, col]
        runs.append(np.round(scores, 4))
    return runs


def pick_threshold(is_high, p_high, target=TARGET_RECALL):
    p_high = np.round(p_high, 4)
    _, recall, thresholds = precision_recall_curve(is_high, p_high)
    reachable = np.where(recall[:-1] >= target)[0]
    return float(thresholds[reachable].max())


# ===== Frozen bundle and scoring of new students =====
def fit_bundle(estimators, dev, target=TARGET_RECALL, n_repeats=3):
    """estimators is a dict with one fitted-ready pipeline for each stage."""
    y = dev["risk"].reset_index(drop=True)
    is_high = (y == "High").astype(int).values
    bundle = {"cols": STAGE_COLS, "models": {}, "thresholds": {}, "target_recall": target}

    for stage, cols in STAGE_COLS.items():
        X = dev[cols].reset_index(drop=True)
        runs = oof_high_proba(estimators[stage], X, y, n_repeats)
        bundle["thresholds"][stage] = pick_threshold(
            np.tile(is_high, len(runs)), np.concatenate(runs), target)
        bundle["models"][stage] = clone(estimators[stage]).fit(X, y)

    return bundle


def score_students(new_data, bundle):
    data = new_data.copy()

    if "G1" not in data.columns:
        raise ValueError("G1 is required")
    if "G2" not in data.columns:
        data["G2"] = np.nan

    for grade in ["G1", "G2"]:
        data[grade] = pd.to_numeric(data[grade], errors="coerce")
    if data["G1"].isna().any():
        raise ValueError("G1 is required for every student")
    if ((data[["G1", "G2"]] < 0) | (data[["G1", "G2"]] > 20)).any().any():
        raise ValueError("Grades must be between 0 and 20")

    out = pd.DataFrame(index=data.index,
                       columns=["stage", "p_high", "p_medium", "p_low", "risk"], dtype=object)
    has_g2 = data["G2"].notna()

    for stage, mask in [("stage2", has_g2), ("stage1", ~has_g2)]:
        if not mask.any():
            continue

        model = bundle["models"][stage]
        classes = list(model.classes_)
        proba = model.predict_proba(data.loc[mask, bundle["cols"][stage]])

        p_high = np.round(proba[:, classes.index("High")], 4)
        p_medium = np.round(proba[:, classes.index("Medium")], 4)
        p_low = np.round(proba[:, classes.index("Low")], 4)

        out.loc[mask, "stage"] = stage
        out.loc[mask, "p_high"] = p_high
        out.loc[mask, "p_medium"] = p_medium
        out.loc[mask, "p_low"] = p_low
        out.loc[mask, "risk"] = np.where(
            p_high >= bundle["thresholds"][stage], "High",
            np.where(p_medium >= p_low, "Medium", "Low"))

    return out


def save_bundle(bundle, path):
    joblib.dump(bundle, path)


def load_bundle(path):
    return joblib.load(path)


# ===== Evaluation helpers =====
def evaluate_predictions(y_true, y_pred):
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    return {
        "recall_high": recall_score(y_true, y_pred, labels=["High"], average="macro", zero_division=0),
        "precision_high": precision_score(y_true, y_pred, labels=["High"], average="macro", zero_division=0),
        "f1_macro": f1_score(y_true, y_pred, average="macro"),
        "accuracy": accuracy_score(y_true, y_pred),
        "flagged_high": int((y_pred == "High").sum()),
        "high_predicted_low": int(((y_true == "High") & (y_pred == "Low")).sum()),
    }

def precision_at_recall(is_high, p_high, target=0.90):
    """Best precision reachable while keeping recall for High at or above the target."""
    precision, recall, _ = precision_recall_curve(is_high, p_high)
    return float(precision[recall >= target].max())

def stage_probabilities(bundle, data, stage):
    """Unrounded class probabilities with columns in alphabetical class order."""
    model = bundle["models"][stage]
    proba = model.predict_proba(data[bundle["cols"][stage]])
    order = [list(model.classes_).index(c) for c in sorted(RISK_ORDER)]
    return proba[:, order]


def full_metrics(y_true, y_pred, proba):
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    return {
        "recall_high": recall_score(y_true, y_pred, labels=["High"], average="macro", zero_division=0),
        "precision_high": precision_score(y_true, y_pred, labels=["High"], average="macro", zero_division=0),
        "high_predicted_low": int(((y_true == "High") & (y_pred == "Low")).sum()),
        "accuracy": accuracy_score(y_true, y_pred),
        "precision_macro": precision_score(y_true, y_pred, labels=RISK_ORDER, average="macro", zero_division=0),
        "recall_macro": recall_score(y_true, y_pred, labels=RISK_ORDER, average="macro", zero_division=0),
        "f1_macro": f1_score(y_true, y_pred, labels=RISK_ORDER, average="macro", zero_division=0),
        "roc_auc_ovr": roc_auc_score(y_true, proba, multi_class="ovr", labels=sorted(RISK_ORDER)),
    }