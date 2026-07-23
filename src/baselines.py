"""Classical baselines for the sentiment classification task.

These exist to answer one question honestly: how good is "good" on this dataset?
A from-scratch 28M-param GPT classifier sounds impressive; a TF-IDF+SVM baseline
tells us whether the GPT is actually adding value over a lexical model, and a
5-fold CV estimate tells us whether a single-split number is signal or noise.
"""
from typing import Dict, Tuple

import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.svm import LinearSVC


def majority_baseline(train_labels, val_labels) -> Dict:
    majority = pd.Series(train_labels).value_counts().idxmax()
    preds = [majority] * len(val_labels)
    return {
        "acc": accuracy_score(val_labels, preds),
        "macro_f1": f1_score(val_labels, preds, average="macro"),
        "majority_class": int(majority),
    }


def tfidf_features(train_texts, val_texts, ngram: Tuple[int, int] = (1, 2), min_df: int = 2):
    """Fits the vectorizer on TRAIN ONLY -- val_texts only ever calls .transform()."""
    vec = TfidfVectorizer(ngram_range=ngram, min_df=min_df, sublinear_tf=True)
    Xtr = vec.fit_transform(train_texts)
    Xva = vec.transform(val_texts)
    return Xtr, Xva, vec


def _make_model(model: str):
    if model == "logreg":
        return LogisticRegression(max_iter=2000, C=3.0, class_weight="balanced")
    if model == "svc":
        # LinearSVC has no predict_proba; calibrate it so its output can feed the ensemble.
        return CalibratedClassifierCV(LinearSVC(C=1.0, class_weight="balanced"), cv=3)
    raise ValueError(f"unknown model: {model}")


def tfidf_baseline(train_df: pd.DataFrame, val_df: pd.DataFrame, model: str = "svc", ngram=(1, 2), min_df=2) -> Dict:
    Xtr, Xva, vec = tfidf_features(train_df["text"], val_df["text"], ngram=ngram, min_df=min_df)
    clf = _make_model(model)
    clf.fit(Xtr, train_df["label_id"])
    preds = clf.predict(Xva)
    probs = clf.predict_proba(Xva)
    return {
        "preds": preds,
        "probs": probs,
        "acc": accuracy_score(val_df["label_id"], preds),
        "macro_f1": f1_score(val_df["label_id"], preds, average="macro"),
        "vectorizer": vec,
        "clf": clf,
    }


def tfidf_baseline_cv(df: pd.DataFrame, model: str = "svc", ngram=(1, 2), min_df=2, n_splits: int = 5, seed: int = 42) -> Dict:
    """Stratified k-fold CV estimate -- the honest ceiling, not a single lucky/unlucky split."""
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    fold_accs, fold_f1s = [], []
    for train_idx, val_idx in skf.split(df["text"], df["label_id"]):
        tr, va = df.iloc[train_idx], df.iloc[val_idx]
        Xtr, Xva, _ = tfidf_features(tr["text"], va["text"], ngram=ngram, min_df=min_df)
        clf = _make_model(model)
        clf.fit(Xtr, tr["label_id"])
        preds = clf.predict(Xva)
        fold_accs.append(accuracy_score(va["label_id"], preds))
        fold_f1s.append(f1_score(va["label_id"], preds, average="macro"))
    return {
        "acc_mean": float(np.mean(fold_accs)),
        "acc_std": float(np.std(fold_accs)),
        "macro_f1_mean": float(np.mean(fold_f1s)),
        "macro_f1_std": float(np.std(fold_f1s)),
        "n_splits": n_splits,
    }


def fit_tfidf_oof_probs(train_df: pd.DataFrame, model: str = "logreg", ngram=(1, 2), min_df=2, n_splits: int = 5, seed: int = 42) -> np.ndarray:
    """Out-of-fold TF-IDF probabilities on the train split, for picking an ensemble
    weight without ever touching the held-out val set (no tuning-on-test).
    """
    vec = TfidfVectorizer(ngram_range=ngram, min_df=min_df, sublinear_tf=True)
    X = vec.fit_transform(train_df["text"])
    clf = _make_model(model)
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    oof_probs = cross_val_predict(clf, X, train_df["label_id"], cv=skf, method="predict_proba")
    return oof_probs
