import json
import pickle
import sys
from pathlib import Path
from typing import Any, Dict, Tuple

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.pipeline import Pipeline

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SRC_DIR = Path(__file__).resolve().parent
for directory in [PROJECT_ROOT, SRC_DIR]:
    if str(directory) not in sys.path:
        sys.path.insert(0, str(directory))

from src.preprocess import (
    DATA_PATH,
    OUTPUT_PATH,
    TARGET,
    build_preprocessing_pipeline,
    clean_data,
    split_data,
)

MODELS_PATH = PROJECT_ROOT / "models"
SEED = 42
N_FOLDS = 5
MODEL_FILENAME = "best_model.pkl"
CONFIG_FILENAME = "model_config.json"


def build_pipeline(c_reg: float = 0.1, seed: int = SEED) -> Pipeline:

    pipeline = Pipeline(
        steps=[
            ("pre", build_preprocessing_pipeline(scale=True)),
            (
                "model",
                LogisticRegression(
                    C=c_reg,
                    class_weight="balanced",
                    max_iter=1000,
                    random_state=seed,
                ),
            ),
        ]
    )
    return pipeline


def run_cross_validation(
    pipeline: Pipeline, X_train: pd.DataFrame, y_train: pd.Series
) -> Dict[str, float]:
    
    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    scoring = {
        "pr_auc": "average_precision",
        "roc_auc": "roc_auc",
        "f1": "f1",
        "recall": "recall",
        "precision": "precision",
    }

    print(f"Running {N_FOLDS}-Fold Stratified Cross-Validation for Logistic Regression...")
    scores = cross_validate(
        pipeline,
        X_train,
        y_train,
        cv=skf,
        scoring=scoring,
        n_jobs=-1,
        return_train_score=False,
    )

    cv_summary = {
        "CV_PR_AUC_Mean": float(scores["test_pr_auc"].mean()),
        "CV_PR_AUC_Std": float(scores["test_pr_auc"].std()),
        "CV_ROC_AUC_Mean": float(scores["test_roc_auc"].mean()),
        "CV_ROC_AUC_Std": float(scores["test_roc_auc"].std()),
        "CV_F1_Mean": float(scores["test_f1"].mean()),
        "CV_F1_Std": float(scores["test_f1"].std()),
        "CV_Recall_Mean": float(scores["test_recall"].mean()),
        "CV_Precision_Mean": float(scores["test_precision"].mean()),
    }
    return cv_summary


def optimize_threshold(y_true: pd.Series, y_probs: np.ndarray) -> Tuple[float, float]:

    precisions, recalls, thresholds = precision_recall_curve(y_true, y_probs)
    f1_scores = (2 * precisions[:-1] * recalls[:-1]) / (
        precisions[:-1] + recalls[:-1] + 1e-12
    )
    best_idx = int(np.argmax(f1_scores))
    return float(thresholds[best_idx]), float(f1_scores[best_idx])


def evaluate_split(
    pipeline: Pipeline,
    X: pd.DataFrame,
    y: pd.Series,
    threshold: float = 0.50,
    split_name: str = "Val",
) -> Dict[str, Any]:

    y_probs = pipeline.predict_proba(X)[:, 1]
    y_pred = (y_probs >= threshold).astype(int)

    metrics = {
        "split": split_name,
        "threshold": round(threshold, 4),
        "pr_auc": round(float(average_precision_score(y, y_probs)), 4),
        "roc_auc": round(float(roc_auc_score(y, y_probs)), 4),
        "f1": round(float(f1_score(y, y_pred, zero_division=0)), 4),
        "recall": round(float(recall_score(y, y_pred, zero_division=0)), 4),
        "precision": round(float(precision_score(y, y_pred, zero_division=0)), 4),
        "confusion_matrix": confusion_matrix(y, y_pred).tolist(),
    }
    return metrics


def train_and_export():

    train_file = OUTPUT_PATH / "train.csv"
    val_file = OUTPUT_PATH / "val.csv"
    test_file = OUTPUT_PATH / "test.csv"

    if train_file.exists() and val_file.exists() and test_file.exists():
        print("Loading existing stratified splits from data/processed/...")
        train_df = pd.read_csv(train_file)
        val_df = pd.read_csv(val_file)
        test_df = pd.read_csv(test_file)

        X_train, y_train = train_df.drop(columns=[TARGET]), train_df[TARGET]
        X_val, y_val = val_df.drop(columns=[TARGET]), val_df[TARGET]
        X_test, y_test = test_df.drop(columns=[TARGET]), test_df[TARGET]
    else:
        print("Processed splits not found. Running clean_data and split_data...")
        raw_df = pd.read_csv(DATA_PATH)
        df_clean = clean_data(raw_df)
        X_train, X_val, X_test, y_train, y_val, y_test = split_data(df_clean, seed=SEED)

    neg_count = int((y_train == 0).sum())
    pos_count = int((y_train == 1).sum())
    scale_pos_weight = round(neg_count / pos_count, 2)
    print(
        f"Training set: Retained (0) = {neg_count}, Attrited (1) = {pos_count} "
        f"(scale_pos_weight = {scale_pos_weight})"
    )

    pipeline = build_pipeline(c_reg=0.1, seed=SEED)
    cv_summary = run_cross_validation(pipeline, X_train, y_train)

    print("\n" + "=" * 80)
    print("LOGISTIC REGRESSION: 5-FOLD STRATIFIED CROSS-VALIDATION SUMMARY")
    print("=" * 80)
    for k, v in cv_summary.items():
        print(f"{k:20s}: {v:.4f}")
    print("=" * 80 + "\n")

    print("Fitting Logistic Regression pipeline on full training data...")
    pipeline.fit(X_train, y_train)

    val_probs = pipeline.predict_proba(X_val)[:, 1]
    best_thresh, best_val_f1 = optimize_threshold(y_val, val_probs)
    tuned_thresh = round(best_thresh, 2)
    print(f"Optimal decision threshold on validation set: {tuned_thresh} (F1 = {best_val_f1:.4f})")

    val_metrics = evaluate_split(
        pipeline, X_val, y_val, threshold=tuned_thresh, split_name="Validation"
    )
    print("\n" + "=" * 80)
    print("VALIDATION SET METRICS (AT TUNED THRESHOLD)")
    print("=" * 80)
    for k, v in val_metrics.items():
        if k != "confusion_matrix":
            print(f"{k:20s}: {v}")
    print("Confusion Matrix:\n", np.array(val_metrics["confusion_matrix"]))
    print("=" * 80 + "\n")

    test_metrics = evaluate_split(
        pipeline, X_test, y_test, threshold=tuned_thresh, split_name="Test"
    )
    y_test_probs = pipeline.predict_proba(X_test)[:, 1]
    y_test_pred = (y_test_probs >= tuned_thresh).astype(int)

    print("=" * 80)
    print("FINAL HOLDOUT TEST REPORT — LOGISTIC REGRESSION")
    print("=" * 80)
    print(
        f"PR-AUC: {test_metrics['pr_auc']:.4f} | ROC-AUC: {test_metrics['roc_auc']:.4f} | "
        f"F1: {test_metrics['f1']:.4f} | Recall: {test_metrics['recall']:.4f} | "
        f"Precision: {test_metrics['precision']:.4f}"
    )
    print("\nClassification Report:")
    print(classification_report(y_test, y_test_pred, target_names=["Retained (0)", "Attrited (1)"]))
    print("Confusion Matrix:")
    print(np.array(test_metrics["confusion_matrix"]))
    print("=" * 80 + "\n")

    model_path = MODELS_PATH / "best_model.pkl"
    config_path = MODELS_PATH / "model_config.json"

    with open(model_path, "wb") as f:
        pickle.dump(pipeline, f)

    config_data = {
        "model_type": "LogisticRegression",
        "optimal_threshold": tuned_thresh,
        "risk_tiers": {
            "low_upper": 0.40,
            "medium_upper": tuned_thresh,
        },
        "cv_metrics": cv_summary,
        "val_metrics": val_metrics,
        "test_metrics": test_metrics,
    }
    with open(config_path, "w", encoding="utf-8") as f:
        json.dump(config_data, f, indent=4)

    print(f"Model successfully saved")


if __name__ == "__main__":
    train_and_export()
