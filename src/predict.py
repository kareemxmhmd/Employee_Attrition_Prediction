import json
import pickle
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SRC_DIR = Path(__file__).resolve().parent
for directory in [PROJECT_ROOT, SRC_DIR]:
    if str(directory) not in sys.path:
        sys.path.insert(0, str(directory))

try:
    from src.preprocess import add_features, clean_data
except ModuleNotFoundError:
    from preprocess import add_features, clean_data

DEFAULT_MODEL_PATH = PROJECT_ROOT / "models" / "best_model.pkl"
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "models" / "model_config.json"


def load_config(config_path: Union[str, Path] = DEFAULT_CONFIG_PATH) -> Dict[str, Any]:
    path = Path(config_path)
    if path.exists():
        try:
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            pass
    return {"optimal_threshold": 0.67, "risk_tiers": {"low_upper": 0.40, "medium_upper": 0.67}}


CONFIG = load_config()
DEFAULT_THRESHOLD = float(CONFIG.get("optimal_threshold", 0.67))


def load_model(model_path: Union[str, Path] = DEFAULT_MODEL_PATH) -> Pipeline:
    path = Path(model_path)
    if not path.exists():
        fallback = path.with_suffix(".joblib")
        if fallback.exists():
            path = fallback
        else:
            raise FileNotFoundError(f"Model file not found at: {path.resolve()}")
    with open(path, "rb") as f:
        model = pickle.load(f)
    return model


def get_risk_level(probability: float, threshold: float = DEFAULT_THRESHOLD) -> str:
    if probability < 0.40:
        return "Low"
    elif probability < threshold:
        return "Medium"
    return "High"


def explain_prediction(model: Pipeline, processed_df: pd.DataFrame, top_k: int = 3) -> Dict[str, List[Dict[str, Any]]]:
    """Extract top contributing features increasing or decreasing attrition hazard for this instance."""
    try:
        pre = model.named_steps.get("pre", model.named_steps.get("prep"))
        clf = model.named_steps.get("model", model.named_steps.get("clf"))
        if not hasattr(clf, "coef_"):
            return {"top_risk_drivers": [], "top_retention_factors": []}

        X_transformed = pre.transform(processed_df)
        if hasattr(X_transformed, "toarray"):
            X_vec = X_transformed.toarray()[0]
        else:
            X_vec = np.asarray(X_transformed)[0]

        col_transformer = pre.named_steps.get("preprocessor", pre) if hasattr(pre, "named_steps") else pre
        feature_names = [
            f.replace("num__", "").replace("cat__", "").replace("bin__", "")
            for f in col_transformer.get_feature_names_out()
        ]
        coefs = clf.coef_[0]
        contributions = X_vec * coefs

        contrib_df = pd.DataFrame({
            "feature": feature_names,
            "contribution": contributions,
            "odds_ratio": np.exp(coefs)
        })

        # Non-zero active contributions
        active_contribs = contrib_df[contrib_df["contribution"] != 0].copy()
        risk_drivers = (
            active_contribs.sort_values(by="contribution", ascending=False)
            .head(top_k)[["feature", "contribution", "odds_ratio"]]
            .to_dict(orient="records")
        )
        retention_factors = (
            active_contribs.sort_values(by="contribution", ascending=True)
            .head(top_k)[["feature", "contribution", "odds_ratio"]]
            .to_dict(orient="records")
        )

        return {
            "top_risk_drivers": risk_drivers,
            "top_retention_factors": retention_factors
        }
    except Exception:
        return {"top_risk_drivers": [], "top_retention_factors": []}


def predict_single(
    employee_data: Dict[str, Any],
    model: Optional[Pipeline] = None,
    threshold: float = DEFAULT_THRESHOLD,
) -> Dict[str, Any]:
    if model is None:
        model = load_model()

    df = pd.DataFrame([employee_data])
    processed_df = add_features(clean_data(df, drop_duplicates=False))

    proba = float(model.predict_proba(processed_df)[:, 1][0])
    prediction = int(proba >= threshold)
    explanation = explain_prediction(model, processed_df)

    return {
        "attrition_prediction": prediction,
        "attrition_label": "Attrition" if prediction == 1 else "Stay",
        "attrition_probability": round(proba, 4),
        "risk_level": get_risk_level(proba, threshold=threshold),
        "threshold_used": threshold,
        "explanation": explanation,
    }


def predict_batch(
    df: pd.DataFrame,
    model: Optional[Pipeline] = None,
    threshold: float = DEFAULT_THRESHOLD,
) -> pd.DataFrame:
    if model is None:
        model = load_model()

    original_df = df.copy()
    processed_df = add_features(clean_data(original_df, drop_duplicates=False))

    probabilities = model.predict_proba(processed_df)[:, 1]
    predictions = (probabilities >= threshold).astype(int)

    results_df = original_df.copy()
    results_df["attrition_probability"] = [round(float(p), 4) for p in probabilities]
    results_df["attrition_prediction"] = predictions
    results_df["attrition_label"] = [
        "Attrition" if p == 1 else "Stay" for p in predictions
    ]
    results_df["risk_level"] = [get_risk_level(p, threshold=threshold) for p in probabilities]

    return results_df


if __name__ == "__main__":
    print("Testing inference module...")
    pipeline = load_model()

    # Sample 1: High-risk employee
    sample_high_risk = {
        "Age": 28,
        "BusinessTravel": "Travel_Frequently",
        "DailyRate": 400,
        "Department": "Sales",
        "DistanceFromHome": 25,
        "Education": 2,
        "EducationField": "Marketing",
        "EnvironmentSatisfaction": 1,
        "Gender": "Male",
        "HourlyRate": 45,
        "JobInvolvement": 1,
        "JobLevel": 1,
        "JobRole": "Sales Representative",
        "JobSatisfaction": 1,
        "MaritalStatus": "Single",
        "MonthlyIncome": 2200,
        "MonthlyRate": 15000,
        "NumCompaniesWorked": 6,
        "OverTime": "Yes",
        "PercentSalaryHike": 11,
        "PerformanceRating": 3,
        "RelationshipSatisfaction": 1,
        "StockOptionLevel": 0,
        "TotalWorkingYears": 3,
        "TrainingTimesLastYear": 1,
        "WorkLifeBalance": 1,
        "YearsAtCompany": 1,
        "YearsInCurrentRole": 0,
        "YearsSinceLastPromotion": 0,
        "YearsWithCurrManager": 0,
    }

    # Sample 2: Low-risk employee
    sample_low_risk = {
        "Age": 48,
        "BusinessTravel": "Non-Travel",
        "DailyRate": 1200,
        "Department": "Research & Development",
        "DistanceFromHome": 2,
        "Education": 4,
        "EducationField": "Life Sciences",
        "EnvironmentSatisfaction": 4,
        "Gender": "Female",
        "HourlyRate": 90,
        "JobInvolvement": 4,
        "JobLevel": 4,
        "JobRole": "Research Director",
        "JobSatisfaction": 4,
        "MaritalStatus": "Married",
        "MonthlyIncome": 16000,
        "MonthlyRate": 20000,
        "NumCompaniesWorked": 1,
        "OverTime": "No",
        "PercentSalaryHike": 20,
        "PerformanceRating": 4,
        "RelationshipSatisfaction": 4,
        "StockOptionLevel": 2,
        "TotalWorkingYears": 22,
        "TrainingTimesLastYear": 4,
        "WorkLifeBalance": 4,
        "YearsAtCompany": 15,
        "YearsInCurrentRole": 10,
        "YearsSinceLastPromotion": 1,
        "YearsWithCurrManager": 10,
    }

    print("\n--- High Risk Inference ---")
    res_high = predict_single(sample_high_risk, model=pipeline)
    print("Prediction:", res_high["attrition_label"])
    print("Probability:", res_high["attrition_probability"])
    print("Risk Level:", res_high["risk_level"])
    print("Top Risk Drivers:", [d["feature"] for d in res_high["explanation"]["top_risk_drivers"]])

    print("\n--- Low Risk Inference ---")
    res_low = predict_single(sample_low_risk, model=pipeline)
    print("Prediction:", res_low["attrition_label"])
    print("Probability:", res_low["attrition_probability"])
    print("Risk Level:", res_low["risk_level"])
    print("Top Retention Factors:", [d["feature"] for d in res_low["explanation"]["top_retention_factors"]])
