from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, RobustScaler

DATA_PATH = Path("data/raw/HR-Employee-Attrition.csv")
OUTPUT_PATH = Path("data/processed")

TARGET = "Attrition"
DROP_COLS = ["EmployeeCount", "StandardHours", "Over18", "EmployeeNumber"]

CONTINUOUS_NUMERIC = [
    "Age",
    "DailyRate",
    "DistanceFromHome",
    "Education",
    "EnvironmentSatisfaction",
    "HourlyRate",
    "JobInvolvement",
    "JobLevel",
    "JobSatisfaction",
    "MonthlyIncome",
    "MonthlyRate",
    "NumCompaniesWorked",
    "PercentSalaryHike",
    "PerformanceRating",
    "RelationshipSatisfaction",
    "StockOptionLevel",
    "TotalWorkingYears",
    "TrainingTimesLastYear",
    "WorkLifeBalance",
    "YearsAtCompany",
    "YearsInCurrentRole",
    "YearsSinceLastPromotion",
    "YearsWithCurrManager",
    # Engineered continuous features
    "income_per_working_year",
    "role_to_company_ratio",
]

BINARY_INDICATORS = [
    "overtime_and_low_wlb",
    "is_first_year",
]

CATEGORICAL = [
    "BusinessTravel",
    "Department",
    "EducationField",
    "Gender",
    "JobRole",
    "MaritalStatus",
    "OverTime",
    "tenure_band",
]


def clean_data(df: pd.DataFrame, drop_duplicates: bool = True) -> pd.DataFrame:
    df = df.copy()

    cols_to_drop = [c for c in DROP_COLS if c in df.columns]
    if cols_to_drop:
        df = df.drop(columns=cols_to_drop)

    str_cols = df.select_dtypes(include=["object", "string"]).columns
    for col in str_cols:
        df[col] = df[col].astype(str).str.strip().replace({"nan": np.nan, "<NA>": np.nan})

    if TARGET in df.columns:
        if not pd.api.types.is_numeric_dtype(df[TARGET]):
            df[TARGET] = (
                df[TARGET]
                .astype(str)
                .str.strip()
                .str.lower()
                .map({"yes": 1, "no": 0, "1": 1, "0": 0})
            )
        df = df.dropna(subset=[TARGET])
        df[TARGET] = df[TARGET].astype(int)

    return df


class HRFeatureEngineer(TransformerMixin, BaseEstimator):

    def fit(self, X: pd.DataFrame, y=None):
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        df = X.copy()

        total_years = df["TotalWorkingYears"].fillna(0).clip(lower=0) + 1.0
        company_years = df["YearsAtCompany"].fillna(0).clip(lower=0) + 1.0

        df["income_per_working_year"] = df["MonthlyIncome"] / total_years
        df["role_to_company_ratio"] = df["YearsInCurrentRole"] / company_years

        if "OverTime" in df.columns:
            overtime_bool = df["OverTime"].astype(str).str.strip().str.lower() == "yes"
        else:
            overtime_bool = pd.Series(False, index=df.index)

        wlb_val = df["WorkLifeBalance"].fillna(3) if "WorkLifeBalance" in df.columns else 3
        df["overtime_and_low_wlb"] = (overtime_bool & (wlb_val <= 2)).fillna(False).astype(int)

        df["is_first_year"] = (df["YearsAtCompany"].fillna(99) <= 1).astype(int)

        bands = pd.cut(
            df["YearsAtCompany"],
            bins=[-np.inf, 1, 3, 5, 10, np.inf],
            labels=["0-1 yr", "2-3 yrs", "4-5 yrs", "6-10 yrs", "11+ yrs"],
        )
        df["tenure_band"] = bands.astype(object).fillna("Unknown")

        return df

    def set_output(self, *, transform=None):
        """Compatibility hook for sklearn Pipeline.set_output(transform='pandas')."""
        return self


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    """Helper function to apply HRFeatureEngineer transformations."""
    return HRFeatureEngineer().transform(df)


def split_data(
    df: pd.DataFrame,
    train_size: float = 0.70,
    val_size: float = 0.15,
    test_size: float = 0.15,
    seed: int = 42,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, pd.Series]:
    
    if not np.isclose(train_size + val_size + test_size, 1.0):
        raise ValueError("train_size, val_size, and test_size must sum to 1.0")

    X = df.drop(columns=[TARGET])
    y = df[TARGET]

    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, train_size=train_size, stratify=y, random_state=seed
    )

    relative_val_size = val_size / (val_size + test_size)
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, train_size=relative_val_size, stratify=y_temp, random_state=seed
    )

    return X_train, X_val, X_test, y_train, y_val, y_test


def build_preprocessing_pipeline(scale: bool = False) -> Pipeline:
    
    num_steps = [("impute", SimpleImputer(strategy="median"))]
    if scale:
        num_steps.append(("scale", RobustScaler()))

    bin_steps = [
        ("impute", SimpleImputer(strategy="most_frequent")),
    ]

    cat_steps = [
        ("impute", SimpleImputer(strategy="most_frequent")),
        (
            "onehot",
            OneHotEncoder(
                drop="if_binary",
                handle_unknown="ignore",
                sparse_output=False,
            ),
        ),
    ]

    col_transformer = ColumnTransformer(
        transformers=[
            ("num", Pipeline(num_steps), CONTINUOUS_NUMERIC),
            ("bin", Pipeline(bin_steps), BINARY_INDICATORS),
            ("cat", Pipeline(cat_steps), CATEGORICAL),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )
    col_transformer.set_output(transform="pandas")

    full_pipeline = Pipeline(
        steps=[
            ("feature_engineering", HRFeatureEngineer()),
            ("preprocessor", col_transformer),
        ]
    )
    return full_pipeline


if __name__ == "__main__":
    raw = pd.read_csv(DATA_PATH)
    print(f"Raw dataset loaded: shape={raw.shape}, exact_duplicates={raw.duplicated().sum()}")

    # 1. Clean raw data (keep raw features intact before splitting)
    df_clean = clean_data(raw)
    df_clean.to_csv(OUTPUT_PATH / "clean.csv", index=False)
    print(f"Cleaned dataset saved: shape={df_clean.shape}")

    # 2. Split on raw cleaned features (prevents any future data leakage)
    X_train, X_val, X_test, y_train, y_val, y_test = split_data(df_clean)

    for split_name, X_part, y_part in [
        ("train", X_train, y_train),
        ("val", X_val, y_val),
        ("test", X_test, y_test),
    ]:
        pd.concat([X_part, y_part], axis=1).to_csv(OUTPUT_PATH / f"{split_name}.csv", index=False)
        print(f"Split [{split_name}]: n={len(y_part)}, attrition_rate={y_part.mean():.4f}")

    # 3. Fit & Transform using the unified pipeline
    preprocessor = build_preprocessing_pipeline(scale=False)
    X_train_t = preprocessor.fit_transform(X_train, y_train)
    X_val_t = preprocessor.transform(X_val)
    X_test_t = preprocessor.transform(X_test)

    print(f"Transformed shapes -> Train: {X_train_t.shape} | Val: {X_val_t.shape} | Test: {X_test_t.shape}")