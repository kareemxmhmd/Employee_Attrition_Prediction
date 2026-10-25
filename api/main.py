import io
import sys
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, Dict, List, Optional

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd
from fastapi import FastAPI, File, HTTPException, Query, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel, Field

from src.predict import (
    DEFAULT_THRESHOLD,
    load_model,
    predict_batch,
    predict_single,
)

MODEL = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global MODEL
    try:
        MODEL = load_model()
        print("Model pipeline loaded successfully on startup.")
    except Exception as e:
        print(f"Warning: Model could not be loaded on startup: {e}")
    yield


app = FastAPI(
    title="Employee Attrition Prediction API",
    description="Production-ready REST API for predicting employee attrition using Regularized Logistic Regression with odds ratio interpretability.",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class EmployeeInput(BaseModel):
    Age: int = Field(35, ge=18, le=100, description="Age of employee in years")
    BusinessTravel: str = Field("Travel_Rarely", description="Travel_Rarely, Travel_Frequently, Non-Travel")
    DailyRate: float = Field(800.0, ge=0, description="Daily rate of compensation")
    Department: str = Field("Research & Development", description="Sales, Research & Development, Human Resources")
    DistanceFromHome: float = Field(5.0, ge=0, description="Commute distance from home in miles")
    Education: int = Field(3, ge=1, le=5, description="Education level (1=Below College, 2=College, 3=Bachelor, 4=Master, 5=Doctor)")
    EducationField: str = Field("Life Sciences", description="Life Sciences, Medical, Marketing, Technical Degree, Other, Human Resources")
    EnvironmentSatisfaction: int = Field(3, ge=1, le=4, description="Environment satisfaction rating (1=Low to 4=Very High)")
    Gender: str = Field("Male", description="Gender (Male, Female)")
    HourlyRate: float = Field(65.0, ge=0, description="Hourly rate of compensation")
    JobInvolvement: int = Field(3, ge=1, le=4, description="Job involvement rating (1=Low to 4=Very High)")
    JobLevel: int = Field(2, ge=1, le=5, description="Job level (1 to 5)")
    JobRole: str = Field("Research Scientist", description="Specific job role")
    JobSatisfaction: int = Field(3, ge=1, le=4, description="Job satisfaction rating (1=Low to 4=Very High)")
    MaritalStatus: str = Field("Married", description="Marital status (Single, Married, Divorced)")
    MonthlyIncome: float = Field(5000.0, ge=0, description="Monthly income in USD")
    MonthlyRate: float = Field(14000.0, ge=0, description="Monthly rate")
    NumCompaniesWorked: int = Field(2, ge=0, description="Number of prior companies worked at")
    OverTime: str = Field("No", description="Overtime status (Yes, No)")
    PercentSalaryHike: float = Field(15.0, ge=0, description="Percentage of last salary hike")
    PerformanceRating: int = Field(3, ge=1, le=4, description="Performance rating (1 to 4)")
    RelationshipSatisfaction: int = Field(3, ge=1, le=4, description="Relationship satisfaction rating (1 to 4)")
    StockOptionLevel: int = Field(1, ge=0, le=3, description="Stock option level (0 to 3)")
    TotalWorkingYears: float = Field(10.0, ge=0, description="Total working career years")
    TrainingTimesLastYear: int = Field(3, ge=0, description="Number of training sessions attended last year")
    WorkLifeBalance: int = Field(3, ge=1, le=4, description="Work-life balance rating (1=Bad to 4=Best)")
    YearsAtCompany: float = Field(5.0, ge=0, description="Total years at current company")
    YearsInCurrentRole: float = Field(3.0, ge=0, description="Years in current role")
    YearsSinceLastPromotion: float = Field(1.0, ge=0, description="Years elapsed since last promotion")
    YearsWithCurrManager: float = Field(3.0, ge=0, description="Years with current manager")

    model_config = {
        "json_schema_extra": {
            "example": {
                "Age": 29,
                "BusinessTravel": "Travel_Frequently",
                "DailyRate": 450.0,
                "Department": "Sales",
                "DistanceFromHome": 20.0,
                "Education": 2,
                "EducationField": "Marketing",
                "EnvironmentSatisfaction": 1,
                "Gender": "Male",
                "HourlyRate": 48.0,
                "JobInvolvement": 2,
                "JobLevel": 1,
                "JobRole": "Sales Representative",
                "JobSatisfaction": 1,
                "MaritalStatus": "Single",
                "MonthlyIncome": 2700.0,
                "MonthlyRate": 12500.0,
                "NumCompaniesWorked": 3,
                "OverTime": "Yes",
                "PercentSalaryHike": 12.0,
                "PerformanceRating": 3,
                "RelationshipSatisfaction": 2,
                "StockOptionLevel": 0,
                "TotalWorkingYears": 4.0,
                "TrainingTimesLastYear": 1,
                "WorkLifeBalance": 1,
                "YearsAtCompany": 1.0,
                "YearsInCurrentRole": 0.0,
                "YearsSinceLastPromotion": 0.0,
                "YearsWithCurrManager": 0.0,
            }
        }
    }


class PredictionOutput(BaseModel):
    attrition_prediction: int
    attrition_label: str
    attrition_probability: float
    risk_level: str
    threshold_used: float
    explanation: Optional[Dict[str, Any]] = None


@app.get("/", tags=["Health"])
@app.get("/health", tags=["Health"])
def health_check() -> Dict[str, Any]:
    global MODEL
    if MODEL is None:
        try:
            MODEL = load_model()
        except Exception:
            pass
    is_ready = MODEL is not None
    return {
        "status": "healthy" if is_ready else "degraded",
        "model_loaded": is_ready,
        "api_version": "1.0.0",
        "model_type": "Regularized Logistic Regression Pipeline",
    }


@app.post("/predict", response_model=PredictionOutput, tags=["Prediction"])
def predict(
    employee: EmployeeInput,
    threshold: float = Query(
        DEFAULT_THRESHOLD,
        ge=0.0,
        le=1.0,
        description="Decision threshold for attrition classification",
    ),
) -> PredictionOutput:
    global MODEL
    if MODEL is None:
        try:
            MODEL = load_model()
        except Exception as e:
            raise HTTPException(status_code=503, detail=f"Model not loaded: {str(e)}")

    try:
        result = predict_single(employee.model_dump(), model=MODEL, threshold=threshold)
        return PredictionOutput(**result)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Inference error: {str(e)}")


@app.post("/predict/batch", tags=["Prediction"])
async def predict_csv(
    file: UploadFile = File(..., description="CSV file containing employee records"),
    threshold: float = Query(DEFAULT_THRESHOLD, ge=0.0, le=1.0, description="Decision threshold"),
):
    global MODEL
    if MODEL is None:
        try:
            MODEL = load_model()
        except Exception as e:
            raise HTTPException(status_code=503, detail=f"Model not loaded: {str(e)}")

    if not file.filename.endswith(".csv"):
        raise HTTPException(status_code=400, detail="Only CSV files are supported.")

    try:
        contents = await file.read()
        df = pd.read_csv(io.BytesIO(contents))
        predictions_df = predict_batch(df, model=MODEL, threshold=threshold)

        output_buffer = io.StringIO()
        predictions_df.to_csv(output_buffer, index=False)
        output_buffer.seek(0)

        response = StreamingResponse(
            iter([output_buffer.getvalue()]),
            media_type="text/csv",
            headers={"Content-Disposition": f"attachment; filename=predictions_{file.filename}"},
        )
        return response
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error processing CSV: {str(e)}")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api.main:app", host="127.0.0.1", port=8000, reload=True, app_dir=str(PROJECT_ROOT))
