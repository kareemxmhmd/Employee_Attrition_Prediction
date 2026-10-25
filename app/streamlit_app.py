import io
import sys
from pathlib import Path
from typing import Any, Dict

import pandas as pd
import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SRC_DIR = PROJECT_ROOT / "src"
for directory in [PROJECT_ROOT, SRC_DIR]:
    if str(directory) not in sys.path:
        sys.path.insert(0, str(directory))

from src.predict import DEFAULT_THRESHOLD, load_model, predict_batch, predict_single


@st.cache_resource(show_spinner="Loading machine learning model pipeline...")
def get_model():
    """Cache loaded model in memory to avoid redundant disk I/O on every interaction."""
    return load_model()


st.set_page_config(
    page_title="Employee Attrition Predictor",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    .metric-card {
        padding: 1.2rem;
        border-radius: 0.8rem;
        background-color: #f8f9fa;
        border-left: 5px solid #007bff;
        margin-bottom: 1rem;
    }
    .badge-high {
        background-color: #dc3545;
        color: white;
        padding: 0.4rem 0.8rem;
        border-radius: 0.4rem;
        font-weight: bold;
    }
    .badge-med {
        background-color: #ffc107;
        color: #212529;
        padding: 0.4rem 0.8rem;
        border-radius: 0.4rem;
        font-weight: bold;
    }
    .badge-low {
        background-color: #28a745;
        color: white;
        padding: 0.4rem 0.8rem;
        border-radius: 0.4rem;
        font-weight: bold;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.sidebar.subheader("System Status")
try:
    _ = get_model()
    st.sidebar.success("Model: Loaded & Ready (Standalone)")
except Exception as e:
    st.sidebar.error(f"Model load error: {e}")

st.sidebar.subheader("Prediction Settings")
threshold = st.sidebar.slider(
    "Decision Threshold",
    min_value=0.10,
    max_value=0.90,
    value=float(DEFAULT_THRESHOLD),
    step=0.01,
    help="Employees with predicted attrition probability >= threshold are flagged as 'Attrition'. Tuned on Out-of-Fold CV.",
)

# Header
st.title("Employee Attrition Prediction Dashboard")
st.markdown(
    "Predict which employees are at risk of leaving before they resign, allowing proactive HR retention actions."
)

tab_single, tab_batch = st.tabs(["Single Employee Prediction", "Batch CSV Prediction"])

# ==========================================================
# TAB 1: SINGLE EMPLOYEE PREDICTION
# ==========================================================
with tab_single:
    st.subheader("Employee Profile Details")

    with st.form("employee_form"):
        col1, col2 = st.columns(2)

        with col1:
            age = st.number_input("Age", min_value=18, max_value=70, value=32, step=1)
            department = st.selectbox(
                "Department",
                ["Sales", "Research & Development", "Human Resources"],
                index=0,
            )
            job_role = st.selectbox(
                "Job Role",
                [
                    "Sales Executive",
                    "Research Scientist",
                    "Laboratory Technician",
                    "Manufacturing Director",
                    "Healthcare Representative",
                    "Manager",
                    "Sales Representative",
                    "Research Director",
                    "Human Resources",
                ],
                index=6,  # Default to Sales Representative
            )
            job_level = st.slider("Job Level (1 to 5)", min_value=1, max_value=5, value=1)
            monthly_income = st.number_input(
                "Monthly Income ($)", min_value=1000.0, max_value=25000.0, value=2800.0, step=100.0
            )
            overtime = st.selectbox("OverTime", ["Yes", "No"], index=0)
            work_life_balance = st.slider(
                "Work-Life Balance (1=Bad, 4=Best)", min_value=1, max_value=4, value=1
            )
            job_satisfaction = st.slider(
                "Job Satisfaction (1=Low, 4=Very High)", min_value=1, max_value=4, value=2
            )
            env_satisfaction = st.slider(
                "Environment Satisfaction (1=Low, 4=Very High)", min_value=1, max_value=4, value=2
            )
            job_involvement = st.slider(
                "Job Involvement (1=Low, 4=Very High)", min_value=1, max_value=4, value=2
            )
            stock_option_level = st.selectbox("Stock Option Level", [0, 1, 2, 3], index=0)
            daily_rate = st.number_input("Daily Rate ($)", min_value=100.0, max_value=1500.0, value=650.0, step=25.0)
            hourly_rate = st.number_input("Hourly Rate ($)", min_value=30.0, max_value=100.0, value=55.0, step=1.0)
            monthly_rate = st.number_input("Monthly Rate ($)", min_value=2000.0, max_value=30000.0, value=13000.0, step=500.0)
            distance_from_home = st.number_input(
                "Distance From Home (miles)", min_value=1.0, max_value=30.0, value=15.0, step=1.0
            )

        with col2:
            total_working_years = st.number_input(
                "Total Working Years", min_value=0.0, max_value=40.0, value=3.0, step=1.0
            )
            years_at_company = st.number_input(
                "Years At Company", min_value=0.0, max_value=40.0, value=1.0, step=1.0
            )
            years_in_role = st.number_input(
                "Years In Current Role", min_value=0.0, max_value=20.0, value=0.0, step=1.0
            )
            years_with_manager = st.number_input(
                "Years With Current Manager", min_value=0.0, max_value=20.0, value=0.0, step=1.0
            )
            years_since_promotion = st.number_input(
                "Years Since Last Promotion", min_value=0.0, max_value=15.0, value=0.0, step=1.0
            )
            business_travel = st.selectbox(
                "Business Travel",
                ["Travel_Frequently", "Travel_Rarely", "Non-Travel"],
                index=0,
            )
            marital_status = st.selectbox(
                "Marital Status",
                ["Single", "Married", "Divorced"],
                index=0,
            )
            education_field = st.selectbox(
                "Education Field",
                ["Marketing", "Life Sciences", "Medical", "Technical Degree", "Other", "Human Resources"],
                index=0,
            )
            education = st.slider(
                "Education (1=Below College, 5=Doctor)", min_value=1, max_value=5, value=2
            )
            gender = st.selectbox("Gender", ["Male", "Female"], index=0)
            num_companies = st.number_input(
                "Number of Companies Worked", min_value=0, max_value=10, value=3, step=1
            )
            percent_hike = st.number_input(
                "Percent Salary Hike (%)", min_value=10.0, max_value=30.0, value=12.0, step=1.0
            )
            perf_rating = st.selectbox("Performance Rating", [3, 4], index=0)
            rel_satisfaction = st.slider(
                "Relationship Satisfaction (1=Low, 4=Very High)", min_value=1, max_value=4, value=2
            )
            training_times = st.number_input(
                "Training Times Last Year", min_value=0, max_value=6, value=1, step=1
            )

        submitted = st.form_submit_button("Predict Attrition Risk", use_container_width=True)

    if submitted:
        employee_payload = {
            "Age": int(age),
            "BusinessTravel": str(business_travel),
            "DailyRate": float(daily_rate),
            "Department": str(department),
            "DistanceFromHome": float(distance_from_home),
            "Education": int(education),
            "EducationField": str(education_field),
            "EnvironmentSatisfaction": int(env_satisfaction),
            "Gender": str(gender),
            "HourlyRate": float(hourly_rate),
            "JobInvolvement": int(job_involvement),
            "JobLevel": int(job_level),
            "JobRole": str(job_role),
            "JobSatisfaction": int(job_satisfaction),
            "MaritalStatus": str(marital_status),
            "MonthlyIncome": float(monthly_income),
            "MonthlyRate": float(monthly_rate),
            "NumCompaniesWorked": int(num_companies),
            "OverTime": str(overtime),
            "PercentSalaryHike": float(percent_hike),
            "PerformanceRating": int(perf_rating),
            "RelationshipSatisfaction": int(rel_satisfaction),
            "StockOptionLevel": int(stock_option_level),
            "TotalWorkingYears": float(total_working_years),
            "TrainingTimesLastYear": int(training_times),
            "WorkLifeBalance": int(work_life_balance),
            "YearsAtCompany": float(years_at_company),
            "YearsInCurrentRole": float(years_in_role),
            "YearsSinceLastPromotion": float(years_since_promotion),
            "YearsWithCurrManager": float(years_with_manager),
        }

        result = None
        with st.spinner("Analyzing attrition risk..."):
            try:
                model = get_model()
                result = predict_single(employee_payload, model=model, threshold=threshold)
            except Exception as ex:
                st.error(f"Prediction failed: {ex}")

        if result:
            st.markdown("---")
            st.subheader("Prediction Results")

            p_col1, p_col2, p_col3 = st.columns(3)
            proba_pct = result["attrition_probability"] * 100
            risk = result["risk_level"]
            is_positive = result["attrition_prediction"] == 1

            with p_col1:
                st.metric(label="Attrition Probability", value=f"{proba_pct:.1f}%")

            with p_col2:
                if is_positive:
                    st.error(f"Predicted Outcome: **{result['attrition_label']}**")
                else:
                    st.success(f"Predicted Outcome: **{result['attrition_label']}**")

            with p_col3:
                if risk == "High":
                    st.markdown(
                        "<p style='font-size:1.1rem;'>Risk Level: <span class='badge-high'>HIGH RISK</span></p>",
                        unsafe_allow_html=True,
                    )
                elif risk == "Medium":
                    st.markdown(
                        "<p style='font-size:1.1rem;'>Risk Level: <span class='badge-med'>MEDIUM RISK</span></p>",
                        unsafe_allow_html=True,
                    )
                else:
                    st.markdown(
                        "<p style='font-size:1.1rem;'>Risk Level: <span class='badge-low'>LOW RISK</span></p>",
                        unsafe_allow_html=True,
                    )

            st.progress(min(max(result["attrition_probability"], 0.0), 1.0))

            # Strategic retention advice
            st.markdown("### Recommended Business Action")
            if is_positive:
                advice = []
                if overtime == "Yes":
                    advice.append("• **Overtime Burnout:** Employee is logging sustained overtime. Recommend immediate workload rebalancing and overtime cap review.")
                if years_at_company <= 1:
                    advice.append("• **First-Year Flight Risk:** Employee is in their critical first year (34.9% cohort turnover). Assign a senior mentor and schedule 30/60/90-day retention check-ins.")
                if stock_option_level == 0:
                    advice.append("• **Equity Considerations:** Employee has no stock options (Level 0 is associated with higher historical cohort turnover, ~24.4% vs 7.6%–9.4% for Levels 1/2). Reviewing equity eligibility is worth exploring.")
                if work_life_balance <= 2:
                    advice.append("• **Work-Life Balance Deficit:** Low work-life balance detected. Explore flexible scheduling or remote work options.")
                if monthly_income < 3500:
                    advice.append("• **Compensation Deficit:** Monthly salary is below peer median ($3,200 vs. $5,200). Conduct compensation market parity review.")
                if not advice:
                    advice.append("• **Proactive Retention Outreach:** Schedule a stay-interview with senior HR leadership within 14 business days.")
                for line in advice:
                    st.warning(line)

            # Top drivers from model explanation
            if "explanation" in result and result["explanation"]:
                st.markdown("### Top Contributing Risk Factors (Odds Ratios)")
                drivers = result["explanation"].get("top_risk_drivers", [])
                if drivers:
                    d_cols = st.columns(len(drivers))
                    for idx, d in enumerate(drivers):
                        with d_cols[idx]:
                            st.info(f"**{d['feature']}**\n\nOdds Ratio: `{d['odds_ratio']:.2f}x`")
            else:
                st.info("• **Healthy Retention Profile:** Employee exhibits strong organizational stability indicators. Continue standard bi-annual career growth reviews.")


# ==========================================================
# TAB 2: BATCH PREDICTION VIA CSV
# ==========================================================
with tab_batch:
    st.subheader("Bulk Employee Scoring")
    st.markdown("Upload a CSV file with employee features to score hundreds or thousands of staff records in bulk.")

    # 10 realistic sample rows
    clean_csv_path = PROJECT_ROOT / "data" / "processed" / "clean.csv"
    if clean_csv_path.exists():
        raw_df = pd.read_csv(clean_csv_path)
        sample_cols = [c for c in raw_df.columns if c not in ["Attrition", "tenure_band", "income_per_working_year", "role_to_company_ratio", "overtime_and_low_wlb", "is_first_year"]]
        sample_df = raw_df[sample_cols].head(10)
    else:
        sample_df = pd.DataFrame()

    csv_sample_buffer = io.StringIO()
    sample_df.to_csv(csv_sample_buffer, index=False)
    st.download_button(
        "Download Sample CSV Template",
        data=csv_sample_buffer.getvalue(),
        file_name="sample_employees.csv",
        mime="text/csv",
    )

    uploaded_file = st.file_uploader("Upload Employee CSV", type=["csv"])

    if uploaded_file is not None:
        try:
            batch_input_df = pd.read_csv(uploaded_file)
            st.write(f"Loaded **{len(batch_input_df)}** records:")
            st.dataframe(batch_input_df.head(), use_container_width=True)

            if st.button("Score Entire Batch", use_container_width=True):
                scored_df = None
                with st.spinner("Scoring records..."):
                    try:
                        model = get_model()
                        scored_df = predict_batch(batch_input_df, model=model, threshold=threshold)
                    except Exception as ex:
                        st.error(f"Batch scoring failed: {ex}")

                if scored_df is not None:
                    total_rows = len(scored_df)
                    flagged_count = int(scored_df["attrition_prediction"].sum())
                    flagged_pct = (flagged_count / total_rows) * 100 if total_rows > 0 else 0.0

                    m1, m2, m3 = st.columns(3)
                    m1.metric("Total Employees Evaluated", total_rows)
                    m2.metric("Flagged At-Risk Employees", flagged_count)
                    m3.metric("Projected Attrition Rate", f"{flagged_pct:.1f}%")

                    st.markdown("### Scored Dataset")
                    st.dataframe(scored_df, use_container_width=True)

                    out_csv = io.StringIO()
                    scored_df.to_csv(out_csv, index=False)
                    st.download_button(
                        "Download Scored Results (CSV)",
                        data=out_csv.getvalue(),
                        file_name=f"scored_{uploaded_file.name}",
                        mime="text/csv",
                        use_container_width=True,
                    )
        except Exception as e:
            st.error(f"Failed to process uploaded file: {e}")
