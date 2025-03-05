from fastapi import FastAPI, HTTPException
import requests
import google.generativeai as genai
from datetime import datetime, timedelta
import os
import json
import logging

# Configure detailed logging
logging.basicConfig(level=logging.DEBUG, format="%(asctime)s - %(levelname)s - %(message)s")

# Initialize FastAPI app
app = FastAPI()

# Configure Gemini 2.0 API key (replace with your actual key)
os.environ["GOOGLE_API_KEY"] = "AIzaSyDKsgLeX1oVJsReSuWuLho9_LNmx49Q6Q0"
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

# Endpoints for performance data
BASE_UNITS_URL = "https://platform.suburbanfiberco.com/api/v2/staff-performance/units/overview"
BASE_OVERALL_URL = "https://platform.suburbanfiberco.com/api/v2/staff-performance/overview"

def get_date_range(report_type: str):
    """Return start and end dates based on the report type."""
    today = datetime.today()
    if report_type == "daily":
        start_date = today.strftime("%Y-%m-%d")
        end_date = today.strftime("%Y-%m-%d")
    elif report_type == "weekly":
        # Start from Monday of current week
        start_date = (today - timedelta(days=today.weekday())).strftime("%Y-%m-%d")
        end_date = today.strftime("%Y-%m-%d")
    elif report_type == "monthly":
        # Start from the first day of the month
        start_date = today.replace(day=1).strftime("%Y-%m-%d")
        end_date = today.strftime("%Y-%m-%d")
    else:
        logging.error("Invalid report type provided: %s", report_type)
        raise ValueError("Invalid report type. Choose from 'daily', 'weekly', or 'monthly'.")
    
    logging.debug("Report type '%s' date range: %s to %s", report_type, start_date, end_date)
    return start_date, end_date

def fetch_report(report_type: str):
    """
    Fetch performance data from two endpoints:
      - Unit-level performance data.
      - Overall performance summary.
    Combines both into a single dictionary.
    """
    start_date, end_date = get_date_range(report_type)
    units_url = f"{BASE_UNITS_URL}?start_date={start_date}&&end_date={end_date}"
    overall_url = f"{BASE_OVERALL_URL}?start_date={start_date}&&end_date={end_date}"
    
    logging.debug("Fetching unit-level data from URL: %s", units_url)
    logging.debug("Fetching overall data from URL: %s", overall_url)
    
    try:
        units_response = requests.get(units_url)
        units_response.raise_for_status()
        units_data = units_response.json()
        logging.debug("Fetched units data: %s", units_data)
    except requests.exceptions.RequestException as e:
        logging.exception("Error fetching units performance data")
        raise HTTPException(status_code=500, detail=f"Error fetching units data: {str(e)}")
    
    try:
        overall_response = requests.get(overall_url)
        overall_response.raise_for_status()
        overall_data = overall_response.json()
        logging.debug("Fetched overall data: %s", overall_data)
    except requests.exceptions.RequestException as e:
        logging.exception("Error fetching overall performance data")
        raise HTTPException(status_code=500, detail=f"Error fetching overall data: {str(e)}")
    
    # Combine data from both endpoints
    combined_data = {
        "unitsData": units_data,
        "overallData": overall_data
    }
    return combined_data

def clean_ai_response(response_text: str) -> str:
    """
    Remove markdown formatting (e.g. triple backticks) and extraneous newline characters,
    and remove any leading 'json' text if present.
    """
    logging.debug("Cleaning AI response")
    cleaned = response_text.strip()
    # Remove markdown formatting if present
    if cleaned.startswith("```"):
        cleaned = cleaned.lstrip("```").rstrip("```")
    # Remove any leading 'json' text if present
    if cleaned.lower().startswith("json"):
        cleaned = cleaned[len("json"):].strip()
    # Remove extraneous newline characters
    cleaned = cleaned.replace("\n", " ").strip()
    logging.debug("Cleaned AI response (first 200 chars): %s", cleaned[:200])
    return cleaned

def generate_ai_report(data: dict) -> dict:
    """
    Sends the combined performance data to Gemini 2.0 and returns a JSON report.
    The prompt instructs the AI to produce a human-readable, highly explainable report that
    identifies which departments/units are performing well, how they perform, and explains the
    two derived visualization metrics:
      - IPI (Integrated Performance Index): (obligation * 0.3) + (efficiency * 0.4) + (compliance * 0.3)
      - PVI (Performance Variability Index): standard deviation of obligation, efficiency, and compliance.
    """
    prompt = (
    
        "You are an AI that generates an Automated Supervisory Report for company supervisors. "
        "This report is designed as a first level of oversight and must be highly explainable and human-readable. Also include what IPI and PVI ins measured out of "
        "You are provided with two sets of performance data:\n"
        "  1. 'unitsData': Contains performance data for each department or unit with custom metrics such as obligation, efficiency, compliance, and overall score.\n"
        "  2. 'overallData': Provides an overall performance summary for the organization.\n\n"
        "Your report should address the following:\n"
        "  - Identify which departments or units are performing well and explain how they are performing based on the data.\n"
        "  - Provide insights into areas that may need further attention.\n"
        "  - Explain the two derived visualization metrics: Explain what they are so executives know what they are and wont be confised seeing new numbers\n"
        
        "      * Integrated Performance Index (IPI): Calculated as (obligation * 0.3) + (efficiency * 0.4) + (compliance * 0.3), representing the overall performance of a unit.\n"
        "      * Performance Variability Index (PVI): The standard deviation of obligation, efficiency, and compliance, indicating how balanced the performance is (a lower value means more consistency).\n\n"
        "Based on the input data, produce a clear, concise, and highly explainable report in valid JSON format with no markdown formatting. "
        "The output JSON must include the following keys:\n"
        "- reportTitle (string),\n"
        "- reportDate (YYYY-MM-DD),\n"
        "- overallSummary (string): an explanation of overall performance trends and key findings,\n"
        "- keyInsights (an array of objects): each object should include:\n"
        "    • unit (string): the name of the department or unit,\n"
        "    • insight (string): a brief explanation of the unit's performance,\n"
        "    • metrics (object): containing the original custom metrics ('score', 'obligation', 'efficiency', 'compliance'),\n"
        "    • visualMetrics (object): containing the derived metrics 'IPI' and 'PVI', with an explanation of what they represent.\n"
        "- recommendations (an array of strings): actionable advice based on the analysis.\n\n"
        "Ignore any extraneous metrics in the input data and focus on providing a meaningful interpretation. "
        "Here is the data: " + json.dumps(data)
    )
    logging.debug("Generated prompt for AI (first 500 chars): %s", prompt[:500])
    
    try:
        # Call Gemini 2.0 using the GenerativeModel from google.generativeai
        model = genai.GenerativeModel("gemini-2.0-flash")
        response = model.generate_content(prompt)
        if not response or not response.text:
            logging.error("Empty response received from Gemini API")
            raise HTTPException(status_code=500, detail="Empty response from Gemini API")
        cleaned_text = clean_ai_response(response.text)
        try:
            ai_report = json.loads(cleaned_text)
            logging.debug("AI report generated successfully: %s", ai_report)
        except json.JSONDecodeError:
            logging.exception("Failed to decode AI response as valid JSON")
            raise HTTPException(status_code=500, detail="AI response is not valid JSON")
        return ai_report
    except Exception as e:
        logging.exception("Error generating AI report")
        raise HTTPException(status_code=500, detail=f"Error generating AI report: {str(e)}")

@app.get("/performance/{report_type}")
def performance_report(report_type: str):
    """
    Endpoint to generate a performance report.
    Acceptable values for report_type: daily, weekly, monthly.
    """
    logging.info("Received performance report request for type: %s", report_type)
    if report_type not in ["daily", "weekly", "monthly"]:
        logging.error("Invalid report type requested: %s", report_type)
        raise HTTPException(status_code=400, detail="Invalid report type; must be 'daily', 'weekly', or 'monthly'.")
    try:
        # Fetch combined performance data from both endpoints
        data = fetch_report(report_type)
        # Generate the AI report based on the fetched data
        ai_report = generate_ai_report(data)
        logging.info("Successfully generated AI report for %s report", report_type)
        return {
            "status": "success",
            "report_type": report_type,
            "report": ai_report
        }
    except HTTPException as he:
        logging.error("HTTPException occurred: %s", he.detail)
        raise he
    except Exception as e:
        logging.exception("Unexpected error in performance_report endpoint")
        raise HTTPException(status_code=500, detail=f"Unexpected error: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True)
