from fastapi import FastAPI, HTTPException
from schemas import TranslationRequest, AuditReportResponse
from services import process_translation_job

from dotenv import load_dotenv
import os

load_dotenv()
api_key = os.getenv("GROQ_API_KEY")

app = FastAPI(
    title="Cortex AI Multilingual Translation & Review API",
    description="Dual-pipeline AI translation platform with consensus evaluation and audit reporting.",
    version="1.0.0"
)

@app.get("/")
def read_root():
    return {"message": "Cortex AI Translator API is running online!"}

@app.post("/api/v1/translate", response_model=AuditReportResponse)
async def translate_endpoint(request: TranslationRequest):
    try:
        return await process_translation_job(request)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))