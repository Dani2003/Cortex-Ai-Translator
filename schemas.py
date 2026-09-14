from pydantic import BaseModel, Field
from typing import List, Optional

class TranslationRequest(BaseModel):
    text: str = Field(..., description="Source text to translate")
    target_languages: List[str] = Field(..., example=["Urdu", "Spanish", "French"])
    methodology: str = Field(default="Neural Machine Translation with Human In The Loop", description="Translation strategy")
    tone: str = Field(default="Professional / Technical", description="Target tone of voice")
    review_cycles: int = Field(default=1, description="Number of audit/refinement passes")

class SingleLanguageOutput(BaseModel):
    target_language: str
    model_a_translation: str
    model_b_translation: str
    final_merged_translation: str
    confidence_score: int
    discrepancy_notes: str

class AuditReportResponse(BaseModel):
    job_id: str
    original_text: str
    estimated_cost_usd: float
    methodology_applied: str
    tone_applied: str
    translations: List[SingleLanguageOutput]