import os
import uuid
import asyncio
import json
from groq import AsyncGroq
from schemas import TranslationRequest, AuditReportResponse, SingleLanguageOutput

client = AsyncGroq(api_key=os.getenv("GROQ_API_KEY"))

# Preferred order of production models
PREFERRED_MODELS = [
    "llama-3.3-70b-versatile",
    "llama-3.1-8b-instant",
    "openai/gpt-oss-120b",
    "qwen/qwen3.6-27b"
]

async def get_active_model() -> str:
    """Fetch live available models for the current API key to avoid 404/decommissioned errors."""
    try:
        model_list = await client.models.list()
        available_ids = [m.id for m in model_list.data]
        
        for pref in PREFERRED_MODELS:
            if pref in available_ids:
                return pref
        return available_ids[0] if available_ids else "llama-3.1-8b-instant"
    except Exception as e:
        # Fallback if listing fails
        return "llama-3.1-8b-instant"

def calculate_price_estimate(text: str, target_langs_count: int, cycles: int) -> float:
    char_count = len(text)
    base_rate = 0.0001
    total = char_count * base_rate * target_langs_count * (1 + (cycles * 0.2))
    return round(total, 4)

async def translate_with_model(model_name: str, text: str, lang: str, methodology: str, tone: str, variant_rule: str) -> str:
    prompt = f"""You are a professional multilingual translator.
Translate the following text into {lang}.
Methodology: {methodology}.
Tone: {tone}.
Execution Guideline: {variant_rule}

Text:
{text}"""

    response = await client.chat.completions.create(
        model=model_name,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.3
    )
    return response.choices[0].message.content.strip()

async def evaluate_and_merge(model_name: str, text: str, lang: str, trans_a: str, trans_b: str) -> dict:
    eval_prompt = f"""You are a translation quality assurance engine.
Original Text: {text}
Target Language: {lang}

Translation Candidate A: {trans_a}
Translation Candidate B: {trans_b}

Compare both translations.
1. Merge them into the single best, most accurate final translation.
2. Provide a Confidence Score from 0 to 100 based on accuracy and agreement.
3. Provide brief discrepancy notes explaining key differences.

Respond STRICTLY in JSON format:
{{
    "final_merged_translation": "...",
    "confidence_score": 95,
    "discrepancy_notes": "..."
}}"""

    response = await client.chat.completions.create(
        model=model_name,
        response_format={"type": "json_object"},
        messages=[{"role": "user", "content": eval_prompt}],
        temperature=0.1
    )
    return json.loads(response.choices[0].message.content)

async def process_translation_job(req: TranslationRequest) -> AuditReportResponse:
    active_model = await get_active_model()
    estimated_cost = calculate_price_estimate(req.text, len(req.target_languages), req.review_cycles)
    results = []

    for lang in req.target_languages:
        trans_a, trans_b = await asyncio.gather(
            translate_with_model(active_model, req.text, lang, req.methodology, req.tone, "Focus strictly on exact structural fidelity"),
            translate_with_model(active_model, req.text, lang, req.methodology, req.tone, "Focus on natural target language fluency")
        )

        eval_data = await evaluate_and_merge(active_model, req.text, lang, trans_a, trans_b)

        results.append(SingleLanguageOutput(
            target_language=lang,
            model_a_translation=trans_a,
            model_b_translation=trans_b,
            final_merged_translation=eval_data["final_merged_translation"],
            confidence_score=eval_data["confidence_score"],
            discrepancy_notes=eval_data["discrepancy_notes"]
        ))

    return AuditReportResponse(
        job_id=str(uuid.uuid4())[:8],
        original_text=req.text,
        estimated_cost_usd=estimated_cost,
        methodology_applied=req.methodology,
        tone_applied=req.tone,
        translations=results
    )