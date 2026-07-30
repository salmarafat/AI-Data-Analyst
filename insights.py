"""
insights.py
Generates "AI Business Insights" and "AI Recommendations" using the local model,
based on the statistical profile of the data (not the raw data) to save tokens
and preserve privacy.
"""

from modules.llm_client import ask_ollama

INSIGHTS_SYSTEM_PROMPT = """
You are an expert Business Data Analyst. You will receive a statistical summary of a company's dataset.
Your task: derive useful, realistic Business Insights based only on the given numbers.
- Write in clear, simple English (understandable to a business owner who is not a statistics expert).
- Organize insights into clear, direct bullet points.
- Focus on: trends, concentrations, anomalies, opportunities, and potential risks.
- Do not invent numbers that are not present in the given summary.
- Do not just repeat the numbers as-is; interpret them and explain their business meaning.
"""

RECOMMENDATIONS_SYSTEM_PROMPT = """
You are an expert Business Consultant. You will receive a statistical summary and analytical insights
about a company's data.
Your task: propose practical, Actionable Recommendations to improve performance.
- Write in clear, simple English.
- Organize recommendations as a numbered list, each recommendation one or two sentences max.
- Each recommendation must be grounded in evidence from the given data.
- Order recommendations by priority (most important first).
"""


def generate_business_insights(profile_text: str, model: str = "llama3.1") -> str:
    prompt = f"Here is the statistical summary of the data:\n\n{profile_text}\n\nExtract the most important Business Insights from this data."
    return ask_ollama(prompt, model=model, system=INSIGHTS_SYSTEM_PROMPT, temperature=0.4)


def generate_recommendations(profile_text: str, insights_text: str, model: str = "llama3.1") -> str:
    prompt = (
        f"Statistical summary of the data:\n{profile_text}\n\n"
        f"Analytical insights derived earlier:\n{insights_text}\n\n"
        "Based on the above, suggest practical recommendations to improve business performance."
    )
    return ask_ollama(prompt, model=model, system=RECOMMENDATIONS_SYSTEM_PROMPT, temperature=0.4)
