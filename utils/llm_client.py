import os

from google.generativeai import client as gemini_client


gemini_client.configure(api_key=os.getenv("GEMINI_API_KEY", ""))


def summarize_text(text: str) -> str:
    if not text:
        return ""
    response = gemini_client.responses.create(
        model="gemini-1.0",
        input=text,
        max_output_tokens=250,
    )
    return response.output_text
