import google.generativeai as genai

API_KEY = "Enter API KEY"
genai.configure(api_key=API_KEY)

MODEL = "gemini-2.5-flash"

def get_summary(text: str, max_tokens: int = 200) -> str:
    """
    Generates a brief summary of the provided text using Gemini 2.5 Flash.
    """
    prompt = (
        "Please read the following text carefully and generate a clear, concise summary of about 5 to 6 sentences. "
        "Focus on the main points, key events, outcomes, and important details without unnecessary repetition or overly technical jargon.\n\n"
        "Make the summary easily understandable for a general audience, maintaining accuracy and coherence. "
        "If the text contains any sentiment or notable topics, highlight them briefly. "
        "Provide the summary in a well-structured paragraph form.\n\n"
        f"{text}"
    )
    try:
        response = genai.GenerativeModel(MODEL).generate_content(prompt)
        return response.text.strip()
    except Exception as e:
        return f"[Error generating summary: {e}]"
