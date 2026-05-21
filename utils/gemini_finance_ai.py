import google.generativeai as genai

# ---------------------------------------------------
# GEMINI API CONFIG
# ---------------------------------------------------

API_KEY = "PASTE_NEW_API_KEY_HERE"

genai.configure(
    api_key=API_KEY
)

# ---------------------------------------------------
# LOAD MODEL
# ---------------------------------------------------

model = genai.GenerativeModel(
    "gemini-1.5-flash"
)

# ---------------------------------------------------
# FINANCE AI CHATBOT
# ---------------------------------------------------

def finance_ai_chat(user_message):

    finance_prompt = f"""

    You are an advanced AI Financial Assistant.

    IMPORTANT RULES:

    1. ONLY answer finance and investment related questions.

    2. Allowed Topics:
       - Mutual Funds
       - SIP
       - Portfolio Analysis
       - Risk Analysis
       - Wealth Management
       - Finance
       - Investment Strategies
       - Tax Saving
       - Market Trends
       - Equity and Debt Funds

    3. If the question is unrelated to finance,
       politely refuse.

    4. Give professional and practical insights.

    User Question:
    {user_message}

    """

    response = model.generate_content(
        finance_prompt
    )

    return response.text