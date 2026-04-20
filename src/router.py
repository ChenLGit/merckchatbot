from groq import Groq
try:
    # We import the dictionary here. 
    # If there's an issue with prompts.py, we catch it immediately.
    from .prompts import ROUTING_PROMPTS
except ImportError:
    ROUTING_PROMPTS = {}

def get_intent(user_input, api_key):
    """
    Determines user intent via Llama 3.3 on Groq.
    Categorizes into OPPORTUNITY, MARKETING, NEWS, or GENERAL.
    """
    try:
        client = Groq(api_key=api_key)

        system_instructions = ROUTING_PROMPTS.get(
            "intent_classifier",
            "Classify user input into: OPPORTUNITY, MARKETING, NEWS, or GENERAL. Return only the word.",
        )

        response = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[
                {"role": "system", "content": system_instructions},
                {"role": "user", "content": user_input},
            ],
            temperature=0,
            max_tokens=20,
        )

        raw_content = response.choices[0].message.content.strip().upper()

        # Check GENERAL last so narrower intents win when the model emits both.
        for valid in ["OPPORTUNITY", "MARKETING", "NEWS", "GENERAL"]:
            if valid in raw_content:
                return valid

        return "GENERAL"

    except Exception as e:
        print(f"Routing Error: {e}")
        return "GENERAL"