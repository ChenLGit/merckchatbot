from groq import Groq
try:
    # We import the dictionary here. 
    # If there's an issue with prompts.py, we catch it immediately.
    from .prompts import ROUTING_PROMPTS
except ImportError:
    ROUTING_PROMPTS = {}

INTENT_PRIORITY = ["GENERAL", "OPPORTUNITY", "MARKETING", "NEWS"]


def get_intents(user_input, api_key):
    """
    Return an ordered list of distinct intents present in the user query,
    sorted into a fixed display priority: GENERAL, OPPORTUNITY, MARKETING, NEWS.

    Falls back to ["GENERAL"] on any error. Always non-empty.
    """
    try:
        client = Groq(api_key=api_key)
        system_instructions = ROUTING_PROMPTS.get(
            "multi_intent_classifier",
            "List all intents in the query from {OPPORTUNITY, MARKETING, NEWS, GENERAL}, comma-separated.",
        )
        response = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[
                {"role": "system", "content": system_instructions},
                {"role": "user", "content": user_input},
            ],
            temperature=0,
            max_tokens=32,
        )
        raw = response.choices[0].message.content.upper()

        found_order = [intent for intent in INTENT_PRIORITY if intent in raw]
        if not found_order:
            return ["GENERAL"]

        # Drop GENERAL if a specific intent is also present, so "top NJ
        # opportunity" doesn't also trigger a GENERAL brand-advisor answer.
        if len(found_order) > 1 and "GENERAL" in found_order:
            found_order = [i for i in found_order if i != "GENERAL"]
        return found_order

    except Exception as e:
        print(f"Routing Error (get_intents): {e}")
        return ["GENERAL"]


def get_intent(user_input, api_key):
    """
    Back-compat: single-intent classifier. Uses the multi-intent classifier
    under the hood and returns the highest-priority intent found.
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