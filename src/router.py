import re

from groq import Groq
try:
    # We import the dictionary here. 
    # If there's an issue with prompts.py, we catch it immediately.
    from .prompts import ROUTING_PROMPTS
except ImportError:
    ROUTING_PROMPTS = {}

INTENT_PRIORITY = ["GENERAL", "OPPORTUNITY", "MARKETING", "NEWS"]

# Cues that indicate the user is explicitly asking a GENERAL / definitional
# question, even if they ALSO ask about a specific intent in the same sentence.
# When any of these appear, we keep GENERAL in the intent list instead of
# treating it as a catch-all.
_GENERAL_CUE_PATTERNS = [
    r"\blist\b",
    r"\bwho are\b",
    r"\bwhat are\b",
    r"\bwhat is\b",
    r"\bname (the|all|keytruda's|merck's)\b",
    r"\bdefine\b",
    r"\bexplain\b",
    r"\bhow does\b",
    r"\bhow do\b",
    r"\btell me about\b",
    r"\bmain competitor",
    r"\bmain competitors\b",
    r"\bdirect competitor",
    r"\bcompete with\b",
]


def _has_general_cue(text):
    """True if the user explicitly asks a definitional/general question."""
    if not text:
        return False
    t = str(text).lower()
    return any(re.search(p, t) for p in _GENERAL_CUE_PATTERNS)


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
            model="llama-3.1-8b-instant",
            messages=[
                {"role": "system", "content": system_instructions},
                {"role": "user", "content": user_input},
            ],
            temperature=0,
            max_tokens=32,
        )
        raw = response.choices[0].message.content.upper()

        found_order = [intent for intent in INTENT_PRIORITY if intent in raw]

        # Guarantee GENERAL when the user explicitly asked a definitional
        # question, even if the classifier didn't return it.
        if _has_general_cue(user_input) and "GENERAL" not in found_order:
            found_order = ["GENERAL"] + found_order

        if not found_order:
            return ["GENERAL"]

        # Only drop GENERAL when it looks like a catch-all. If the user
        # query carries an explicit definitional cue ("list", "who are",
        # "what are", etc.) we keep it alongside the specific intents so
        # both get answered.
        if (
            len(found_order) > 1
            and "GENERAL" in found_order
            and not _has_general_cue(user_input)
        ):
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
            model="llama-3.1-8b-instant",
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