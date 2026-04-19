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
    Categorizes into OPPORTUNITY, MARKETING, or NEWS.
    """
    try:
        client = Groq(api_key=api_key)
        
        # Safe access to the dictionary using .get() 
        # This prevents the app from crashing if the key is missing.
        system_instructions = ROUTING_PROMPTS.get(
            "intent_classifier", 
            "Classify user input into: OPPORTUNITY, MARKETING, or NEWS. Return only the word."
        )
        
        response = client.chat.completions.create(
            model="llama-3.3-70b-versatile", 
            messages=[
                {"role": "system", "content": system_instructions},
                {"role": "user", "content": user_input}
            ],
            temperature=0,
            max_tokens=20 
        )
        
        # Clean the output
        raw_content = response.choices[0].message.content.strip().upper()
        
        # Robust check for valid intent keywords
        valid_intents = ["OPPORTUNITY", "MARKETING", "NEWS"]
        
        for valid in valid_intents:
            if valid in raw_content:
                return valid
        
        return "OPPORTUNITY"
            
    except Exception as e:
        print(f"Routing Error: {e}")
        return "OPPORTUNITY"