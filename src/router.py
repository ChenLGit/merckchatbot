from groq import Groq
from .prompts import ROUTING_PROMPTS

def get_intent(user_input, api_key):
    """
    Determines user intent via Llama 3.3 on Groq.
    Categorizes into OPPORTUNITY, MARKETING, or NEWS.
    """
    try:
        client = Groq(api_key=api_key)
        
        # Migrated to llama-3.3-70b-versatile for 2026 compatibility
        response = client.chat.completions.create(
            model="llama-3.3-70b-versatile", 
            messages=[
                {"role": "system", "content": ROUTING_PROMPTS["intent_classifier"]},
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
        
        # Default fallback to OPPORTUNITY (Structured RAG)
        return "OPPORTUNITY"
            
    except Exception as e:
        # Log error to console for debugging
        print(f"Routing Error: {e}")
        return "OPPORTUNITY"