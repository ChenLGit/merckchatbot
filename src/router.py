from groq import Groq
# Import the specific dictionary from our centralized prompts file
from .prompts import ROUTING_PROMPTS

def get_intent(user_input, api_key):
    """
    Determines user intent via Llama 3 on Groq using centralized prompts.
    """
    try:
        client = Groq(api_key=api_key)
        
        # We now pull the intent classifier prompt from ROUTING_PROMPTS
        response = client.chat.completions.create(
            model="llama3-70b-8192", 
            messages=[
                {"role": "system", "content": ROUTING_PROMPTS["intent_classifier"]},
                {"role": "user", "content": user_input}
            ],
            temperature=0,
            max_tokens=20 
        )
        
        raw_content = response.choices[0].message.content.strip().upper()
        
        # Validation keywords
        valid_intents = ["OPPORTUNITY", "MARKETING", "NEWS"]
        
        for valid in valid_intents:
            if valid in raw_content:
                return valid
        
        return "OPPORTUNITY"
            
    except Exception as e:
        # Fallback to OPPORTUNITY ensures the app doesn't crash on API lag
        return "OPPORTUNITY"