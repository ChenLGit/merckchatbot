# In src/router.py
from groq import Groq
from prompts import SYSTEM_PROMPTS # Removed 'src.' or '.'

def get_intent(user_input, api_key):
    """
    Determines the user's intent using Groq's fast inference.
    Passed api_key should come from st.secrets in the main app.
    """
    try:
        client = Groq(api_key=api_key)
        
        response = client.chat.completions.create(
            model="llama3-70b-8192",  # Llama 3 is excellent for classification
            messages=[
                {"role": "system", "content": SYSTEM_PROMPTS["router"]},
                {"role": "user", "content": user_input}
            ],
            temperature=0,
            max_tokens=10  # We only need a one-word answer
        )
        
        intent = response.choices[0].message.content.strip().upper()
        
        # Validation to ensure the router doesn't return garbage
        valid_intents = ["OPPORTUNITY", "MARKETING", "NEWS"]
        if intent not in valid_intents:
            # Default to Opportunity if the LLM gets creative
            return "OPPORTUNITY"
            
        return intent
    except Exception as e:
        print(f"Routing Error: {e}")
        return "OPPORTUNITY"