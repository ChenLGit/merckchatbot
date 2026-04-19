import pandas as pd
import re

def get_hcp_scorecard(query, df):
    # 1. Robust State Detection
    # Get unique list of states from your data
    state_list = df['Rndrng_Prvdr_State_Abrvtn'].unique().tolist()
    target_state = None
    
    # Check for the state abbreviation as a standalone word (e.g., "in CA" or "CA opportunities")
    for state in state_list:
        if re.search(rf'\b{state}\b', query.upper()):
            target_state = state
            break
            
    # 2. Filter the Data
    if target_state:
        targets = df[(df['pred_class'] == 1) & (df['Rndrng_Prvdr_State_Abrvtn'] == target_state)]
    else:
        targets = df[df['pred_class'] == 1]
    
    # Sort by payment volume (Opportunity size)
    targets = targets.sort_values(by='OP_MDCR_PYMT_PC', ascending=False)
    
    if targets.empty:
        return None

    top_hcp = targets.iloc[0]
    
    # 3. SHAP Drivers
    shap_cols = [c for c in df.columns if c.startswith('SHAP_')]
    shap_values = top_hcp[shap_cols].to_dict()
    sorted_drivers = sorted(shap_values.items(), key=lambda x: abs(x[1]), reverse=True)[:3]
    
    # Clean up names for the AI
    driver_summary = ", ".join([f"{k.replace('SHAP_', '')}" for k, v in sorted_drivers])
    
    return {
        "npi": top_hcp.get('Rndrng_NPI', 'N/A'),
        "state": top_hcp.get('Rndrng_Prvdr_State_Abrvtn', 'N/A'),
        "type": top_hcp.get('Cleaned_Prvdr_Type', 'N/A'),
        "payment": top_hcp.get('OP_MDCR_PYMT_PC', 0),
        "drivers": driver_summary,
        "score": top_hcp.get('pred_proba', 0)
    }