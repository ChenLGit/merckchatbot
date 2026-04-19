import pandas as pd

def get_hcp_scorecard(query, df):
    # 1. State Detection Logic
    # We check if a state abbreviation (e.g., 'CA') is in the user query
    state_list = df['Rndrng_Prvdr_State_Abrvtn'].unique()
    target_state = None
    
    for state in state_list:
        if f" {state} " in f" {query.upper()} ":
            target_state = state
            break
            
    # 2. Filter the Data
    if target_state:
        # Filter for the specific state AND high propensity
        targets = df[(df['pred_class'] == 1) & (df['Rndrng_Prvdr_State_Abrvtn'] == target_state)]
    else:
        # Default to national top targets if no state mentioned
        targets = df[df['pred_class'] == 1]
    
    # Sort by payment volume
    targets = targets.sort_values(by='OP_MDCR_PYMT_PC', ascending=False)
    
    if targets.empty:
        return None

    top_hcp = targets.iloc[0]
    
    # 3. Extract SHAP drivers
    shap_cols = [c for c in df.columns if c.startswith('SHAP_')]
    shap_values = top_hcp[shap_cols].to_dict()
    sorted_drivers = sorted(shap_values.items(), key=lambda x: abs(x[1]), reverse=True)[:3]
    
    # Clean the names (remove SHAP_ prefix) for the LLM
    driver_summary = ", ".join([f"{k.replace('SHAP_', '')}" for k, v in sorted_drivers])
    
    return {
        "npi": top_hcp.get('Rndrng_NPI', 'N/A'),
        "state": top_hcp.get('Rndrng_Prvdr_State_Abrvtn', 'N/A'),
        "type": top_hcp.get('Cleaned_Prvdr_Type', 'N/A'),
        "payment": top_hcp.get('OP_MDCR_PYMT_PC', 0),
        "drivers": driver_summary,
        "score": top_hcp.get('pred_proba', 0)
    }