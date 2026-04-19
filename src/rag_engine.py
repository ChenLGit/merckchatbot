import pandas as pd
import re

def get_hcp_scorecard(query, df):
    # 1. Stricter State Detection using Word Boundaries
    state_list = df['Rndrng_Prvdr_State_Abrvtn'].unique().tolist()
    target_state = None
    
    # query.upper() ensures we match "nj", "NJ", or "New Jersey" if the abrv is present
    for state in state_list:
        # \b ensures "IN" matches the state, but not the word "insight" or "in"
        if re.search(rf'\b{state}\b', query.upper()):
            target_state = state
            break
            
    # 2. Filtering
    if target_state:
        targets = df[(df['pred_class'] == 1) & (df['Rndrng_Prvdr_State_Abrvtn'] == target_state)]
    else:
        targets = df[df['pred_class'] == 1]
    
    # Sort by Opportunity Size (Average Medicare Payment)
    targets = targets.sort_values(by='OP_MDCR_PYMT_PC', ascending=False)
    
    if targets.empty:
        return None

    top_hcp = targets.iloc[0]
    
    # 3. Clean SHAP Drivers
    shap_cols = [c for c in df.columns if c.startswith('SHAP_')]
    shap_values = {}
    for col in shap_cols:
        try:
            shap_values[col] = float(str(top_hcp[col]).replace("'", ""))
        except:
            shap_values[col] = 0.0

    sorted_drivers = sorted(shap_values.items(), key=lambda x: abs(x[1]), reverse=True)[:5]
    driver_list = [f"{k.replace('SHAP_', '').replace('_', ' ').title()} ({v:+.2f})" for k, v in sorted_drivers]
    
    return {
        "npi": top_hcp.get('Rndrng_NPI', 'N/A'),
        "state": top_hcp.get('Rndrng_Prvdr_State_Abrvtn', 'N/A'),
        "city": top_hcp.get('Rndrng_Prvdr_City', 'N/A'), # Added for Location display
        "type": top_hcp.get('Cleaned_Prvdr_Type', 'N/A'),
        "payment": top_hcp.get('OP_MDCR_PYMT_PC', 0),
        "drivers": ", ".join(driver_list),
        "score": top_hcp.get('pred_proba', 0)
    }