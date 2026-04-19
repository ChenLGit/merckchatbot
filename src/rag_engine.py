import pandas as pd
import re

def get_hcp_scorecard(query, df):
    """
    RAG Engine: Detects state, filters high-propensity targets, 
    and cleans SHAP drivers for AI interpretation.
    """
    # 1. Robust State Detection
    state_list = df['Rndrng_Prvdr_State_Abrvtn'].unique().tolist()
    target_state = None
    for state in state_list:
        if re.search(rf'\b{state}\b', query.upper()):
            target_state = state
            break
            
    # 2. Filtering
    if target_state:
        targets = df[(df['pred_class'] == 1) & (df['Rndrng_Prvdr_State_Abrvtn'] == target_state)]
    else:
        targets = df[df['pred_class'] == 1]
    
    targets = targets.sort_values(by='OP_MDCR_PYMT_PC', ascending=False)
    
    if targets.empty:
        return None

    top_hcp = targets.iloc[0]
    
    # 3. Clean SHAP Drivers (Top 5 only for clarity)
    shap_cols = [c for c in df.columns if c.startswith('SHAP_')]
    
    # Ensure values are numeric before sorting (safety catch)
    shap_values = {}
    for col in shap_cols:
        try:
            val = float(str(top_hcp[col]).replace("'", ""))
            shap_values[col] = val
        except:
            shap_values[col] = 0.0

    # Sort and take top 5
    sorted_drivers = sorted(shap_values.items(), key=lambda x: abs(x[1]), reverse=True)[:5]
    
    # Human-readable formatting: SHAP_Tot_Benes -> Total Beneficiaries
    driver_list = []
    for k, v in sorted_drivers:
        clean_name = k.replace('SHAP_', '').replace('_', ' ').title()
        driver_list.append(f"{clean_name} ({v:+.2f})")
    
    driver_summary = ", ".join(driver_list)
    
    return {
        "npi": top_hcp.get('Rndrng_NPI', 'N/A'),
        "state": top_hcp.get('Rndrng_Prvdr_State_Abrvtn', 'N/A'),
        "type": top_hcp.get('Cleaned_Prvdr_Type', 'N/A'),
        "payment": top_hcp.get('OP_MDCR_PYMT_PC', 0),
        "drivers": driver_summary,
        "score": top_hcp.get('pred_proba', 0)
    }