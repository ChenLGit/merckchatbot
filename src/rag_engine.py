import pandas as pd
import re

def get_hcp_scorecard(query, df):
    """
    Robust Structured RAG: Filters by state mentioned in query and
    extracts cleaned SHAP drivers for AI interpretation.
    """
    # 1. Advanced State Detection
    state_list = df['Rndrng_Prvdr_State_Abrvtn'].unique().tolist()
    target_state = None
    
    # Searches for state abbreviation as a standalone word (e.g., "in CA")
    for state in state_list:
        if re.search(rf'\b{state}\b', query.upper()):
            target_state = state
            break
            
    # 2. Filtering Logic
    if target_state:
        targets = df[(df['pred_class'] == 1) & (df['Rndrng_Prvdr_State_Abrvtn'] == target_state)]
    else:
        targets = df[df['pred_class'] == 1]
    
    # Sort by Opportunity Size (Medicare Payment)
    targets = targets.sort_values(by='OP_MDCR_PYMT_PC', ascending=False)
    
    if targets.empty:
        return None

    top_hcp = targets.iloc[0]
    
    # 3. Clean SHAP Drivers for LLM Readability
    shap_cols = [c for c in df.columns if c.startswith('SHAP_')]
    shap_values = top_hcp[shap_cols].to_dict()
    
    # Get top 3 drivers by absolute impact
    sorted_drivers = sorted(shap_values.items(), key=lambda x: abs(x[1]), reverse=True)[:3]
    
    # Format drivers: "SHAP_Tot_Clms" -> "Tot Clms"
    driver_list = []
    for k, v in sorted_drivers:
        clean_name = k.replace('SHAP_', '').replace('_', ' ')
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