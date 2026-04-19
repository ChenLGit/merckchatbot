import pandas as pd

def get_hcp_scorecard(query, df):
    # Filter for high-propensity targets
    targets = df[df['pred_class'] == 1].sort_values(by='OP_MDCR_PYMT_PC', ascending=False)
    
    if targets.empty:
        return None

    # Logic: Pick the top target for the demo
    top_hcp = targets.iloc[0]
    
    # Extract SHAP drivers (explainability)
    shap_cols = [c for c in df.columns if c.startswith('SHAP_')]
    shap_values = top_hcp[shap_cols].to_dict()
    
    # Sort to find the Top 3 most influential features
    sorted_drivers = sorted(shap_values.items(), key=lambda x: abs(x[1]), reverse=True)[:3]
    driver_summary = ", ".join([f"{k.replace('SHAP_', '')} ({v:+.2f})" for k, v in sorted_drivers])
    
    return {
        "npi": top_hcp.get('Rndrng_NPI', 'N/A'),
        "state": top_hcp.get('Rndrng_Prvdr_State_Abrvtn', 'N/A'),
        "type": top_hcp.get('Cleaned_Prvdr_Type', 'N/A'),
        "payment": top_hcp.get('OP_MDCR_PYMT_PC', 0),
        "drivers": driver_summary,
        "score": top_hcp.get('pred_proba', 0)
    }