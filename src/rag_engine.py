import pandas as pd
import re

# Full names -> USPS abbreviations present in the dataset
_STATE_NAME_TO_ABBR = {
    "ALABAMA": "AL", "ALASKA": "AK", "ARIZONA": "AZ", "ARKANSAS": "AR", "CALIFORNIA": "CA",
    "COLORADO": "CO", "CONNECTICUT": "CT", "DELAWARE": "DE", "DISTRICT OF COLUMBIA": "DC",
    "FLORIDA": "FL", "GEORGIA": "GA", "HAWAII": "HI", "IDAHO": "ID", "ILLINOIS": "IL",
    "INDIANA": "IN", "IOWA": "IA", "KANSAS": "KS", "KENTUCKY": "KY", "LOUISIANA": "LA",
    "MAINE": "ME", "MARYLAND": "MD", "MASSACHUSETTS": "MA", "MICHIGAN": "MI", "MINNESOTA": "MN",
    "MISSISSIPPI": "MS", "MISSOURI": "MO", "MONTANA": "MT", "NEBRASKA": "NE", "NEVADA": "NV",
    "NEW HAMPSHIRE": "NH", "NEW JERSEY": "NJ", "NEW MEXICO": "NM", "NEW YORK": "NY",
    "NORTH CAROLINA": "NC", "NORTH DAKOTA": "ND", "OHIO": "OH", "OKLAHOMA": "OK", "OREGON": "OR",
    "PENNSYLVANIA": "PA", "RHODE ISLAND": "RI", "SOUTH CAROLINA": "SC", "SOUTH DAKOTA": "SD",
    "TENNESSEE": "TN", "TEXAS": "TX", "UTAH": "UT", "VERMONT": "VT", "VIRGINIA": "VA",
    "WASHINGTON": "WA", "WEST VIRGINIA": "WV", "WISCONSIN": "WI", "WYOMING": "WY",
}

def _valid_state_abbrs(df):
    out = set()
    for s in df["Rndrng_Prvdr_State_Abrvtn"].dropna().unique():
        s = str(s).strip().upper()
        if len(s) == 2:
            out.add(s)
    return out

def _normalize_query_for_state_match(query, valid_abbrs):
    """
    Uppercase query for abbr matching, but remove the classic false positive:
    the English preposition 'in' becomes 'IN' (Indiana). Pattern 'IN <STATE>'
    is almost always 'in [State]', not Indiana + another state.
    """
    upper = query.upper()

    def mask_in_before_state(m):
        nxt = m.group(1)
        return f"__PREP__ {nxt}" if nxt in valid_abbrs else m.group(0)

    upper = re.sub(r"\bIN\s+([A-Z]{2})\b", mask_in_before_state, upper)
    return upper

def _infer_state_from_query(query, df):
    """Return a single USPS abbr or None, using leftmost mention in the query."""
    valid_abbrs = _valid_state_abbrs(df)
    normalized = _normalize_query_for_state_match(query, valid_abbrs)

    # Longest name first so e.g. 'NEW YORK' wins over 'NEW' noise.
    for name in sorted(_STATE_NAME_TO_ABBR, key=len, reverse=True):
        abbr = _STATE_NAME_TO_ABBR[name]
        if abbr not in valid_abbrs:
            continue
        if re.search(rf"\b{re.escape(name)}\b", normalized):
            return abbr

    best_abbr, best_pos = None, len(normalized) + 1
    for abbr in sorted(valid_abbrs):
        m = re.search(rf"\b{re.escape(abbr)}\b", normalized)
        if m and m.start() < best_pos:
            best_pos, best_abbr = m.start(), abbr
    return best_abbr

def get_hcp_scorecard(query, df):
    target_state = _infer_state_from_query(query, df)
            
    # 2. Filtering Logic
    if target_state:
        state_col = df["Rndrng_Prvdr_State_Abrvtn"].astype(str).str.strip().str.upper()
        targets = df[(df["pred_class"] == 1) & (state_col == target_state)]
    else:
        targets = df[df['pred_class'] == 1]
    
    # Sort by the requested metric: Average Medicare Payment
    targets = targets.sort_values(by='OP_MDCR_PYMT_PC', ascending=False)
    
    if targets.empty:
        return None

    top_hcp = targets.iloc[0]
    
    # 3. Clean SHAP Drivers (Top 5)
    shap_cols = [c for c in df.columns if c.startswith('SHAP_')]
    shap_values = {}
    for col in shap_cols:
        try:
            shap_values[col] = float(str(top_hcp[col]).replace("'", ""))
        except:
            shap_values[col] = 0.0

    sorted_drivers = sorted(shap_values.items(), key=lambda x: abs(x[1]), reverse=True)[:5]
    driver_list = [f"{k.replace('SHAP_', '').replace('_', ' ').title()} ({v:+.2f})" for k, v in sorted_drivers]
    
    # 4. Final Scorecard Dictionary
    return {
        "npi": top_hcp.get('Rndrng_NPI', 'N/A'),
        "state": top_hcp.get('Rndrng_Prvdr_State_Abrvtn', 'N/A'),
        "city": top_hcp.get('Rndrng_Prvdr_City', 'N/A'), 
        "type": top_hcp.get('Cleaned_Prvdr_Type', 'N/A'),
        "payment": top_hcp.get('OP_MDCR_PYMT_PC', 0),
        "drivers": ", ".join(driver_list),
        "score": top_hcp.get('pred_proba', 0),
        # --- SYNTHETIC MARKETING FIELDS ---
        "digital_score": top_hcp.get('Digital_Adoption_Score', 0),
        "last_engagement": top_hcp.get('Last_Engagement_Days', 0),
        "channel": top_hcp.get('Preferred_Channel', 'Unknown')
    }