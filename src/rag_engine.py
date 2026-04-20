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

def _infer_state_from_query(query, df):
    """
    Return a single USPS state abbreviation or None.

    Strategy (in order):
      1. Match a full state name case-insensitively ('New Jersey', 'Indiana').
      2. Match a two-letter abbreviation that is UPPERCASE in the original
         query ('NJ', 'TX'). This avoids the classic false positives where
         common English words collide with state codes after an uppercase()
         call: 'me' -> ME (Maine), 'or' -> OR (Oregon), 'hi' -> HI (Hawaii),
         'in' -> IN (Indiana), 'ok' -> OK (Oklahoma).
      3. Skip 'IN' when it's being used as the English preposition immediately
         before another state abbreviation (e.g. 'opportunities IN NJ').
    """
    valid_abbrs = _valid_state_abbrs(df)
    q = str(query or "")
    if not q:
        return None

    upper = q.upper()
    for name in sorted(_STATE_NAME_TO_ABBR, key=len, reverse=True):
        abbr = _STATE_NAME_TO_ABBR[name]
        if abbr not in valid_abbrs:
            continue
        if re.search(rf"\b{re.escape(name)}\b", upper):
            return abbr

    matches = list(re.finditer(r"\b([A-Z]{2})\b", q))
    for i, m in enumerate(matches):
        abbr = m.group(1)
        if abbr not in valid_abbrs:
            continue
        # 'IN' immediately followed by another state abbr is the preposition.
        if abbr == "IN" and i + 1 < len(matches) and matches[i + 1].group(1) in valid_abbrs:
            continue
        return abbr
    return None

def extract_npi(query):
    """Return the first standalone 10-digit NPI in the query, else None."""
    if not query:
        return None
    m = re.search(r"(?<!\d)(\d{10})(?!\d)", str(query))
    return m.group(1) if m else None


_extract_npi = extract_npi


def _lookup_by_npi(df, npi):
    """Return the row for a specific NPI (as a Series) or None if not found."""
    npi_str = str(npi).strip()
    npi_series = df["Rndrng_NPI"].astype(str).str.strip()
    match = df[npi_series == npi_str]
    if match.empty:
        # Fallback: compare numerically to cover zero-padded or formatted values.
        try:
            match = df[pd.to_numeric(df["Rndrng_NPI"], errors="coerce") == int(npi_str)]
        except ValueError:
            match = df.iloc[0:0]
    if match.empty:
        return None
    return match.iloc[0]


def get_hcp_scorecard(query, df):
    # 1. Explicit NPI lookup overrides any other filter so the user always
    #    gets exactly the provider they asked about, regardless of intent.
    matched_by_npi = False
    npi_requested = extract_npi(query)
    if npi_requested:
        row = _lookup_by_npi(df, npi_requested)
        if row is None:
            return None
        top_hcp = row
        matched_by_npi = True
    else:
        # 2. Otherwise filter by (optional) state and pick the top propensity.
        target_state = _infer_state_from_query(query, df)
        if target_state:
            state_col = df["Rndrng_Prvdr_State_Abrvtn"].astype(str).str.strip().str.upper()
            targets = df[(df["pred_class"] == 1) & (state_col == target_state)]
        else:
            targets = df[df["pred_class"] == 1]

        # Rank "top" by model propensity so OPPORTUNITY and MARKETING agree
        # on what the #1 provider is for a given filter.
        targets = targets.sort_values(by="pred_proba", ascending=False)
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
        # --- MARKETING FIELDS ---
        "digital_score": top_hcp.get('Digital_Adoption_Score', 0),
        "last_engagement": top_hcp.get('Last_Engagement_Days', 0),
        "channel": top_hcp.get('Preferred_Channel', 'Unknown'),
        # --- Lookup metadata ---
        "matched_by_npi": matched_by_npi,
        "requested_npi": npi_requested,
    }