import pandas as pd
import plotly.express as px

def plot_executive_map(df):
    """
    Creates a US State map with custom hover data.
    Safe version: handles NaN (missing) values to prevent IntCastingNaNError.
    """
    # 1. Aggregate data by State
    state_data = df.groupby('Rndrng_Prvdr_State_Abrvtn').agg({
        'Rndrng_NPI': 'count',
        'Tot_Benes': 'sum',
        'Tot_Bene_Day_Srvcs': 'sum',
        'BENE_AVG_AGE': 'mean',
        'OP_MDCR_PYMT_PC': 'mean',
        'pred_class': 'sum'
    }).reset_index()

    # 2. SAFE CONVERSION: Fill missing values with 0 before converting to integer
    state_data['Total Providers'] = state_data['Rndrng_NPI'].fillna(0).astype(int)
    state_data['Total Beneficiaries'] = state_data['Tot_Benes'].fillna(0).astype(int)
    state_data['Total Service Days'] = state_data['Tot_Bene_Day_Srvcs'].fillna(0).astype(int)
    state_data['Avg Beneficiary Age'] = state_data['BENE_AVG_AGE'].fillna(0).astype(int)
    state_data['Avg Medicare Payment'] = state_data['OP_MDCR_PYMT_PC'].fillna(0).astype(int)
    state_data['AI High Propensity (Yes)'] = state_data['pred_class'].fillna(0).astype(int)
    
    state_data['State'] = state_data['Rndrng_Prvdr_State_Abrvtn']

    # 3. Create the Choropleth Map
    fig = px.choropleth(
        state_data,
        locations='State',
        locationmode="USA-states",
        color='Total Providers',
        scope="usa",
        title="<b>National Market Presence & AI Targeting Overview</b>",
        color_continuous_scale=["lightgrey", "#00857c"], # Merck Heritage Green
        hover_name='State',
        hover_data={
            'State': False,
            'Total Providers': True,
            'Total Beneficiaries': True,
            'Total Service Days': True,
            'Avg Beneficiary Age': True,
            'Avg Medicare Payment': True,
            'AI High Propensity (Yes)': True
        }
    )
    
    # 4. Final Layout Polishing
    fig.update_layout(
        margin={"r":0,"t":40,"l":0,"b":0},
        geo=dict(bgcolor='rgba(0,0,0,0)'),
        coloraxis_colorbar=dict(title="Provider Count")
    )
    
    # Clean integer Tooltip formatting
    fig.update_traces(hovertemplate="<br>".join([
        "<b>%{hovertext}</b>",
        "Total Providers: %{customdata[0]}",
        "Total Beneficiaries: %{customdata[1]}",
        "Total Service Days: %{customdata[2]}",
        "Avg Beneficiary Age: %{customdata[3]}",
        "Avg Medicare Payment: $%{customdata[4]}",
        "AI High Propensity (Yes): %{customdata[5]}"
    ]))

    return fig