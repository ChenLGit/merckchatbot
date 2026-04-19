import pandas as pd
import plotly.express as px

def plot_executive_map(df):
    """
    Creates a US State map with custom hover data.
    Colors: Light Grey to Merck Dark Green (#00857c).
    Hover: Integer-only values with AI High Propensity count added.
    """
    # 1. Aggregate data by State for the pop-up/hover effect
    state_data = df.groupby('Rndrng_Prvdr_State_Abrvtn').agg({
        'Rndrng_NPI': 'count',
        'Tot_Benes': 'sum',
        'Tot_Bene_Day_Srvcs': 'sum',
        'BENE_AVG_AGE': 'mean',
        'OP_MDCR_PYMT_PC': 'mean',
        'pred_class': 'sum' # This counts how many are labeled '1' (High Propensity)
    }).reset_index()

    # 2. Convert to integers for the Tooltip
    state_data['Total Providers'] = state_data['Rndrng_NPI'].astype(int)
    state_data['Total Beneficiaries'] = state_data['Tot_Benes'].astype(int)
    state_data['Total Service Days'] = state_data['Tot_Bene_Day_Srvcs'].astype(int)
    state_data['Avg Beneficiary Age'] = state_data['BENE_AVG_AGE'].astype(int)
    state_data['Avg Medicare Payment'] = state_data['OP_MDCR_PYMT_PC'].astype(int)
    state_data['AI High Propensity (Yes)'] = state_data['pred_class'].astype(int)
    state_data['State'] = state_data['Rndrng_Prvdr_State_Abrvtn']

    # 3. Create the Choropleth Map
    # Color scale: 'lightgrey' to '#00857c' (Merck Heritage Green)
    fig = px.choropleth(
        state_data,
        locations='State',
        locationmode="USA-states",
        color='Total Providers',
        scope="usa",
        title="<b>National Market Presence & AI Targeting Overview</b>",
        color_continuous_scale=["lightgrey", "#00857c"],
        hover_name='State', # Header of the tooltip
        hover_data={
            'State': False, # Hide because it's in the header
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
    
    # Force tooltip formatting to be clean integers (removes .0000)
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