import pandas as pd
import plotly.express as px

def plot_executive_map(df):
    """
    Refined US State map.
    - Colors: Light Grey to Merck Heritage Green (#00857c).
    - Tooltip: Clean integers with high-propensity counts.
    - Data Safety: Handles missing values without skewing averages.
    """
    
    # 1. Primary Aggregation (Counts and Sums)
    state_data = df.groupby('Rndrng_Prvdr_State_Abrvtn').agg({
        'Rndrng_NPI': 'count',
        'Tot_Benes': 'sum',
        'Tot_Bene_Day_Srvcs': 'sum',
        'BENE_AVG_AGE': 'mean',     # Mean correctly ignores NaNs
        'OP_MDCR_PYMT_PC': 'mean'   # Mean correctly ignores NaNs
    }).reset_index()

    # 2. Precise Count of High Propensity (pred_class == 1)
    # We filter specifically for the '1' class and merge to ensure accuracy
    high_propensity_counts = df[df['pred_class'] == 1].groupby('Rndrng_Prvdr_State_Abrvtn')['Rndrng_NPI'].count().reset_index()
    high_propensity_counts.columns = ['Rndrng_Prvdr_State_Abrvtn', 'AI_High_Propensity']

    # Merge the count back into our state data
    state_data = state_data.merge(high_propensity_counts, on='Rndrng_Prvdr_State_Abrvtn', how='left')
    
    # Fill missing propensity counts with 0 (since N/A here means zero matches)
    state_data['AI_High_Propensity'] = state_data['AI_High_Propensity'].fillna(0)

    # Rename for the Plotly engine
    state_data.rename(columns={
        'Rndrng_Prvdr_State_Abrvtn': 'State',
        'Rndrng_NPI': 'Total Providers',
        'Tot_Benes': 'Total Beneficiaries',
        'Tot_Bene_Day_Srvcs': 'Total Service Days',
        'BENE_AVG_AGE': 'Avg Age',
        'OP_MDCR_PYMT_PC': 'Avg Payment',
        'AI_High_Propensity': 'High Propensity Yes'
    }, inplace=True)

    # 3. Create the Choropleth Map
    fig = px.choropleth(
        state_data,
        locations='State',
        locationmode="USA-states",
        color='Total Providers',
        scope="usa",
        title="<b>National Market Presence & AI Targeting Overview</b>",
        # Color scale: Light Grey (#E5E7E9) to Merck Dark Green (#00857c)
        color_continuous_scale=["#E5E7E9", "#00857c"],
        hover_name='State',
        # Pass data to the template without casting to int (to avoid crashes)
        custom_data=[
            'Total Providers', 
            'Total Beneficiaries', 
            'Total Service Days', 
            'Avg Age', 
            'Avg Payment', 
            'High Propensity Yes'
        ]
    )
    
    # 4. Professional Integer Formatting in Tooltip
    # :.0f handles the "integer look" for floats and handles NaNs gracefully
    fig.update_traces(
        hovertemplate="<br>".join([
            "<b>%{hovertext}</b>",
            "Total Providers: %{customdata[0]:,d}",
            "Total Beneficiaries: %{customdata[1]:, .0f}",
            "Total Service Days: %{customdata[2]:, .0f}",
            "Avg Beneficiary Age: %{customdata[3]:.0f}",
            "Avg Medicare Payment: $%{customdata[4]:,.0f}",
            "AI High Propensity (Yes): %{customdata[5]:.0f}"
        ])
    )

    fig.update_layout(
        margin={"r":0,"t":40,"l":0,"b":0},
        geo=dict(bgcolor='rgba(0,0,0,0)'),
        coloraxis_colorbar=dict(title="Provider Count")
    )
    return fig