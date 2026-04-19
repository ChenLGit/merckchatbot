import pandas as pd
import plotly.express as px

def plot_executive_map(df):
    """
    Final Executive Version of the US State Map:
    - Centers title specifically to the map chart.
    - Uses Merck Branding (#00857c) for the high-end color scale.
    - Features accurate High Propensity targeting (pred_class == 1).
    - Formats all tooltips as integers while maintaining calculation accuracy.
    """
    
    # 1. Primary Aggregation (Counts and Sums)
    # Mean calculations correctly ignore NaNs to prevent skewing averages
    state_data = df.groupby('Rndrng_Prvdr_State_Abrvtn').agg({
        'Rndrng_NPI': 'count',
        'Tot_Benes': 'sum',
        'Tot_Bene_Day_Srvcs': 'sum',
        'BENE_AVG_AGE': 'mean',     
        'OP_MDCR_PYMT_PC': 'mean'   
    }).reset_index()

    # 2. Precise Count of High Propensity Targets (pred_class == 1)
    # We explicitly filter for the '1' class to ensure the count is accurate
    high_propensity_counts = df[df['pred_class'] == 1].groupby('Rndrng_Prvdr_State_Abrvtn')['Rndrng_NPI'].count().reset_index()
    high_propensity_counts.columns = ['Rndrng_Prvdr_State_Abrvtn', 'AI_High_Propensity']

    # Merge the count back into the state dataframe
    state_data = state_data.merge(high_propensity_counts, on='Rndrng_Prvdr_State_Abrvtn', how='left')
    
    # Fill missing propensity counts with 0 (since N/A here means no doctors met the criteria)
    state_data['AI_High_Propensity'] = state_data['AI_High_Propensity'].fillna(0)

    # Standardize column names for the Plotly engine
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
    # projection_type 'albers usa' ensures the map stays centered and scaled properly
    fig = px.choropleth(
        state_data,
        locations='State',
        locationmode="USA-states",
        color='Total Providers',
        scope="usa",
        # Custom color scale: Neutral Light Grey to Merck Heritage Green
        color_continuous_scale=["#E5E7E9", "#00857c"],
        hover_name='State',
        custom_data=[
            'Total Providers', 
            'Total Beneficiaries', 
            'Total Service Days', 
            'Avg Age', 
            'Avg Payment', 
            'High Propensity Yes'
        ]
    )
    
    # 4. Centralized Title and Refined Layout
    # title_x: 0.5 centers the text relative to the chart container
    fig.update_layout(
        title={
            'text': "<b>National Market Presence & AI Targeting Overview</b>",
            'y': 0.95,
            'x': 0.5,
            'xanchor': 'center',
            'yanchor': 'top'
        },
        margin={"r":20,"t":80,"l":20,"b":0},
        geo=dict(
            bgcolor='rgba(0,0,0,0)',
            projection_type='albers usa'
        ),
        coloraxis_colorbar=dict(title="Provider Count")
    )
    
    # 5. Professional Integer Formatting in Tooltip
    # Using :.0f in the template removes decimals without forcing hard-casts on NaN data
    fig.update_traces(
        hovertemplate="<br>".join([
            "<b>%{hovertext}</b>",
            "Total Providers: %{customdata[0]:,d}",
            "Total Beneficiaries: %{customdata[1]:,.0f}",
            "Total Service Days: %{customdata[2]:,.0f}",
            "Avg Beneficiary Age: %{customdata[3]:.0f}",
            "Avg Medicare Payment: $%{customdata[4]:,.0f}",
            "AI High Propensity (Yes): %{customdata[5]:.0f}"
        ])
    )
    
    return fig