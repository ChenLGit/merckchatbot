import pandas as pd
import plotly.express as px

def plot_executive_map(df):
    """
    Creates a US State map with custom hover data.
    Aggregates: Provider Count, Total Benes, Total Services, Avg Age, and Avg Payment.
    """
    # Aggregate data by State for the pop-up/hover effect
    state_data = df.groupby('Rndrng_Prvdr_State_Abrvtn').agg({
        'Rndrng_NPI': 'count',
        'Tot_Benes': 'sum',
        'Tot_Bene_Day_Srvcs': 'sum',
        'BENE_AVG_AGE': 'mean',
        'OP_MDCR_PYMT_PC': 'mean'
    }).reset_index()

    # Rename columns for cleaner hover labels in the UI
    state_data.columns = [
        'State', 'Total Providers', 'Total Beneficiaries', 
        'Total Service Days', 'Avg Beneficiary Age', 'Avg Medicare Payment'
    ]

    # Create the Choropleth Map using State Abbreviations
    fig = px.choropleth(
        state_data,
        locations='State',
        locationmode="USA-states",
        color='Total Providers',
        scope="usa",
        title="<b>National Market Presence & Provider Metrics</b>",
        color_continuous_scale="Viridis",
        hover_data={
            'State': True,
            'Total Providers': ':,',
            'Total Beneficiaries': ':,',
            'Total Service Days': ':,',
            'Avg Beneficiary Age': ':.1f',
            'Avg Medicare Payment': ':$ ,.2f'
        }
    )
    
    fig.update_layout(
        margin={"r":0,"t":40,"l":0,"b":0},
        geo=dict(bgcolor='rgba(0,0,0,0)')
    )
    
    return fig