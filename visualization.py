import pandas as pd
import plotly.express as px
import json
from urllib.request import urlopen
import plotly.graph_objects as go

def get_state_name_from_abbreviation(state_abrv):
    """
    Get the full state name from a state abbreviation. Used to get the geo_codes for zip code boundaries
    :param state_abrv: string state abbreviation
    :return: string state name
    """
    state_names = {
        'AL': 'Alabama',
        'AK': 'Alaska',
        'AZ': 'Arizona',
        'AR': 'Arkansas',
        'CA': 'California',
        'CO': 'Colorado',
        'CT': 'Connecticut',
        'DE': 'Delaware',
        'DC': 'District of Columbia',
        'FL': 'Florida',
        'GA': 'Georgia',
        'HI': 'Hawaii',
        'ID': 'Idaho',
        'IL': 'Illinois',
        'IN': 'Indiana',
        'IA': 'Iowa',
        'KS': 'Kansas',
        'KY': 'Kentucky',
        'LA': 'Louisiana',
        'ME': 'Maine',
        'MT': 'Montana',
        'NE': 'Nebraska',
        'NV': 'Nevada',
        'NH': 'New Hampshire',
        'NJ': 'New Jersey',
        'NM': 'New Mexico',
        'NY': 'New York',
        'NC': 'North Carolina',
        'ND': 'North Dakota',
        'OH': 'Ohio',
        'OK': 'Oklahoma',
        'OR': 'Oregon',
        'MD': 'Maryland',
        'MA': 'Massachusetts',
        'MI': 'Michigan',
        'MN': 'Minnesota',
        'MS': 'Mississippi',
        'MO': 'Missouri',
        'PA': 'Pennsylvania',
        'RI': 'Rhode Island',
        'SC': 'South Carolina',
        'SD': 'South Dakota',
        'TN': 'Tennessee',
        'TX': 'Texas',
        'UT': 'Utah',
        'VT': 'Vermont',
        'VA': 'Virginia',
        'WA': 'Washington',
        'WV': 'West Virginia',
        'WI': 'Wisconsin',
        'WY': 'Wyoming'
    }
    return state_names[state_abrv]


def get_geo_json_codes(state_abbr, desired_zip_codes):
    """
    Retrieves the geo boundaries for all zip codes in the desired list.
    :param state: string indicating long state name of where the zip codes reside
    :param desired_zip_codes: list of zip codes in the desired metro area
    :return: dictionary containing the geo boundary codes of all desired zip codes
    """
    # Create URL for geo_json_codes
    base = 'https://raw.githubusercontent.com/OpenDataDE/State-zip-code-GeoJSON/master/'
    state = f'{state_abbr.lower()}_{get_state_name_from_abbreviation(state_abbr).lower()}'
    suffix = '_zip_codes_geo.min.json'
    geojson_url = f'{base}{state}{suffix}'

    # Extract geojson boundaries for all zip codes in the state
    with urlopen(geojson_url) as response:
        all_zip_code_boundaries = json.load(response)

    # Filter zip code boundaries dictionary to zip codes in desired metro region
    desired_zip_code_boundaries = {
        'type': 'FeatureCollection',
        'features': []
    }
    for zcb in all_zip_code_boundaries['features']:
        if int(zcb['properties']['ZCTA5CE10']) in desired_zip_codes:
            desired_zip_code_boundaries['features'].append(zcb)

    return desired_zip_code_boundaries


def render_choropleth_map(df, metro_area, desired_metric):
    """
    Renders a choropleth map of housing data of all the zip codes in a desired metropolitan area
    :param df: pandas dataframe containing appeal index data and other valuable housing metrics
    :param metro_area: string indicating desired metropolitan area to render
    :return: plotly.express choropleth figure
    """
    desired_zip_codes = df.loc[df.metro == metro_area, 'zip_code'].unique().tolist()
    desired_state_abbr = df.loc[df.metro == metro_area,'state'].iloc[0]
    zip_code_boundaries = get_geo_json_codes(desired_state_abbr, desired_zip_codes)

    fig = px.choropleth(df.loc[df.metro == metro_area],
                        geojson=zip_code_boundaries,
                        locations='zip_code',
                        color=desired_metric,
                        color_continuous_scale="bluyl",
                        featureidkey="properties.ZCTA5CE10",
                        scope="usa",
                        fitbounds='locations',
                        labels={
                            'city': 'City',
                            # 'ZHVI': 'ZHVI',
                            # 'ZHVI_diff': 'Change in Price',
                            # 'mean_monthly_change': 'Mean Monthly Change',
                            desired_metric: desired_metric,
                            'Affordability_Ratio': 'Affordability Ratio'
                        },
                        hover_data=['city', 'zip_code', desired_metric, 'Affordability_Ratio'],
                        # width = 1500,
                        height = 700
                        )
    # fig.update_layout(margin={"r": 20, "t": 0, "l": 20, "b": 0}) #removing because it re-renders the viz taking longer

    return fig


def make_progress_graph(progress, total):
    progress_graph = (
        go.Figure(data=[go.Bar(x=[progress])])
        .update_xaxes(range=[0, total])
        .update_yaxes(
            showticklabels=False,
        )
        .update_layout(height=100, margin=dict(t=20, b=40))
    )
    return progress_graph