import pandas as pd
import plotly.express as px
import json
from urllib.request import urlopen
import plotly.graph_objects as go
from all_labels import get_metric_labels
import math
import numpy as np


def lat_lon_to_cartesian(lat, lon):
    """
    Converts latitude and longitude to Cartesian coordinates.

    Args:
        lat (float): Latitude in degrees.
        lon (float): Longitude in degrees.

    Returns:
        tuple: A tuple containing the Cartesian coordinates (x, y, z).
    """
    lat, lon = math.radians(lat), math.radians(lon)
    x = math.cos(lat) * math.cos(lon)
    y = math.cos(lat) * math.sin(lon)
    z = math.sin(lat)
    return x, y, z


def cartesian_to_lat_lon(x, y, z):
    """
    Converts Cartesian coordinates to latitude and longitude.

    Args:
        x (float): X coordinate.
        y (float): Y coordinate.
        z (float): Z coordinate.

    Returns:
        tuple: A tuple containing the latitude and longitude in degrees.
    """
    lon = math.atan2(y, x)
    hyp = math.sqrt(x * x + y * y)
    lat = math.atan2(z, hyp)
    return math.degrees(lat), math.degrees(lon)


def find_centroid(coords):
    """
    Finds the geographical center (centroid) of a set of latitude and longitude coordinates.

    Args:
        coords (list): A list of tuples, where each tuple contains the latitude and longitude.

    Returns:
        tuple: A tuple containing the latitude and longitude of the centroid.
    """
    x_sum = y_sum = z_sum = 0
    num_coords = len(coords)

    for lat, lon in coords:
        x, y, z = lat_lon_to_cartesian(lat, lon)
        x_sum += x
        y_sum += y
        z_sum += z

    x_avg = x_sum / num_coords
    y_avg = y_sum / num_coords
    z_avg = z_sum / num_coords

    return cartesian_to_lat_lon(x_avg, y_avg, z_avg)
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


def get_geo_json_codes(state_abbr_lst, desired_zip_codes):
    """
    Retrieves the geo boundaries for all zip codes in the desired list.
    :param state_abbr_lst: list of strings indicating short state name of where the zip codes reside
    :param desired_zip_codes: list of zip codes in the desired metro area
    :return: dictionary containing the geo boundary codes of all desired zip codes
    """
    # Filter zip code boundaries dictionary to zip codes in desired metro region
    desired_zip_code_boundaries = {
        'type': 'FeatureCollection',
        'features': []
    }
    for state_abbr in state_abbr_lst:
        # Create URL for geo_json_codes
        base = 'https://raw.githubusercontent.com/OpenDataDE/State-zip-code-GeoJSON/master/'
        state = f'{state_abbr.lower()}_{'_'.join(get_state_name_from_abbreviation(state_abbr).split(' ')).lower()}'
        suffix = '_zip_codes_geo.min.json'
        geojson_url = f'{base}{state}{suffix}'

        # Extract geojson boundaries for all zip codes in the state
        with urlopen(geojson_url) as response:
            all_zip_code_boundaries = json.load(response)

        for zcb in all_zip_code_boundaries['features']:
            if zcb['properties']['ZCTA5CE10'] in desired_zip_codes:
                desired_zip_code_boundaries['features'].append(zcb)

    return desired_zip_code_boundaries


def render_choropleth_map(df, metro_area, desired_metric, num_bedrooms):
    """
    Renders a choropleth map of housing data of all the zip codes in a desired metropolitan area
    :param df: pandas dataframe containing appeal index data and other valuable housing metrics
    :param metro_area: string indicating desired metropolitan area to render
    :return: plotly.express choropleth figure
    """
    desired_zip_codes = df.loc[df.metro == metro_area, 'zip_code'].unique().tolist()
    desired_state_abbrs = df.loc[df.metro == metro_area,'state'].unique().tolist()
    zip_code_boundaries = get_geo_json_codes(desired_state_abbrs, desired_zip_codes)
    if desired_metric == 'zhvi':
        fmt = ',.0f'
    else:
        fmt = True

    fig = px.choropleth(df.loc[(df.metro == metro_area) & (df.bedrooms == num_bedrooms)],
                        geojson=zip_code_boundaries,
                        locations='zip_code',
                        color=desired_metric,
                        color_continuous_scale="bluyl",
                        featureidkey="properties.ZCTA5CE10",
                        scope="usa",
                        fitbounds='locations',
                        labels={
                            'zip_code': 'Zip Code',
                            'city': 'City',
                            desired_metric: get_metric_labels()[desired_metric]
                        },
                        hover_data={
                            'city':True,
                            'zip_code':True,
                            desired_metric:True
                        },
                        title = f'{metro_area} - {num_bedrooms} bedrooms'
                        # width = 1500,
                        # height = 500
                        )

    fig.update_layout(
        margin=dict(
            l=0,  # left margin
            r=0,  # right margin
            b=0,  # bottom margin
            t=45  # top margin
        ),
        legend=dict(
            x=0,  # Set the x position of the legend (0 is the far left, 1 is the far right)
            xanchor='left',  # Set the x anchor
            y=1  # Set the y position of the legend (1 is the top, 0 is the bottom)
        )
    )

    return fig


def render_choropleth_mapbox(df, metro_area, desired_metric, num_bedrooms):
    """
    Renders a choropleth map of housing data of all the zip codes in a desired metropolitan area
    :param df: pandas dataframe containing appeal index data and other valuable housing metrics
    :param metro_area: string indicating desired metropolitan area to render
    :return: plotly.express choropleth figure
    """
    desired_zip_codes = df.loc[df.metro == metro_area, 'zip_code'].unique().tolist()
    desired_state_abbrs = df.loc[df.metro == metro_area, 'state'].unique().tolist()
    zip_code_boundaries = get_geo_json_codes(desired_state_abbrs, desired_zip_codes)

    coords = zip_code_boundaries['features'][0]['geometry']['coordinates'][0]

    # Catch incorrect nestings
    if (len(coords) == 1) & (len(desired_zip_codes) != 1):
        coords = coords[0]

    if desired_metric == 'zhvi':
        fmt = ',.0f'
    else:
        fmt = True

    center = np.array(coords).mean(axis=0).tolist()
    fig = px.choropleth_mapbox(df.loc[(df.metro == metro_area) & (df.bedrooms == num_bedrooms)],
                        geojson=zip_code_boundaries,
                        locations='zip_code',
                        color=desired_metric,
                        color_continuous_scale="bluyl",
                        featureidkey="properties.ZCTA5CE10",
                        # fitbounds='locations',
                        mapbox_style='carto-positron',
                        labels={
                            'zip_code': 'Zip Code',
                            'city': 'City',
                            desired_metric: get_metric_labels()[desired_metric]
                        },
                        hover_data={
                            'city':True,
                            'zip_code':True,
                            desired_metric: True
                        },
                        title = f'{metro_area} - {num_bedrooms} bedrooms',
                        opacity=0.25,
                        center = {'lat': center[1], 'lon': center[0]},
                        zoom = 8
                        # width = 1500,
                        # height = 500
                        )

    fig.update_layout(
        margin=dict(
            l=0,  # left margin
            r=0,  # right margin
            b=0,  # bottom margin
            t=45  # top margin
        ),
        legend=dict(
            x=0,  # Set the x position of the legend (0 is the far left, 1 is the far right)
            xanchor='left',  # Set the x anchor
            y=1  # Set the y position of the legend (1 is the top, 0 is the bottom)
        )
    )

    # fig.update_traces(
    #     hovertemplate=f"Value: ${desired_metric:,.0f}<br>"  # Format number with comma separator
    # )

    return fig


def render_time_series_plot(df, zip_code, bedrooms):
    df.zip_code = df.zip_code.apply(lambda x: f'{x:05}')
    viz_df = df.loc[(df.zip_code == zip_code) & (df.bedrooms == int(bedrooms))].sort_values('date')

    fig = px.line(
        viz_df,
        x='date',
        y='zhvi',
        color='zip_code',
        # color_discrete_sequence=['bluyl'],
        labels = {
            'zip_code':'Zip Code',
            'zhvi':'Zillow Home Value Index',
            'date':'Date'
        },
        hover_data={'zhvi':True},
        title = f'Zillow Home Values for {zip_code}'
    )

    fig.update_layout(
        margin=dict(
            l=0,  # left margin
            r=0,  # right margin
            b=0,  # bottom margin
            t=45  # top margin
        )
    )

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