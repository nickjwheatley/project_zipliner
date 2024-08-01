import dash_bootstrap_components as dbc
from dash import html, dash_table
import json
from all_labels import get_metric_labels
import numpy as np

metric_labels = get_metric_labels()

label_style = {
        'font-weight': 'bold',
        'text-align': 'center',
        'font-size': '20px',
        'color': '#161616'
}
paragraph_style = {
    'color': '#676767'
}
card_style = {
    'border': '2px solid #4d4d4d',
    'padding': '5',
    'paddingLeft': '20px',
    'paddingRight': '20px',
    'borderRadius': '5px',
    'backgroundColor': 'white'
}

cell_style = {'border':'1px solid black', 'padding':'5px'}

def generate_table_body_rows(df_gs):
    body = []
    for i in df_gs.index:
        series = df_gs.loc[i]
        body.append(
            html.Tr([
                html.Td(series['School Type'], style=cell_style),
                html.Td(series['Level'], style=cell_style),
                html.Td(f"{round(series['Rating'],1)}/10", style=cell_style),
                html.Td(round(series['Distance (M)'],1), style=cell_style),
            ])
        )
    return body

def generate_gs_rating_tooltip(df_gs):
    return dbc.Tooltip([
            html.P('Average school ratings from GreatSchools.org'),
            html.Table([
                html.Thead([
                    html.Tr([
                        html.Th("School Type", style= cell_style),
                        html.Th("Level", style=cell_style),
                        html.Th("Rating", style=cell_style),
                        html.Th("Distance (M)", style=cell_style)
                    ])
                ], style={'border':'3px solid black', 'fontWeight':'bold'}),
                html.Tbody(generate_table_body_rows(df_gs), style={'border':'3px solid black'})
            ], style={
                'border':'1px solid black'
            })],
            target="great_schools_ratings",
            placement="bottom",
            class_name='custom-tooltip'
            # style={"backgroundColor": "green", "border": "1px solid #ccc", "padding": "1px", "width": "600", "opactiy": "1"}  # Adjust the width if necessary
        )
def generate_card_element(el, items, df, df_gs, data_dictionary):
    if el == 'mean_education_rating':
        title = None
        return html.Span([
            generate_gs_rating_tooltip(df_gs),
            html.Span([
                metric_labels[el],
                '' if items['sup'] is None else html.Sup(items['sup']),  # Add superscript if needed
                ': '
            ], style={'float': 'left', 'font-weight': 'bold'}
            ),
            html.Span(
                'No Data' if np.isnan(df[el].iloc[0]) \
                    else f"{items['format_prefix']}{df[el].iloc[0]:{items['value_format']}}{items['format_suffix']}",
                style={'float': 'right'}
            )],
            id=items['id'])
    else:
        return html.Span([
            html.Span([
                metric_labels[el],
                '' if items['sup'] is None else html.Sup(items['sup']), # Add superscript if needed
                ': '
            ], style={'float': 'left', 'font-weight': 'bold'}
            ),
            html.Span(
                'No Data' if np.isnan(df[el].iloc[0]) \
                    else f"{items['format_prefix']}{df[el].iloc[0]:{items['value_format']}}{items['format_suffix']}",
                style={'float': 'right'}
            )],
            title=data_dictionary[el]
        )


def generate_card(label, dict, df, df_gs, data_dictionary):
    CARD_WIDTH = 3
    card_content = []
    for el in dict[label].keys():
        card_content.append(generate_card_element(el, dict[label][el], df, df_gs, data_dictionary))
        card_content.append(html.Br())
    return dbc.Col([
        html.Label(
            label,
            style = label_style
        ),
        html.P(card_content, style=paragraph_style)
    ], width=CARD_WIDTH, style=card_style)


def generate_populated_cards(df, df_gs, data_dictionary):
    card_sections = {
        'Economic': {
            'economic_diversity_index': {
                'sup': 'c',
                'id':'None',
                'format_prefix':'',
                'format_suffix':'',
                'value_format':'.2f'
            },
            'est_number_of_jobs': {
                'sup': 'c',
                'id':'None',
                'format_prefix': '',
                'format_suffix': '',
                'value_format': ',.0f'
            },
            'median_income_families': {
                'sup': None,
                'id':'None',
                'format_prefix': '$',
                'format_suffix': '/yr',
                'value_format': ',.0f'
            },
            'median_income': {
                'sup': None,
                'id':'None',
                'format_prefix': '$',
                'format_suffix': '/yr',
                'value_format': ',.0f'
            }
        },
        'Cost of Living': {
            'zhvi': {
                'sup': None,
                'id':'None',
                'format_prefix': '$',
                'format_suffix': '',
                'value_format': ',.0f'
            },
            'median_real_estate_taxes': {
                'sup': None,
                'id':'None',
                'format_prefix': '',
                'format_suffix': '',
                'value_format': ',.0f'
            },
            'affordability_ratio': {
                'sup': None,
                'id':'None',
                'format_prefix': '',
                'format_suffix': '',
                'value_format': '.2f'
            }
        },
        'Quality of Life': {
            'mean_travel_time_to_work': {
                'sup': None,
                'id':'None',
                'format_prefix': '',
                'format_suffix': '',
                'value_format': '.1f'
            },
            'total_crime_rate': {
                'sup': None,
                'id':'None',
                'format_prefix': '',
                'format_suffix': '',
                'value_format': '.3f'
            },
            'mean_education_rating': {
                'sup': None,
                'id':'great_schools_ratings',
                'format_prefix': '',
                'format_suffix': '',
                'value_format': '.1f'
            }
        },
        'Demographic': {
            'total_working_age_population': {
                'sup': None,
                'id':'None',
                'format_prefix': '',
                'format_suffix': '',
                'value_format': ',.0f'
            },
            'median_age': {
                'sup': None,
                'id':'None',
                'format_prefix': '',
                'format_suffix': '',
                'value_format': '.0f'
            },
            'higher_education': {
                'sup': None,
                'id':'None',
                'format_prefix': '',
                'format_suffix': '%',
                'value_format': '.1f'
            }
        }
    }

    cards_content = []
    for label in card_sections.keys():
        cards_content.append(generate_card(label, card_sections, df, df_gs, data_dictionary))

    return dbc.Row(cards_content)




