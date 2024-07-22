from dash import html, Dash, dash_table
import pandas as pd
import dash_bootstrap_components as dbc
from data_extract import query_rds, get_rds_schema
import os

ON_HEROKU = os.getenv('ON_HEROKU')
if (ON_HEROKU == True) | (ON_HEROKU == 'TRUE'):
    host = '0.0.0.0'
else:
    host = '127.0.0.1'


df_query = "SELECT metro,zip_code FROM prelim_merged_pivoted_snapshot LIMIT 10;"

df = query_rds(df_query)

app = Dash(__name__, external_stylesheets=[dbc.themes.CYBORG])
app.layout = html.Div([
        html.H1("Heroku Boot Test"),
    html.Div([
        dash_table.DataTable(
            id='table',
            columns=[{"name": i, "id": i} for i in df.columns],
            data=df.to_dict('records'),
            style_table={'overflowX': 'auto'},
            style_cell={'textAlign': 'left'},
            style_header={
                'backgroundColor': 'rgb(230, 230, 230)',
                'fontWeight': 'bold'
            },
            style_data_conditional=[
                {
                    'if': {'row_index': 'odd'},
                    'backgroundColor': 'rgb(248, 248, 248)'
                }
            ]
        )
    ])
])

if __name__ == "__main__":
    app.run_server(debug=True, port=int(os.environ.get('PORT', 8050)), host=host)
