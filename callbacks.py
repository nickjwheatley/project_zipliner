from dash.dependencies import Input, Output
from app import app
from pages import home, page1, page2

@app.callback(Output('page-content', 'children'),
              Input('url', 'pathname'))
def display_page(pathname):
    if pathname == '/page1':
        return page1.layout()
    elif pathname == '/page2':
        return page2.layout()
    else:
        return home.layout()
