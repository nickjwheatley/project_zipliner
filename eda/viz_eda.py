import pandas as pd
from visualization import render_choropleth_map, render_choropleth_mapbox

br = 4
desired_metro_area = 'Phoenix-Mesa-Chandler, AZ'
df = pd.read_csv('../data/processed/prelim_merged_pivoted_data.csv')
df1 = df.loc[df.bedrooms == br]

viz = render_choropleth_mapbox(df, desired_metro_area, 'zhvi', 4)
viz.show()
print('test')