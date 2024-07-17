import pandas as pd
from visualization import render_choropleth_map

br = 4
desired_metro_area = 'Phoenix-Mesa-Chandler, AZ'
df = pd.read_csv('../data/processed/zillow_current_snapshot.csv')
df1 = df.loc[df.bedrooms == br]

viz = render_choropleth_map(df1, desired_metro_area, 'zhvi')
viz.show()
print('test')