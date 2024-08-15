# MADS Capstone: Project Zipliner
## Overview
[Project Zipliner](http://www.projectzipliner.com/) is a Dash web application designed to assist potential home buyers select which region to buy a home. The site provides information at the zip code level, including the following:

- Zillow Home Value Index (Median home price)
- Crime rates
- School ratings
- Commute time
- Median income
- Population
- Number of jobs
- etc

![image](https://github.com/user-attachments/assets/49899b81-d0ce-4f16-a6ef-66cd803d2ee3)

## How to Run the App
With ChatGPT and AWS credentials from the SECRETS.ini file not included in the repo (contact authors for info - must be stored in project root path), users can launch the web app from app.py or by accessing www.projectzipliner.com. Data can be refreshed using data_extract.py.
Keep in mind that this app is experimental and can only support a few users at a time.

## Data Access Statement
All data powering the web app was extracted from one of the sources below according to their license agreements.
| App Metric | Data Source | Extraction Method |
| ------------- | ------------- | ------------- |
| Zillow Home Value Index (ZHVI) | [Zillow Research](https://www.zillow.com/research/data/) | [data/zillow_data.py](https://github.com/nickjwheatley/project_zipliner/blob/main/data/zillow_data.py) |
| School Ratings | [GreatSchools.org API](https://documenter.getpostman.com/view/13485071/2s93sgXWUY?utm_campaign=API&utm_source=gs&utm_medium=textlink&utm_content=api_documentation) | [data/great_schools_data.py](https://github.com/nickjwheatley/project_zipliner/blob/main/data/great_schools_data.py)
|Commuter Characteristics (ACS) | [American Census Survey Commuter Characteristics](https://data.census.gov/table?q=S0801:%20Commuting%20Characteristics%20by%20Sex&g=010XX00US$8600000) | [data/commuter_characteristics.py](https://github.com/nickjwheatley/project_zipliner/blob/main/data/commuter_characteristics.py) |
|Housing Costs (ACS) | [American Census Survey Housing Costs](https://data.census.gov/table?q=S2506:%20Financial%20Characteristics%20for%20Housing%20Units%20With%20a%20Mortgage&g=010XX00US$8600000) | [data/housing_cost_5_year_lookback.py](https://github.com/nickjwheatley/project_zipliner/blob/main/data/housing_costs_5_year_lookback.py) |
|Median Income (ACS) | [American Census Survey Median Income](https://data.census.gov/table?q=S1903:%20Median%20Income%20in%20the%20Past%2012%20Months%20(in%202022%20Inflation-Adjusted%20Dollars)&g=010XX00US$8600000) | [data/med_income.py](https://github.com/nickjwheatley/project_zipliner/blob/main/data/med_income.py)
|Median Age (ACS) | [American Census Survey Median Age](https://data.census.gov/table?q=B01002&g=010XX00US$8600000) | [data/median_age.py](https://github.com/nickjwheatley/project_zipliner/blob/main/data/median_age.py) |
|County Business Patterns (ACS) | [American Census Survey CBP](https://data.census.gov/table?q=CBP&g=010XX00US$0500000) | [data/operating_businesses.py](https://github.com/nickjwheatley/project_zipliner/blob/main/data/operating_businesses.py) |
|Overall Population (ACS) | [American Census Survey Overall Population](https://data.census.gov/table/ACSDP5Y2022.DP05?q=DP05:%20ACS%20Demographic%20and%20Housing%20Estimates&g=010XX00US$8600000) | [data/overall_population.py](https://github.com/nickjwheatley/project_zipliner/blob/main/data/overall_population.py) |
|Population Change (ACS) | [American Census Survey Population Change](https://data.census.gov/table?q=S0701:%20Geographic%20Mobility%20by%20Selected%20Characteristics%20in%20the%20United%20States&g=010XX00US$8600000,$8600000) | [data/population_change.py](https://github.com/nickjwheatley/project_zipliner/blob/main/data/population_change.py) |

## Supplemental Development Resources
[Plotly Dash User Guide](https://dash.plotly.com/?_gl=1*17kyid5*_gcl_au*MTE2MzIyODE0Ny4xNzIwNzE0NTkx*_ga*MTI3ODQ5Nzc4Mi4xNzIwNzE0NTky*_ga_6G7EE0JNSC*MTcyMjU2NTQ1OS41LjEuMTcyMjU2NTY0OC41Ni4wLjA.)

