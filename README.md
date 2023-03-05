# Teeny-Tiny API 🏰

This is an example of an ML model (`XGBRegressor`), with a particular pre-processing technique and configuration, that can be used for regression/forecasting tasks, served as an API.

## How to send `requests`

```python

import requests
import pandas as pd

# The data you are going to use
df = pd.read_csv('time_series_data.csv')

# The API endpoint URL
url = "https://teeny-tiny-api.onrender.com/predict"

# Define the input data
data = {
    "product": df.product_id[0], # name of the product
    "dates": list(df.dates), # the list of dates
    "sales": list(df.sales), # the list of sales
    "ahead": 15 # how many days ahead we want to look
}

# Send a POST request with the input data
response = requests.post(url, json=data)

# Done!
print(response.json())

```
