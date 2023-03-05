import numpy as np
import pandas as pd
import xgboost as xgb
from typing import List
from pydantic import BaseModel
from datetime import timedelta
from xgboost import XGBRegressor
from fastapi import FastAPI, HTTPException

app = FastAPI()

"""
To run this script, you will need to install
`fastapi` and `uvicorn`. All other libraries
should be part of the requirements described
by the `requirements.txt` from this repository.
"""

FEATURES = ['difference_1', 'difference_2', 'difference_3',
            'difference_4', 'difference_5', 'difference_6', 'difference_7',
            'day_of_week', 'day_of_year', 'quarter', 'month', 'year']

TARGET = ['sales']


def create_dataframe(list1, list2):
    """
    Creates a dataframe form two arrays.
    It alsos strips outliers off.
    """
    data = {'dates': list1, 'sales': list2}
    df = pd.DataFrame(data)
    df.loc[df['sales'] > df.sales.mean() + (df.sales.std() * 3),
           'sales'] = df.sales.mean() + (df.sales.std() * 3)
    return df


def create_sales_features(df):
    """
    Creates a copy of the input df, and
    clalculates the difference in sales
    up to 7 days back
    """
    df = df.copy()

    previous = df.sales.shift(1)
    df['difference_1'] = df.sales - previous

    for i in range(1, 7):
        column = 'difference_' + str(i+1)
        df[column] = df['difference_1'].shift(i)

    df = df.dropna()

    return df


def create_time_features(df):
    """
    Creates a copy of the input df, turns
    the index (should come as dates) into a
    `datetime format`, and gives you back
    a DataFrame with all time features.
    """

    df = df.copy()

    df = df.set_index('dates')

    df.index = pd.to_datetime(df.index)

    df['day_of_week'] = df.index.day_of_week
    df['day_of_year'] = df.index.day_of_year
    df['quarter'] = df.index.quarter
    df['month'] = df.index.month
    df['year'] = df.index.year

    df = df.sort_index()

    return df


def train_model(df):
    """
    Trains an XGBoost Regression Model.
    """

    model = XGBRegressor(n_estimators=400, booster='gbtree',
                         learning_rate=0.05)

    train_df = create_sales_features(df)
    train_df = create_time_features(train_df)

    x_features = train_df[FEATURES]
    y_target = train_df[TARGET]

    model.fit(x_features, y_target,
              eval_set=[(x_features, y_target)],
              verbose=100)
    return model


def generate_forecast(model, df, ahead):
    """
    This functions call the original dataframe, sets
    the `dates` columns as the index and turns the
    index into `datetime`. After, it loops for a range
    equal to `ahead`. For each iteration. It creates the
    features dataframe with one additional day (a future day),
    and uses the lag features to predict the value of this day.
    In the end, we append this day on the bottom of the seed 
    `df` and repeat. The function returns only the future
    predictions.
    """

    df = df.set_index('dates')
    df.index = pd.to_datetime(df.index)

    for i in range(ahead):

        future_date = df.index.max() + timedelta(days=1)

        future_dates = pd.date_range(start=future_date.strftime("%Y-%m-%d"),
                                     end=future_date.strftime("%Y-%m-%d"))

        future_df = pd.DataFrame(index=future_dates)
        present_df = df[['sales']].copy()

        df_with_future = pd.concat([present_df, future_df]).reset_index().rename(
            columns={"index": "dates"}).fillna(0)
        df_with_future = create_sales_features(df_with_future)
        df_with_future = create_time_features(df_with_future)

        pred = model.predict(df_with_future.tail(1)[FEATURES])

        future_df['sales'] = abs(np.round(pred))

        df = pd.concat([present_df, future_df])

    return df.tail(ahead)


def api_call(list1, list2, ahead):
    """
    This fuction ties all othr fuctions together and
    generates two lists ([dates], [sales]).
    """
    df = create_dataframe(list1, list2)
    model = train_model(df)
    forecast = generate_forecast(model, df, abs(ahead))
    return list(forecast.index.strftime("%Y-%m-%d")), list(forecast.sales)


class InputData(BaseModel):
    dates: List[str]
    sales: List[float]
    ahead: int


@app.get("/")
async def root():
    return 'Welcome to the Teeny-Tiny API :)'


@app.post("/predict")
async def predict(data: InputData):
    days, sales = api_call(data.dates, data.sales, data.ahead)
    return {"dates": days,
            "sales": sales}
