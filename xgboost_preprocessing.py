import numpy as np
import pandas as pd
import xgboost as xgb
from datetime import timedelta
from sklearn.preprocessing import StandardScaler

df = pd.read_csv('time_series_data.csv')

scaler = StandardScaler()

FEATURES = ['difference_1', 'difference_2', 'difference_3',
            'difference_4', 'difference_5', 'difference_6',
            'difference_7', 'moving_average_week',
            'moving_average_two_weeks', 'difference_year',
            'day_of_week', 'day_of_year', 'quarter', 'month', 'year']

TARGET = ['sales']


def create_dataframe(list1, list2):
    """
    Create a Pandas DataFrame from two lists.

    Parameters:

        list1 (list): A list of dates.
        list2 (list): A list of sales figures.
    Returns:

        df (Pandas DataFrame): A DataFrame object containing 
        two columns: 'dates' and 'sales', with each column 
        representing the corresponding input list.
    """
    data = {'dates': list1, 'sales': list2}

    df = pd.DataFrame(data)

    return df


def create_sales_features(df):
    """
    Creates new features based on the `sales` column 
    of the given DataFrame.

    Args:
        df (pandas.DataFrame): The DataFrame to create 
        features for.

    Returns:
        pandas.DataFrame: A new DataFrame with the original 
        columns and additional columns for each feature created:
            - difference in sales from 1 to 7 days back.
            - difference in sales from 28 days back.
            - difference in sales from 366 days back.
            - moving average from a 1 week window.
            - moving average from a 2 week window.
    """
    df = df.copy()

    df.loc[df['sales'] > df.sales.mean() + (df.sales.std() * 3),
           'sales'] = df.sales.mean() + (df.sales.std() * 3)

    previous = df.sales.shift(1)
    df['difference_1'] = df.sales - previous

    for i in range(1, 7):
        column = 'difference_' + str(i+1)
        df[column] = df['difference_1'].shift(i)

    df['moving_average_week'] = df.sales.rolling(window=7).mean()

    df['moving_average_two_weeks'] = df.sales.rolling(window=14).mean()

    df['difference_month'] = df.sales - df.sales.shift(28)

    df['difference_year'] = df.sales - df.sales.shift(366)

    df = df.dropna()

    return df


def create_time_features(df):
    """
    Extracts various time-related features from a DataFrame 
    containing time-series data and returns the updated DataFrame.

    Args:
        - df (pandas.DataFrame): The DataFrame containing time-series 
        data to process. This DataFrame must have a 'dates' column 
        with datetime values.

    Returns:
        - pandas.DataFrame: The updated DataFrame with additional 
        time-related features added as columns:
            - day of week.
            - day of year.
            - quarter of the year.
            - month of the year.
            - year.
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


def scale_dataset(df):
    """
    Preprocesses a given DataFrame by scaling its numerical features using a MinMaxScaler, 
    one-hot encoding its categorical features, and concatenating them together with the 
    product ID and target variables.

    Parameters
    ----------
    df : pandas.DataFrame
        The DataFrame to be preprocessed.

    Returns
    -------
    pandas.DataFrame
        The preprocessed DataFrame with scaled numerical features, one-hot encoded 
        categorical features, and concatenated product ID and target variables.
    """

    df = df.copy()

    numerical_features = df[['difference_1', 'difference_2', 'difference_3',
                             'difference_4', 'difference_5', 'difference_6', 'difference_7',
                             'moving_average_week', 'moving_average_two_weeks', 'difference_month',
                             'difference_year']]

    categorical_features = df[['day_of_week', 'day_of_year',
                               'quarter', 'month', 'year']]

    categorical_features = pd.get_dummies(categorical_features,
                                          columns=['day_of_week', 'day_of_year',
                                                   'quarter', 'month', 'year'])

    target = df[['sales']]

    scaler.fit(numerical_features)

    numerical_features = pd.DataFrame(
        scaler.transform(numerical_features),
        columns=numerical_features.columns,
        index=numerical_features.index)

    df = pd.concat([target, numerical_features, categorical_features], axis=1)

    return df


def train_model(df):
    """
    Trains a gradient boosting regression model on the input DataFrame.

    Args:
        df: A pandas DataFrame containing two columns, 'dates' and 'sales'.

    Returns:
        A trained XGBoost model.
    """

    model = xgb.XGBRegressor(n_estimators=1000, booster='gbtree',
                             max_depth=2,
                             learning_rate=0.1)

    train_df = create_sales_features(df)
    train_df = create_time_features(train_df)
    train_df = scale_dataset(train_df)

    x_features = train_df[train_df.columns[1:]]
    y_target = train_df['sales']

    model.fit(x_features, y_target,
              eval_set=[(x_features, y_target)],
              verbose=100)

    return model


def generate_forecast(model, df, ahead):
    """
    Generates a forecast for future sales based 
    on a time-series dataframe.

    Parameters:
    -----------
    df: pandas.DataFrame
        A time-series dataframe with dates as index and sales as a column.
    ahead: int
        The number of future periods to forecast.

    Returns:
    --------
    pandas.DataFrame
        A dataframe with the forecasted sales for the next `ahead` periods.
    """

    df = df.copy()

    df_time = create_time_features(df)
    monthly_sales = pd.DataFrame(df_time.groupby('month')['sales'].mean())

    df = df.set_index('dates')
    df.index = pd.to_datetime(df.index)

    for i in range(ahead):

        future_date = df.index.max() + timedelta(days=1)
        future_dates = pd.date_range(start=future_date.strftime("%Y-%m-%d"),
                                     end=future_date.strftime("%Y-%m-%d"))

        future_df = pd.DataFrame({"sales": None}, index=future_dates)
        future_df['sales'] = monthly_sales.loc[future_df.index.month[0]]['sales']

        df_with_future = pd.concat([df, future_df]).reset_index().rename(
            columns={"index": "dates"})
        df_with_future = create_sales_features(df_with_future)
        df_with_future = create_time_features(df_with_future)
        df_with_future = scale_dataset(df_with_future)

        pred = model.predict(df_with_future.tail(1)[
                             df_with_future.columns[1:]])

        future_df['sales'] = abs(pred)

        df = pd.concat([df, future_df])

    return df.tail(ahead)


def api_call(list1, list2, ahead):
    """
    This function combines the other functions in this 
    module to generate a sales forecast

    Parameters:
    -----------
        `list1` and `list2` are lists of equal length containing 
        dates and sales values respectively.

        `ahead` is the number of time units to forecast ahead, 
        which should be a positive integer.

    Returns
    -----------

    A tuple of three items:
        - a list of forecasted dates in the format "YYYY-MM-DD"
        - a list of forecasted sales values
        - a dictionary of statistical values for the sales data, including mean, minimum,
            maximum, variance, and standard deviation.
    """
    df = create_dataframe(list1, list2)

    statistics = dict(mean=df.sales.mean(),
                      minimum=df.sales.min(),
                      maximum=df.sales.max(),
                      variance=df.sales.var(),
                      std=df.sales.std(),)

    model = train_model(df)

    forecast = generate_forecast(model, df, abs(ahead))

    return list(forecast.index.strftime("%Y-%m-%d")), list(forecast.sales), statistics
