"""
This is a demo for time-series-forecast/app.py written for the Hugging Face Time Series Forecasting.
What does this demo do? 
1. Loads in the data from the given file path.
2. Imputes the data if any columns have at least 5% missing data.
3. Resamples the data to weekly.
4. Adds an inferred frequency to the index.
5. Splits the data into train and test sets.
6. Fits an *ARIMA model to the training data. TO DO: Add more models.
7. Makes predictions on the test data.
By choice it can plot:
1. Plots the predictions.
2. Plots the seasonal component.
3. Plots the autocorrelation.
"""
import gradio as gr
import pandas as pd
import requests
# import sklearn
from io import StringIO
from pathlib import Path
# from typing import Unionimport OSError
from sklearn.model_selection import train_test_split
import statsmodels.api as sm
import matplotlib.pyplot as plt
# import numpy as np

from data.impute_data import impute
# from data.dataset import Dataset, Result

# from predictions.ARIMA import arima
from typing import Tuple, TypeVar
import plotly.express as px
import plotly.graph_objects as go
# from plotly.subplots import make_subplots
from statsmodels.tsa.forecasting.stl import STLForecast
# from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

Model = TypeVar("Model")

selected_city = ""


class Data(pd.DataFrame):
    pass


def __load_data(file_path: str) -> Data:
    """Load in the data from the given file path."""
    df = pd.read_csv(file_path, parse_dates=["Date"], usecols=[
                     'City', 'Date', 'PM2.5'])
    df.Date = pd.to_datetime(df.Date)
    df.set_index("Date", inplace=True)
    return df


def __impute_data_if_needed(data: pd.DataFrame) -> pd.DataFrame:
    """Impute the data if any columns have at least 5% missing data."""
    print("Imputing data")
    data = impute(data, target_columns=["PM2.5"])
    print("Data imputed")
    return data


def __resample(dataframe: Data) -> pd.DataFrame:
    """Resample the data to weekly."""
    return dataframe.resample("w").mean()


def __add_inferred_freq_to_index(dataframe: pd.DataFrame) -> pd.DataFrame:
    """Add an inferred frequency to the index."""
    dataframe.index.freq = dataframe.index.inferred_freq
    return dataframe


def __get_period_of_seasonality(data: pd.DataFrame) -> int:
    """
    Returns the period of seasonality based on the frequency of the data.
    """
    period = data.index.freq
    if isinstance(period, pd.tseries.offsets.YearEnd):
        # For yearly data, set the period as 12 (12 months in a year)
        period = 12
    elif isinstance(period, pd.tseries.offsets.MonthEnd):
        # For monthly data, set the period as 12 (12 months in a year)
        period = 12
    elif isinstance(period, pd.tseries.offsets.Week):
        # For weekly data, set the period as 52 (52 weeks in a year)
        period = 52
    elif isinstance(period, pd.tseries.offsets.Day):
        # For daily data, set the period as 365 (365 days in a year)
        period = 365
    return period


def split_data(df,  target_column, test_size, random_state=None):
    """
    Split a DataFrame into train and test sets.

    Args:
        df (pd.DataFrame): The DataFrame to be split.
        test_size (float): The proportion of the data to be included in the test set (default: 0.2).
        random_state (int or None): Random seed for reproducibility (default: None).

    Returns:
        pd.DataFrame, pd.DataFrame: The train and test sets.
    """
    # Separate the target variable from the features
    X = df.drop(columns=[target_column])
    y = df[target_column]

    # Split the data into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, shuffle=False)

    # Create train and test DataFrames
    train_df = pd.concat([X_train, y_train], axis=1)
    test_df = pd.concat([X_test, y_test], axis=1)

    return train_df, test_df


def string_to_tuple(string: str) -> Tuple[int, int, int]:
    """Convert a string to a tuple."""
    return tuple(map(int, string.split(",")))


def fit_arima_model(train_data: pd.Series, test_data: pd.Series, order: Tuple[int, int, int], trend: str):
    """ fit an ARIMA model to the training data and return the model"""

    # Reshape train_data into 1-dimensional arrays
    train_data_array = train_data.values.ravel()

    # Determine the period of seasonality
    period = __get_period_of_seasonality(train_data)
    print(f"Period of seasonality: {period}")

    # Create the STLForecast model with ARIMA as the base model
    model = sm.tsa.STLForecast(
        endog=train_data_array,
        period=period,
        model=sm.tsa.ARIMA,
        model_kwargs={"order": string_to_tuple(order), "trend": trend},
    )

    # Fit the model to the training data
    model = model.fit()

    return model


def process_dataset(df, test_size: float):
    """Process the dataset. This includes imputing the data, resampling it, adding an inferred frequency to the index and splitting it into train and test sets."""
    global test_size_slider
    test_size_slider = test_size
    df = df[df['City'] == selected_city]
    df = __impute_data_if_needed(df)
    df = __resample(df)
    df = __add_inferred_freq_to_index(df)

    # Split the data into train and test sets
    train_df, test_df = split_data(
        df, test_size=test_size, target_column="PM2.5", random_state=42)
    return train_df, test_df


def load_and_display_dataset(dataset_url: str, csv_file: gr.inputs.File, city: str, test_size)->pd.DataFrame:
    global selected_city
    global test_size_slider
    test_size_slider = test_size
    selected_city = city
    if dataset_url:
        response = requests.get(dataset_url)
        if response.status_code == 200:
            data = response.content.decode("utf-8")
            if data.strip():  # Check if the data is not empty
                df = __load_data(StringIO(data))
                # return process_dataset(df,test_size_slider)
                processed_dataset = process_dataset(df, test_size=test_size)
                return processed_dataset
            else:
                return "The file at the provided URL is empty or contains no columns."
        else:
            print(f"Response status code: {response.status_code}")
            return "Unable to load the dataset from the provided URL."
    elif csv_file is not None:
        if csv_file.name.endswith(".csv"):
            file_content = csv_file.read().decode("utf-8")
            print(f"Type of file_content: {type(file_content)}")
            if len(file_content) > 0:  # Check if the file content is not empty
                with open("temp_file.csv", "w") as temp_file:
                    temp_file.write(file_content)
                test_size = gr.Interface.args["Test Size"]
                df = __load_data("temp_file.csv")
                processed_dataset = process_dataset(df, test_size=test_size)
                return processed_dataset
            else:
                return "The uploaded CSV file is empty or contains no columns."
        else:
            return "Please upload a valid CSV file."
    else:
        return "Please provide a dataset URL or upload a CSV file."


order_values = list(range(0, 5))  # Example values for p, d, q

order_slider = gr.Textbox(
    label="ARIMA Order (p, d, q) (e.g., 1,0,2)",
    placeholder="Enter order values separated by commas (e.g., 1,0,2)"
)

trend_dropdown = gr.Dropdown(
    label="Trend Type",
    choices=["c", "t", "ct"],
    value="t",
)


def calculate_arima_predictions(dataset_url: str, csv_file: gr.inputs.File, city: str, test_size, order, trend)->Tuple[pd.DataFrame, str, pd.DataFrame, pd.DataFrame, pd.Series]:
    global order_slider
    order_slider = order
    global trend_dropdown
    trend_dropdown = trend
    processed_dataset = load_and_display_dataset(
        dataset_url, csv_file, city, test_size)
    model = fit_arima_model(
        processed_dataset[0], processed_dataset[1], order, trend)
    # Unpack the arima_predictions tuple
    prediction_in_sample = model.get_prediction(
        0, len(processed_dataset[0])-1).summary_frame()
    # add date to index in prediction_in_sample
    prediction_in_sample = prediction_in_sample.set_index(
        processed_dataset[0].index)

    prediction = model.forecast(steps=len(processed_dataset[1]))
    # prediction_summary = model.model_result.get_forecast(steps=len(processed_dataset[1])).summary_frame()
    prediction_summary = model.get_prediction(len(processed_dataset[0]), len(
        processed_dataset[0])-1+len(processed_dataset[1])).summary_frame()
    # add date to index in prediction_summary and make a DataFrame
    prediction_summary = pd.DataFrame(
        prediction_summary.set_index(processed_dataset[1].index))

    # Seasonal component then to pd.Series
    residuals = pd.Series(model.model_result.resid)
    # index
    residuals.index = processed_dataset[0].index
    seasonal_component = processed_dataset[0]['PM2.5'] - residuals

    # Generate plot title
    plot_title = f"ARIMA Model (order p,d,q = {order}, trend = {trend})"

    return prediction_summary, plot_title, prediction_in_sample, prediction, seasonal_component,


# List of Indian cities
indian_cities = ['Ahmedabad', 'Aizawl', 'Amaravati', 'Amritsar', 'Bengaluru', 'Bhopal', 'Brajrajnagar', 'Chandigarh',
                 'Chennai', 'Coimbatore', 'Delhi', 'Ernakulam', 'Gurugram', 'Guwahati', 'Hyderabad', 'Jaipur', 'Jorapokhar',
                 'Kochi', 'Kolkata', 'Lucknow', 'Mumbai', 'Patna', 'Shillong', 'Talcher', 'Thiruvananthapuram', 'Visakhapatnam'
                 # Add more cities here
                 ]

city_dropdown = gr.Dropdown(
    choices=indian_cities,
    label="Select an Indian city for which you want to predict the PM2.5 value",
    value=None
)


key_dropdown = gr.Dropdown(
    choices=["ARIMA Model fit", "Seasonal Component Plot", "ACF/PACF Plots"],
    label="Select a plot to display",
    value=None
)

test_size_slider = gr.Slider(
    minimum=0.1,
    maximum=1.0,
    value=0.2,
    step=0.1,
    label="Size of Test Set",
)


def plot_predictions(predictions, title, test_data=None, train_data=None, predictions_in_sample=None, prediction=None)->go.Figure:
    # Create the plot using Plotly
    fig = px.line(predictions, x=predictions.index,
                  y="mean", labels={"mean": "Predictions"})

    # Add prediction in sample
    if predictions_in_sample is not None:
        fig.add_trace(px.line(predictions_in_sample, x=predictions_in_sample.index,
                      y="mean", labels={"mean": "Prediction in sample"}).data[0])

    # Add prediction
    if prediction is not None:
        fig.add_trace(
            px.line(prediction, x=predictions.index, y=prediction).data[0])

    # Add scatter plot for test data
    if test_data is not None:
        if isinstance(test_data, pd.Series):
            test_data_df = pd.DataFrame({"PM2.5": test_data})
        elif isinstance(test_data, pd.DataFrame):
            test_data_df = test_data
        else:
            raise ValueError(
                "test_data should be a pandas Series or DataFrame.")
        fig.add_trace(
            go.Scatter(
                x=test_data_df.index,
                y=test_data_df["PM2.5"],
                mode="markers",
                name="Test Data",
                marker=dict(
                    color="red",
                    symbol="circle",
                    size=5
                )
            )
        )

    # Add scatter plot for train data
    if train_data is not None:
        if isinstance(train_data, pd.Series):
            train_data_df = pd.DataFrame({"PM2.5": train_data})
        elif isinstance(train_data, pd.DataFrame):
            train_data_df = train_data
        else:
            raise ValueError(
                "train_data should be a pandas Series or DataFrame.")
        fig.add_trace(
            go.Scatter(
                x=train_data_df.index,
                y=train_data_df["PM2.5"],
                mode="markers",
                name="Train Data",
                marker=dict(
                    color="blue",
                    symbol="circle",
                    size=5
                )
            )
        )

    fig.update_layout(
        title=title,
        xaxis_title="X-axis",
        yaxis_title="Y-axis",
        legend_title="Legend"
    )
    # Add a range from 0 to the maximum value of the test data or train data
    fig.update_yaxes(
        range=[0, max(test_data_df["PM2.5"].max(), train_data_df["PM2.5"].max())])
    return fig


def plot_seasonal_component(data, title):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=data.index, y=data.seasonal,
                  mode='lines', name='Seasonal Component'))
    fig.update_layout(
        title=title,
        xaxis_title='Date',
        yaxis_title='Seasonal Component',
        legend=dict(orientation='h'),
        template='plotly_white'
    )
    return fig


def plot_autocorrelation(train_data):

    # what's the number of lags we want to plot?
    number_of_lags = __get_period_of_seasonality(train_data)
    # try plotting if there's enough data
    try:

        # plot the autocorrelation and partial autocorrelation plots nicely side by side
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 10))
        plot_acf(train_data, ax=ax1, lags=number_of_lags)
        ax1.set_title('Autocorrelation Function (ACF)')
        plot_pacf(train_data, ax=ax2, lags=number_of_lags, method="ywm")
        ax2.set_title('Partial Autocorrelation Function (PACF)')
    except ValueError:
        raise ValueError(
            "Unable to plot partial autocorrelation. Insufficient data or invalid number of lags.")
    return fig


def generate_outputs(dataset_url, csv_file, city, test_size, order, trend, key):
    global key_dropdown
    key_dropdown = key
    processed_dataset = load_and_display_dataset(
        dataset_url, csv_file, city, test_size)
    prediction_table, plot_title, prediction_in_sample, prediction, seasonal_component = calculate_arima_predictions(
        dataset_url, csv_file, city, test_size, order, trend)
    # Create a plot for the predictions
    plot1 = plot_predictions(prediction_table, plot_title,
                             test_data=processed_dataset[1], train_data=processed_dataset[0], predictions_in_sample=prediction_in_sample, prediction=prediction)

    # Show seasonal component
    seasonal_component = seasonal_component
    print(f"seasonal_component: {seasonal_component}")
    seasonal_component.index = processed_dataset[0].index

    # Convert the seasonal component to a DataFrame
    seasonal_component_df = pd.DataFrame(
        seasonal_component, columns=['seasonal'])
    print(f"seasonal_component_df: {seasonal_component_df}")

    # Create a plot for the seasonal component
    plot2 = plot_seasonal_component(
        seasonal_component_df, 'Seasonal Component Plot')

    # Create a plot for the autocorrelation
    try:
        plot3 = plot_autocorrelation(processed_dataset[0])
    except ValueError:
        plot3 = "Insufficient data to plot ACF/PACF"

    # Return the plots
    if key == "ARIMA Model fit":
        return plot1
    elif key == "Seasonal Component Plot":
        return plot2
    elif key == "ACF/PACF Plots":
        if isinstance(plot3, str):
            return print("Insufficient data to plot autocorrelation")
        else:
            return plot3


outputs = gr.Plot()

iface = gr.Interface(
    fn=generate_outputs,
    inputs=[
        gr.Textbox(placeholder="Enter dataset URL",
                value="https://huggingface.co/datasets/kkarukes/city_day/raw/main/city_day.csv"),
        gr.File(label="Upload a CSV file"),
        city_dropdown,
        test_size_slider,
        order_slider,
        trend_dropdown,
        key_dropdown,
    ],
    outputs=[
        outputs,
    ],
    # title="ARIMA Prediction",
    description="Forecast time series data using ARIMA model. The model is fitted using the training data and the predictions are made on the test data. p,d,q are the parameters of the ARIMA model. p,d,q can be inputted manually within the range of 0-5. The trend parameter can be either 'c' for constant, 't' for linear or 'ct' for constant with linear trend."
)


iface.launch(share=False)
