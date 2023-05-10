import numpy as np                      # pip install numpy
import pandas as pd                     # pip install pandas
import streamlit as st                  # pip install streamlit
from sklearn.metrics import recall_score, precision_score, accuracy_score

# All pages


def load_dataset(data):
    """
    Input: data is the filename or path to file (string)
    Output: pandas dataframe df
    - Checkpoint 1 - Read .csv file containing a dataset
    """
    #error checking - check for string
    df = pd.read_csv(data)
    return df


#Page A 
def display_missingValue(df_cab, df_weather):
    """
    This function displays the missing value of the datasets

    Input:
        - two df_cab, df_weather
    Output:
        - missing values of the datasets
    """
#code here
    st.write("Missing values in df_cab")
    st.write(df_cab.isnull().sum())
    st.write("Missing values in df_weather")
    st.write(df_weather.isnull().sum())



