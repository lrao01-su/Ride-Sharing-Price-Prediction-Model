import numpy as np                      # pip install numpy
import pandas as pd                     # pip install pandas
import streamlit as st                  # pip install streamlit
from sklearn.metrics import recall_score, precision_score, accuracy_score

# All pages

#Page A 
def display_missingValue(df_cab, df_weather):
    """
    This function displays the missing value of the datasets

    Input:
        - two df_cab, df_weather
    Output:
        - missing values of the datasets
    """
    #displaying the missing values of the datasets
    st.markdown('### Missing Values')
    st.write('df_cab')
    st.write(df_cab.isnull().sum())
    st.write('df_weather')
    st.write(df_weather.isnull().sum())
    return df_cab.isnull().sum(), df_weather.isnull().sum()

def remove_outliers(df, feature):
    """
    This function removes the outliers of the given feature(s)

    Input: 
        - df: pandas dataframe
        - feature: the feature(s) to remove outliers
    Output: 
        - dataset: the updated data that has outliers removed
        - lower_bound: the lower 25th percentile of the data
        - upper_bound: the upper 25th percentile of the data
    """
    dataset, lower_bound, upper_bound = None, -1, -1

    # Add code here

    st.write('remove_outliers not implemented yet.')
    return dataset, lower_bound, upper_bound









