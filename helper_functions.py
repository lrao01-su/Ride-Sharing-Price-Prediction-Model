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
#code here
    #show missing data for df_cab and df_weather
    missing_data = df_cab.isnull().sum()
    missing_data = missing_data.reset_index()
    missing_data.columns = ['column_name', 'missing_count']
    missing_data['filling_factor'] = (df_cab.shape[0]
                                        - missing_data['missing_count']) / df_cab.shape[0] * 100
    missing_data = missing_data.sort_values('filling_factor').reset_index(drop=True)
    return missing_data


