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
    #show missing data for df_cab and df_weather in two dataframes
    st.markdown('### df_cab Missing Values')
    st.dataframe(df_cab.isnull().sum())
    st.markdown('### df_weather Missing Values')
    st.dataframe(df_weather.isnull().sum())
    #show missing data for df_cab and df_weather in two bar charts
    st.markdown('### df_cab Missing Values')
    st.bar_chart(df_cab.isnull().sum())
    st.markdown('### df_weather Missing Values')
    st.bar_chart(df_weather.isnull().sum())
 


    

