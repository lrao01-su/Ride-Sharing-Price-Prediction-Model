import streamlit as st                  # pip install streamlit
from helper_functions import  display_missingValue
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from pandas.plotting import scatter_matrix
import os
import tarfile
import urllib.request
from itertools import combinations

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from sklearn.linear_model import LinearRegression

#############################################

st.markdown("# Practical Applications of Machine Learning (PAML)")

#############################################

st.markdown("### Farelytics ML: Using Machine Learning to Analyze and Compare Uber and Lyft Pricing Models")

#############################################

st.markdown('# Explore & Preprocess Dataset')

#############################################
col1, col2 = st.columns(2)
# with(col1):
with(col1):
    if 'df_cab' in st.session_state:
        df_cab = st.session_state['cab_data']
    else:
        cab_data = st.file_uploader("Upload Your df_cab Dataset", type=['csv','txt'])
        if (cab_data):
            df_cab = pd.read_csv(cab_data)
            st.session_state['cab_data'] = df_cab
# with(col2): #upload from cloud
with(col2):
    if 'df_weather' in st.session_state:
        df_weather = st.session_state['weather_data']
    else:
        weather_data = st.file_uploader("Upload Your df_weather Dataset", type=['csv','txt'])
        if (weather_data):
            df_weather = pd.read_csv(weather_data)
            st.session_state['weather_data'] = df_weather

if cab_data and weather_data is not None:
    #display df_cab and df_weather dataframe
    st.write('You have successfully uploaded your dataset.')
    st.write('Continue to Explore and Preprocess Data')
    #show df_cab and df_weather dataframes
    st.markdown('### df_cab Dataframe')
    st.dataframe(df_cab)
    st.markdown('### df_weather Dataframe')
    st.dataframe(df_weather)
   
   
    # Inspect the dataset
    st.markdown('### Inspect and visualize some interesting features')
    #display missing data for df_cab and df_weather/ Olga
    st.markdown('### Missing Data') 
    missing_data = display_missingValue(df_cab, df_weather)
    st.dataframe(missing_data)
    # Deal with missing values for cab /Olga
    st.markdown('### Handle missing values for cab')

    # Deal with missing values for weather /Mary
    st.markdown('### Handle missing values for weather') 

    #merge df_cab and df_weather /Mary
    st.markdown('### Merge cab and weather data')

    # Handle Text and Categorical Attributes
    st.markdown('### Handling Non-numerical Features')



    st.markdown('### You have preprocessed the dataset.')
    #st.dataframe(df)

    st.write('Continue to Train Model')
