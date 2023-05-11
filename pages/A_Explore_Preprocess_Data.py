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
st.sidebar.image: st.sidebar.image("https://www.alistdaily.com/wp-content/uploads/2018/11/UberLyft_Hero_111518-1024x576.jpg", use_column_width=True)

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
    #display missing data for df_cab and df_weather
    missing_data = display_missingValue(df_cab, df_weather)
    st.dataframe(missing_data)

    #display df_cab and df_weather dataframes
    st.markdown('### df_cab Dataframe')
    st.dataframe(df_cab)
    st.markdown('### df_weather Dataframe')
    st.dataframe(df_weather)

    #remove missing data in df_cab
    st.markdown('### Remove Missing df_cab Data')
    #create a button to run in streamlit for remove missing data in df_cab
    if st.button('Remove Missing df_cab Data'):
        df_cab = df_cab.dropna()
        st.write(df_cab)
        #write df_cab has removed missing data
        st.write('df_cab has removed missing data')

    # Deal with missing values for df_weather 
    #fill missing data in df_weather
    st.markdown('### Fill Missing df_weather Data')
    #create a button to run in streamlit for fill missing data in df_weather with 0
    if st.button('Fill Missing df_weather Data'):
        df_weather = df_weather.fillna(0)
        st.write(df_weather)
        #write df_weather has filled missing data
        st.write('df_weather has filled missing data')

    
    #drop time_stamp column in df_weather
    st.markdown('### Drop df_weather Time_stamp Column')
    df_weather = df_weather.drop(['time_stamp'], axis=1)
    st.write(df_weather)
    # group weather data by average based on the same location
    st.markdown('### Group df_weather Data by average based on the same location')
    df_weather_avg = df_weather.groupby(['location']).mean().reset_index()
    st.write(df_weather_avg)

    #create source_weather, destination_weather
    source_weather_df= df_weather_avg.rename(columns={
    'location':'source',
    'temp':'source_temp',
    'clouds': 'source_clouds',
    'pressure':'source_pressure',
    'rain': 'source_rain',
    'humidity':'source_humidity',
    'wind':'source_wind'})
    source_weather_df
    destination_weather_df= df_weather_avg.rename(columns={
    'location':'destination',
    'temp':'destination_temp',
    'clouds': 'destination_clouds',
    'pressure':'destination_pressure',
    'rain': 'destination_rain',
    'humidity':'destination_humidity',
    'wind':'destination_wind'})
    destination_weather_df

    #merge df_cab and df_weather
    st.markdown('### Merge cab and weather data')
    df_cab.merge(source_weather_df,on='source')
    df = df_cab\
    .merge(source_weather_df, on ='source')\
    .merge(destination_weather_df, on='destination')
    df



    # Handle Text and Categorical Attributes
    st.markdown('### Handling Non-numerical Features')

    st.markdown('### You have preprocessed the dataset.')
    #st.dataframe(df)

    st.write('Continue to Train Model')
