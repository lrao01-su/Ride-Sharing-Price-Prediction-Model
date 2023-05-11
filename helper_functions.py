import numpy as np                      # pip install numpy
import pandas as pd                     # pip install pandas
import streamlit as st                  # pip install streamlit
from sklearn.metrics import recall_score, precision_score, accuracy_score

# All pages


def fetch_dataset():
    """
    This function renders the file uploader that fetches the dataset either from local machine

    Input:
        - page: the string represents which page the uploader will be rendered on
    Output: None
    """
    # Check stored data
    data_list = None
    uploaded_files = None
    if 'uploaded_files' in st.session_state:
        data_list = st.session_state['uploaded_files']
    else:
        uploaded_files = st.file_uploader(
            'Upload a Dataset', type=['csv', 'txt'], accept_multiple_files=True)

        if (uploaded_files):
            #df = pd.read_csv(data)
            data_list = []
            for f in uploaded_files:
                uploaded_files = pd.read_csv(f)
                data_list.append(uploaded_files)
                #store as df_cab and df_weather
                df_cab = uploaded_files[0]
                df_weather = uploaded_files[1]
    if data_list is not None:
        st.session_state['uploaded_files'] = data_list
    
    return df_cab, df_weather

#Page A 
def display_missingValue():
    """
    This function displays the missing value of the datasets

    Input:
        - two df_cab, df_weather
    Output:
        - missing values of the datasets
    """
    df_rides = pd.read_csv('/datasets/cab_rides.csv')
    df_weather = pd.read_csv('/datasets/weather.csv')

    #dropping missing values from columns and rows in dataset

    df_rides=df_rides.dropna(axis=0).reset_index(drop=True)
    df_weather=df_weather.dropna(axis=0).reset_index(drop=True)

    return df_rides, df_weather



