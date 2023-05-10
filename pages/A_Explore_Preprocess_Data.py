import streamlit as st                  # pip install streamlit
from helper_functions import fetch_dataset, display_missingValue
import streamlit as st
import pandas as pd
import numpy as np
#############################################

st.markdown("# Practical Applications of Machine Learning (PAML)")

#############################################

st.markdown("### Farelytics ML: Using Machine Learning to Analyze and Compare Uber and Lyft Pricing Models")

#############################################

st.markdown('# Explore & Preprocess Dataset')

#############################################
df = None
df = fetch_dataset()




if df is not None:
    # Display original dataframe
    st.markdown('View initial data with missing values or invalid inputs')
    st.markdown('You have uploaded the dataset.')
    #write multiple dataframes
    for i in range(len(df)):
        st.dataframe(df[i])

    # Inspect the dataset
    st.markdown('### Inspect and visualize some interesting features')

    #display missing data for df_cab and df_weather/ Olga
    st.markdown('### Missing Data') 
    missing_data = display_missingValue(df_rides, df_weather)

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
    st.dataframe(df)

    st.write('Continue to Train Model')
