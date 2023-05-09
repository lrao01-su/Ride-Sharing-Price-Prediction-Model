import streamlit as st                  # pip install streamlit
from helper_functions import fetch_dataset

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

    # Deal with missing values
    st.markdown('### Handle missing values')

    # Handle Text and Categorical Attributes
    st.markdown('### Handling Non-numerical Features')

    # Some feature selections/engineerings here
    st.markdown('### Remove Irrelevant/Useless Features')

    # Remove outliers
    st.markdown('### Remove outliers')

    # Normalize your data if needed
    st.markdown('### Normalize data')

    st.markdown('### You have preprocessed the dataset.')
    st.dataframe(df)

    st.write('Continue to Train Model')
