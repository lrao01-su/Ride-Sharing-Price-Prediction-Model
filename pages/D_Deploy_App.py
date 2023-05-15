import streamlit as st
from helper_functions import one_hot_encode_feature, ordinal
import pandas as pd
#############################################

st.markdown("# Practical Applications of Machine Learning (PAML)")

#############################################

st.markdown("### Farelytics ML: Using Machine Learning to Analyze and Compare Uber and Lyft Pricing Models")

#############################################

st.title('Deploy Application')

#############################################


def deploy_model(df):
    model = st.session_state['deploy_model']
    price = model.predict(df)
    return price



df = None
if 'data' in st.session_state:
    df = st.session_state['data']


# Deploy App
if df is not None:
    st.markdown('### Introducing the ML Powered Uber and Lyft Price Prediction')

    st.markdown('## Select Cab Provider')

    provider_options = ['Lyft', 'Uber']

    provider_select = st.selectbox(
        label='Select the cab provider',
        options=provider_options,
        key='provider',
        index=1
        ) 

    st.markdown("## Select the source location")

    source_options = ['Haymarket Square',
    'Back Bay',
     'North End',
     'North Station',
     'Beacon Hill',
     'Boston University',
     'Fenway',
     'South Station',
     'Theatre District',
     'West End',
     'Financial District',
     'Northeastern University']

    source_select = st.selectbox(
        label='Select the source location',
        options=source_options,
        key='source',
        index=8
        ) 



    st.markdown("## Approximate distance of the trip")

    distance = st.slider('Approximate distance of the trip', 0.5, 10.0, 0.2)

    st.markdown("## Select the Day of the Week")

    day_options = ['Sunday', 'Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday']

    day_select = st.selectbox(
        label='Select the Day of the Week',
        options=day_options,
        key='day',
        index=6
        ) 

    st.markdown("## Select the Time of the Day")

    time_options = ['Night', 'Afternoon', 'Morning', 'Evening']

    time_select = st.selectbox(
        label='Select the Time of the Day',
        options=time_options,
        key='time',
        index=3
        ) 
    
    st.markdown("## Select the cab type")

    cab_options = ['Shared',
     'Lux',
     'Lyft',
     'Lux Black XL',
     'Lyft XL',
     'Lux Black',
     'UberXL',
     'Black',
     'UberX',
     'WAV',
     'Black SUV',
     'UberPool',
     'Taxi']


    cab_select = st.selectbox(
        label='Select the cab type',
        options=cab_options,
        key='cab',
        index=8
        ) 
    
    user_input = {}
    user_input['distance'] = distance
    user_input['name'] = cab_select
    user_input['source'] = source_select
    #user_input['destination'] = destination_select
    user_input['cab_type'] = provider_select

    user_input['time_of_day'] = time_select
    user_input['weekday'] = day_select

    selected_features_df = pd.DataFrame.from_dict(user_input, orient='index').T

    df = ordinal(selected_features_df, ['source', 'name','cab_type', 'time_of_day', 'weekday'])

    if st.button('Predict the Price'):
        cab_price = deploy_model(df)
        


        cab_price = str(cab_price[0])

        st.markdown("Price for {} for the entered inputs is:".format(cab_select))

        st.markdown(cab_price)
