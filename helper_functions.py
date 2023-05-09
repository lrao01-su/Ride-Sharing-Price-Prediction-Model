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

    if data_list is not None:
        st.session_state['uploaded_files'] = data_list
    return data_list
