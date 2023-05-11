import streamlit as st
#############################################

st.markdown("# Practical Applications of Machine Learning (PAML)")

#############################################

st.markdown("### Farelytics ML: Using Machine Learning to Analyze and Compare Uber and Lyft Pricing Models")

#############################################

st.title('Deploy Application')

#############################################

df = None
if 'data' in st.session_state:
    df = st.session_state['data']
else:
    st.write(
        '### The <project> Application is under construction. Coming to you soon.')

# Deploy App
if df is not None:
    st.markdown('### <Deployment app name>')

    st.markdown('#### Some descriptions about the deployment app')
