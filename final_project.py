import streamlit as st              # pip install streamlit

st.markdown("# Practical Applications of Machine Learning (PAML)")

#############################################
st.sidebar.image: st.sidebar.image("https://www.alistdaily.com/wp-content/uploads/2018/11/UberLyft_Hero_111518-1024x576.jpg", use_column_width=True)

#############################################

st.markdown("## Farelytics ML: Using Machine Learning to Analyze and Compare Uber and Lyft Pricing Models")

#############################################

st.markdown("### Olga Acuña Leanos (oea9), Linjing Rao (lr534) and Meet Oza (mgo26)")

#############################################

st.markdown("""
### Problem
- Uber and Lyft are two of the most popular ride-sharing services in the world.
- Both companies have a similar business model, but they have different pricing models.

The problem we aim to address is the current inability to predict and compare the pricing models of Uber and Lyft with a high level of accuracy. This limitation hinders consumers' ability to make informed choices and ride-sharing companies' capacity to optimize their pricing strategies effectively. The lack of robust insights into the dynamic pricing dynamics also restricts the industry's ability to understand market fluctuations, resulting in potential challenges for consumer decision-making, company competitiveness, and the overall advancement of the ride-sharing sector.

### Our Goal
Build a machine learning model using techniques such as:
- Linear Regression
- Lasso Regression
- Random Forest
- Gradient Boosting

While also incorporating a comprehensive set of variables including:
- Distance
- Weather
- Ride Type (Uber Black, Lux, premier, X, XL
- Time of Day
- Demand
- Company specific factors)

In an effort to:
- Uncover the significant factors that influence pricing and provide a holistic understanding of Uber and Lyft's pricing strategies
- Empower consumers to make informed decisions about their ride-sharing choices
"""
)


#############################################

st.markdown("Click **Explore Preprocess Data** to get started.")
