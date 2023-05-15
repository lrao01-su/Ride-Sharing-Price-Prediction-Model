import streamlit as st              # pip install streamlit

#write section title
st.markdown("# Project Summary")
st.markdown("### Farelytics ML: Using Machine Learning to Analyze and Compare Uber and Lyft Pricing Models")

#############################################
st.write("## Project Overview")
st.markdown("- Four different regression models were evaluated: Linear Regression, Lasso Regression, Random Forest, and Gradient Boosting.")
st.markdown("- The evaluation metrics used were mean absolute error, root mean squared error, and R2 score.")
st.markdown("- The best performing model on the training set was Random Forest with an MAE of 7.0405, RMSE of 8.6838, and R2 score of 0.1291.")
st.markdown("- The best performing model on the test set was Random Forest with an MAE of 7.0405, RMSE of 8.6838, and R2 score of 0.1291.")
st.markdown("- The most important features were distance, time of day, and weather.")

#############################################

#############################################


