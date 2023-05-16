# Practical Applications in Machine Learning Final Project Template

# Predicting Product Review Sentiment Using Classification

The problem: Predicting and comparing the pricing models of Uber and Lyft
Importance: Ride-sharing services have become popular and their pricing models can vary greatly depending on various factors
Goal: Develop a machine learning model to predict and compare the pricing models of Uber and Lyft
Benefit: Help consumers make more informed decisions and help ride sharing companies optimize their pricing strategy, also it can help understand these two companies and their pricing model better.

Machine Learning Pipelins:
Data exploration: 
Exploring Uber and Lyft Datasets to identify the variables that are most relevant to the pricing models
Analyze the distribution and correlation of the variables to ensure their suitability for the models
Preprocessing: 
Cleaning, transforming and scaling the variables 
Handle missing values with mean values of that column
Merging weather data with cab ride data by timestamp
Outlier Removal: We have removed the outliers based on the IQR
One-hot Integers: Since we have categorical columns, we will use one-hot to convert them
Model Training and Evaluation:
Linear regression
Lasso ridge
Random forest
Gradient boosting
Deployment:
Deploy trained models in an application that can predict and compare the pricing models of Uber and Lyft based on user inputs

Run the application:
```
streamlit run final_project.py
```
