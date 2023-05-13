import numpy as np                      # pip install numpy
import pandas as pd                     # pip install pandas
import streamlit as st                  # pip install streamlit
from sklearn.metrics import recall_score, precision_score, accuracy_score
from sklearn.linear_model import Lasso
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import GridSearchCV

# All pages

#Page A 
def display_missingValue(df_cab, df_weather):
    """
    This function displays the missing value of the datasets

    Input:
        - two df_cab, df_weather
    Output:
        - missing values of the datasets
    """
    #displaying the missing values of the datasets
    st.markdown('### Missing Values')
    st.write('df_cab')
    st.write(df_cab.isnull().sum())
    st.write('df_weather')
    st.write(df_weather.isnull().sum())
    return df_cab.isnull().sum(), df_weather.isnull().sum()

def remove_outliers(df, feature):
    """
    This function removes the outliers of the given feature(s)

    Input: 
        - df: pandas dataframe
        - feature: the feature(s) to remove outliers
    Output: 
        - dataset: the updated data that has outliers removed
        - lower_bound: the lower 25th percentile of the data
        - upper_bound: the upper 25th percentile of the data
    """
    df = df.dropna()
    #dataset, lower_bound, upper_bound = None, -1, -1
    # Add code here

    Q1 = np.percentile(df[feature], 25)
    Q3 = np.percentile(df[feature], 75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    dataset = df[(df[feature] > lower_bound) & (df[feature] < upper_bound)]
    #st.write('remove_outliers not implemented yet.')
    return dataset, lower_bound, upper_bound



#Page B

def lasso_reg(x, y, cv, alpha_range):
    model = Lasso()
    param_grid = {'alpha': alpha_range}
    grid_search = GridSearchCV(model, param_grid=param_grid, cv=cv, scoring='r2')
    grid_search.fit(x, y)
    st.session_state['Lasso Regression'] = grid_search.best_estimator_
    return grid_search.best_estimator_


def linear_reg(x, y, cv):
    model = LinearRegression()
    param_grid = {}
    grid_search = GridSearchCV(model, param_grid=param_grid, cv=cv, scoring='r2')
    grid_search.fit(x, y)
    st.session_state['Linear Regression'] = grid_search.best_estimator_
    return grid_search.best_estimator_


def random_forest(x, y, cv, n_estimators_range, max_depth_range):
    model = RandomForestRegressor()
    param_grid = {'n_estimators': n_estimators_range, 'max_depth': max_depth_range}
    grid_search = GridSearchCV(model, param_grid=param_grid, cv=cv, scoring='r2')
    grid_search.fit(x, y)
    st.session_state['Random Forest'] = grid_search.best_estimator_
    return grid_search.best_estimator_


def gradient_boost(x, y, cv, n_estimators_range, learning_rate_range):
    model = GradientBoostingRegressor()
    param_grid = {'n_estimators': n_estimators_range, 'learning_rate': learning_rate_range}
    grid_search = GridSearchCV(model, param_grid=param_grid, cv=cv, scoring='r2')
    grid_search.fit(x, y)
    st.session_state['Gradient Boosting'] = grid_search.best_estimator_
    return grid_search.best_estimator_


def one_hot_encode_feature(df, feature):
    """
    This function performs one-hot-encoding on the given features using pd.get_dummies

    Input: 
        - df: the pandas dataframe
        - feature: the feature(s) to perform one-hot-encoding
    Output: 
        - df: dataframe with one-hot-encoded feature
    """    
    # Add code here
    df = pd.get_dummies(df, columns=[feature])

    
    st.write('Feature {} has been one-hot encoded.'.format(feature))
    return df








