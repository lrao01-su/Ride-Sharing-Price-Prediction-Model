import numpy as np                      # pip install numpy
import pandas as pd                     # pip install pandas
import streamlit as st                  # pip install streamlit
from sklearn.metrics import recall_score, precision_score, accuracy_score
from sklearn.linear_model import Lasso
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from plotly.subplots import make_subplots
import plotly.graph_objects as go
from sklearn.preprocessing import OrdinalEncoder


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



def ordinal(df, features):

    enc = OrdinalEncoder()
    


    X_enc = enc.fit_transform(df[features])
    df[features] = pd.DataFrame(X_enc, columns=[features])

    return df



def one_hot_encode_feature(df, features):
    """
    This function performs one-hot-encoding on the given features using pd.get_dummies

    Input: 
        - df: the pandas dataframe
        - feature: the feature(s) to perform one-hot-encoding
    Output: 
        - df: dataframe with one-hot-encoded feature
    """    
    # Add code here
    for feature in features:
        st.write(feature)
        df = pd.get_dummies(df, columns=[feature])
        st.write('Feature {} has been one-hot encoded.'.format(feature))

    
    return df

def split_dataset(X, y, number,random_state=45):
    """
    This function splits the dataset into the train data and the test data

    Input: 
        - X: training features
        - y: training targets
        - number: the ratio of test samples
    Output: 
        - X_train: training features
        - X_val: test/validation features
        - y_train: training targets
        - y_val: test/validation targets
    """
    X_train = []
    X_val = []
    y_train = []
    y_val = []
    

    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=number/100, random_state=random_state)
        

    train_percentage = (len(X_train) /
                            (len(X_train)+len(X_val)))*100

    test_percentage = ((len(X_val))/
                           (len(X_train)+len(X_val)))*100


    # Print dataset split result
    st.markdown('The training dataset contains {0:.2f} observations ({1:.2f}%) and the test dataset contains {2:.2f} observations ({3:.2f}%).'.format(len(X_train),
                                                                                                                                                          train_percentage,
                                                                                                                                                          len(X_val),
                                                                                                                                                          test_percentage))
    # Save state of train and test splits in st.session_state
    st.session_state['X_train'] = X_train
    st.session_state['X_val'] = X_val
    st.session_state['y_train'] = y_train
    st.session_state['y_val'] = y_val

    return X_train, X_val, y_train, y_val

#Page C

def restore_data_splits(df):
    """
    This function restores the training and validation/test datasets from the training page using st.session_state
                Note: if the datasets do not exist, re-split using the input df

    Input: 
        - df: the pandas dataframe
    Output: 
        - X_train: the training features
        - X_val: the validation/test features
        - y_train: the training targets
        - y_val: the validation/test targets
    """
    X_train = None
    y_train = None
    X_val = None
    y_val = None
    # Restore train/test dataset
    if ('X_train' in st.session_state):
        X_train = st.session_state['X_train']
        y_train = st.session_state['y_train']
        st.write('Restored train data ...')
    if ('X_val' in st.session_state):
        X_val = st.session_state['X_val']
        y_val = st.session_state['y_val']
        st.write('Restored test data ...')
    if (X_train is None):
        # Select variable to explore
        numeric_columns = list(df.select_dtypes(include='number').columns)
        feature_select = st.selectbox(
            label='Select variable to predict',
            options=numeric_columns,
        )
        X = df.loc[:, ~df.columns.isin([feature_select])]
        Y = df.loc[:, df.columns.isin([feature_select])]

        # Split train/test
        st.markdown(
            '### Enter the percentage of test data to use for training the model')
        number = st.number_input(
            label='Enter size of test set (X%)', min_value=0, max_value=100, value=30, step=1)

        X_train, X_val, y_train, y_val = split_dataset(X, Y, number, feature_select, 'TF-IDF')
        st.write('Restored training and test data ...')
    return X_train, X_val, y_train, y_val

def mae(y_true, y_pred):
    """
    Measures the absolute difference between predicted and actual values

    Input:
        - y_true: true targets
        - y_pred: predicted targets
    Output:
        - mean absolute error
    """
    #mae_score=-1
    # Add code here
    mae_score = mean_absolute_error(y_true, y_pred)
    #st.write('rmse not implemented yet.')
    return mae_score

def rmse(y_true, y_pred):
    """
    This function computes the root mean squared error. 
    Measures the difference between predicted and 
    actual values using Euclidean distance.

    Input:
        - y_true: true targets
        - y_pred: predicted targets
    Output:
        - root mean squared error
    """
    #rmse_score=-1
    rmse_score = np.sqrt(mean_squared_error(y_true, y_pred))
    # Add code here
   #st.write('rmse not implemented yet.')
    return rmse_score

def r2(y_true, y_pred):
    """
    Compute Coefficient of determination (R2 score). 
    Rrepresents proportion of variance in predicted values 
    that can be explained by the input features.

    Input:
        - y_true: true targets
        - y_pred: predicted targets
    Output:
        - r2 score
    """
    #r2_score=-1  
    # Add code here
    r2_scor = r2_score(y_true, y_pred)
    #st.write('r2 not implemented yet.')
    return r2_scor

def compute_eval_metrics(X, y_true, model, metrics):
    """
    This function checks the metrics of the models

    Input:
        - X: pandas dataframe with training features
        - y_true: pandas dataframe with true targets
        - model: the model to evaluate
        - metrics: the metrics to evlauate performance 
    Output:
        - metric_dict: a dictionary contains the computed metrics of the selected model, with the following structure:
            - {metric1: value1, metric2: value2, ...}
    """
    metric_dict = {}
    # Add code here
    y_pred = model.predict(X)
    
    if 'mean_absolute_error' in metrics:
        mae = mean_absolute_error(y_true, y_pred)
        metric_dict['mean_absolute_error'] = mae
    if 'root_mean_squared_error' in metrics:
        rmse = mean_squared_error(y_true, y_pred, squared=False)
        metric_dict['root_mean_squared_error'] = rmse
    if 'r2_score' in metrics:
        r2 = r2_score(y_true, y_pred)
        metric_dict['r2_score'] = r2

    #st.write('compute_eval_metrics not implemented yet.')
    return metric_dict


def plot_learning_curve(X_train, X_val, y_train, y_val, trained_model, metrics, model_name):
    """
    This function plots the learning curve. Note that the learning curve is calculated using 
    increasing sizes of the training samples
    Input:
        - X_train: training features
        - X_val: validation/test features
        - y_train: training targets
        - y_val: validation/test targets
        - trained_model: the trained model to be calculated learning curve on
        - metrics: a list of metrics to be computed
        - model_name: the name of the model being checked
    Output:
        - fig: the plotted figure
        - df: a dataframe containing the train and validation errors, with the following keys:
            - df[metric_fn.__name__ + " Training Set"] = train_errors
            - df[metric_fn.__name__ + " Validation Set"] = val_errors
    """
    fig = make_subplots(rows=len(metrics), cols=1,
                        shared_xaxes=True, vertical_spacing=0.1)
    df = pd.DataFrame()
    # Add code here
    fig = make_subplots(rows=len(metrics), cols=1,
                        shared_xaxes=True, vertical_spacing=0.1)
    df = pd.DataFrame()
    METRICS_MAP = {
    'mean_absolute_error': mae,
    'root_mean_squared_error': rmse,
    'r2_score': r2
}

    for i, metric in enumerate(metrics):
        metric_fn = METRICS_MAP[metric]
        train_errors, val_errors = [], []

        for m in range(500, len(X_train)+1, 500):
            trained_model.fit(X_train[:m], y_train[:m])
            y_train_pred = trained_model.predict(X_train[:m])
            y_val_pred = trained_model.predict(X_val)
            train_errors.append(metric_fn(y_train[:m], y_train_pred))
            val_errors.append(metric_fn(y_val, y_val_pred))

        fig.add_trace(go.Scatter(
            x=np.arange(500, len(X_train)+1, 500),
            y=train_errors,
            name=metric_fn.__name__+" Train"),
            row=i+1,
            col=1)
        fig.add_trace(go.Scatter(
            x=np.arange(500, len(X_train)+1, 500),
            y=val_errors,
            name=metric_fn.__name__+" Val"),
            row=i+1,
            col=1)
        fig.update_yaxes(title_text=metric_fn.__name__, row=i+1, col=1)
        
        df[metric_fn.__name__ + " Training Set"] = train_errors
        df[metric_fn.__name__ + " Validation Set"] = val_errors

    fig.update_xaxes(title_text="Training Set Size", row=len(metrics), col=1)
    fig.update_layout(title=model_name)
    st.plotly_chart(fig)

    return fig, df









