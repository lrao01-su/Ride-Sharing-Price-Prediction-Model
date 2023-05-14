import streamlit as st                  # pip install streamlit
#from helper_functions import fetch_dataset
from sklearn.model_selection import train_test_split

from helper_functions import lasso_reg, linear_reg, random_forest, gradient_boost, one_hot_encode_feature, split_dataset

#############################################

st.markdown("# Practical Applications of Machine Learning (PAML)")

#############################################

st.markdown("### Farelytics ML: Using Machine Learning to Analyze and Compare Uber and Lyft Pricing Models")

#############################################

st.title('Train Model')

#############################################

df = st.session_state['data']

def inspect_coefficients(models):
    pass


if df is not None:
    # Display dataframe as table
    st.dataframe(df)
    string_columns = list(df.select_dtypes(['object']).columns)

    text_feature_select_onehot = st.selectbox('Select text features for One-hot encoding', string_columns)

    df = df.dropna()

    if (text_feature_select_onehot and st.button('One-hot Encode feature')):
        if 'one_hot_encode' not in st.session_state:
            st.session_state['one_hot_encode'] = {}
        if text_feature_select_onehot not in st.session_state['one_hot_encode']:
            st.session_state['one_hot_encode'][text_feature_select_onehot] = True
        else:
            st.session_state['one_hot_encode'][text_feature_select_onehot] = True
        df = one_hot_encode_feature(df, text_feature_select_onehot)

    feature_predict_select = st.selectbox(
        label='Select variable to predict',
        options=list(df.select_dtypes(include='number').columns),
        key='feature_selectbox',
        index=8
    )

    st.session_state['target'] = feature_predict_select

    # Select input features
    feature_input_select = st.multiselect(
        label='Select features for regression input',
        options=[f for f in list(df.select_dtypes(
            include='number').columns) if f != feature_predict_select],
        key='feature_multiselect'
    )

    st.session_state['feature'] = feature_input_select

    st.write('You selected input {} and output {}'.format(
        feature_input_select, feature_predict_select))


    X = df.loc[:, df.columns.isin(feature_input_select)]
    Y = df.loc[:, df.columns.isin([feature_predict_select])]



    # Split dataset
    st.markdown('### Split dataset into Train/Validation/Test sets')
    st.markdown(
        '#### Enter the percentage of validation/test data to use for training the model')
    number = st.number_input(
        label='Enter size of test set (X%)', min_value=0, max_value=100, value=30, step=1)


    X_train, X_val, y_train, y_val = split_dataset(X, Y, number)

    st.write(len(X_train))
    st.write(len(X_val))

    regression_methods_options = ['Linear Regression',
                                  'Lasso Regression', 'Random Forest', 'Gradient Boosting']

    # Collect ML Models of interests
    regression_model_select = st.multiselect(
        label='Select regression model for prediction',
        options=regression_methods_options,
    )
    st.write('You selected the follow models: {}'.format(
        regression_model_select))



    #  Linear Regression


    if (regression_methods_options[0] in regression_model_select):
        st.markdown('#### ' + regression_methods_options[0])

        cv_linear = st.number_input(
            label='Enter the Cross Validation',
            min_value=0,
            max_value=10,
            value=3,
            step=1,
            key='cv_linear'
        )

        st.write('You set the CV to: {}'.format(cv_linear))

        
        if st.button('Train Linear Regression Model'):
            linear_reg(
                X_train, y_train, cv_linear)

        if regression_methods_options[0] not in st.session_state:
            st.write('Linear Regression Model is untrained')
        else:
            st.write('Linear Regression Model trained')

    #  Lasso Regression

    if (regression_methods_options[1] in regression_model_select):
        st.markdown('#### ' + regression_methods_options[1])

        cv_lasso = st.number_input(
            label='Enter the Cross Validation',
            min_value=0,
            max_value=10,
            value=3,
            step=1,
            key='cv_lasso'
        )

        st.write('You set the CV to: {}'.format(cv_lasso))

        lasso_alpha = st.text_input(
                label='Input alpha values, separate by comma',
                value='0.001,0.0001',
                key='lasso_alpha'
            )
        lasso_alpha = [float(val) for val in lasso_alpha.split(',')]

        st.write('You select the following alpha value: {}'.format(lasso_alpha))


        
        if st.button('Train Lasso Regression Model'):
            lasso_reg(X_train, y_train, cv_lasso, lasso_alpha)

        if regression_methods_options[1] not in st.session_state:
            st.write('Lasso Regression Model is untrained')
        else:
            st.write('Lasso Model trained')

    #  Random Forest

    if (regression_methods_options[2] in regression_model_select):
        st.markdown('#### ' + regression_methods_options[2])

        cv_random_forest = st.number_input(
            label='Enter the Cross Validation',
            min_value=0,
            max_value=10,
            value=3,
            step=1,
            key='cv_random_forest'
        )

        st.write('You set the CV to: {}'.format(cv_random_forest))

        estimator_range = st.text_input(
                label='Input estimator range',
                value='100, 300',
                key='estimator_range'
            )

        estimator_range = [int(val) for val in estimator_range.split(',')]

        st.write('You select the following estimator range value: {}'.format(estimator_range))

        max_depth = st.text_input(
                label='Input max_depth',
                value='5, 10',
                key='max_depth'
            )

        max_depth = [int(val) for val in max_depth.split(',')]

        st.write('You select the following max_depth value: {}'.format(max_depth))

        
        if st.button('Train Random Forest Model'):
            random_forest(X_train, y_train, cv_random_forest, estimator_range, max_depth)

        if regression_methods_options[2] not in st.session_state:
            st.write('Random Forest Model is untrained')
        else:
            st.write('Random Forest Model trained')


    #  Gradient Boosting

    if (regression_methods_options[3] in regression_model_select):
        st.markdown('#### ' + regression_methods_options[3])

        cv_gb = st.number_input(
            label='Enter the Cross Validation',
            min_value=0,
            max_value=10,
            value=3,
            step=1,
            key='cv_gb'
        )

        st.write('You set the CV to: {}'.format(cv_gb))

        estimator_range_gb = st.text_input(
                label='Input estimator range',
                value='100, 300',
                key='estimator_range_gb'
            )

        estimator_range_gb = [int(val) for val in estimator_range_gb.split(',')]

        st.write('You select the following estimator range value: {}'.format(estimator_range_gb))

        lr = st.text_input(
                label='Input learning rate',
                value='0.01, 0.1',
                key='lr'
            )

        lr = [float(val) for val in lr.split(',')]

        st.write('You select the following estimator range value: {}'.format(lr))

        
        if st.button('Train Gradient Boosting Model'):
            gradient_boost(X_train, y_train, cv_random_forest, estimator_range_gb, lr)

        if regression_methods_options[3] not in st.session_state:
            st.write('Gradient Boosting Model is untrained')
        else:
            st.write('Gradient Boosting Model trained')


    st.write('Continue to Test Model')
