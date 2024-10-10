import warnings
from math import sqrt
from statsmodels.tsa.arima.model import ARIMA
import numpy as np
from sklearn.metrics import mean_squared_error
import pandas as pd
from pmdarima.arima import auto_arima



#TODO: make it so the data entering the evaluate models function is in the format [[idx, val]]

def prepare_data(train, test):
    new_train = []
    new_test = []
    
    for i in range(len(train)):
        new_train.append([i, train[i][0][0]])
    for i in range(len(test)):
        new_test.append([len(train)+i, test[i][0][0]])

    new_train = pd.DataFrame(new_train, columns=['idx', 'val'])
    new_test = pd.DataFrame(new_test, columns=['idx', 'val'])
    
    return new_train, new_test

def prepare_data1(train, test, n):
    # Extract the first elements for the train data array
    first_elements_train = train[:, n, 1]
    # Create an array of indices for the train data array
    indices_train = np.arange(len(first_elements_train))
    
    # Create a pandas DataFrame for the train data array
    df_train = pd.DataFrame({
        'idx': indices_train,
        'val': first_elements_train
    })
    
    # Extract the first elements for the test data array
    first_elements_test = test[:, n, 1]
    # Create an array of indices for the test data array, starting after the last index of the train array
    indices_test = np.arange(len(first_elements_test)) + len(first_elements_train)
    
    # Create a pandas DataFrame for the test data array
    df_test = pd.DataFrame({
        'idx': indices_test,
        'val': first_elements_test
    })

    return df_train, df_test

def prepare_data2(train, idx):
    # Extract the first elements
    first_elements = train[:, idx, 1]

    # Create an array of indices
    indices = np.arange(len(first_elements))

    # Create a pandas DataFrame
    df = pd.DataFrame({
        'idx': indices,
        'val': first_elements
    })

    return df

def prepare_data3(train):
    # Create an array of indices
    indices = np.arange(len(train))

    # Create a pandas DataFrame
    df = pd.DataFrame({
        'idx': indices,
        'val': train
    })

    return df


def evaluate_arima_model(train, test, arima_order):
    history = [x for x in train]
    # make predictions
    predictions = list()
    for t in range(len(test)):
        model = ARIMA(history, order=arima_order)
        model_fit = model.fit()
        yhat = model_fit.forecast()[0]
        predictions.append(yhat)
        history.append(test[t])
    # calculate out of sample error
    rmse = sqrt(mean_squared_error(test, predictions))
    return rmse

def evaluate_models(train, test, p_values, d_values, q_values):
    warnings.filterwarnings("ignore")
    train, test = prepare_data(train, test)
    train = train['val']
    test = test['val']
    print(f"Training data: {train}")
    print(f"Test data: {test}")

    best_score, best_cfg = float("inf"), None
    for p in p_values:
        for d in d_values:
            for q in q_values:
                order = (p,d,q)
                try:
                    rmse = evaluate_arima_model(train, test, order)
                    if rmse < best_score:
                        best_score, best_cfg = rmse, order
                    print('ARIMA%s RMSE=%.3f' % (order,rmse))
                except:
                    continue
    print('Best ARIMA%s RMSE=%.3f' % (best_cfg, best_score))
    return best_score


def eval_model_auto(train, test):
    warnings.filterwarnings("ignore")
    train, test = prepare_data(train, test)
    
    df = pd.concat([train, test], ignore_index=True)
    model=auto_arima(df.set_index('idx'), trace=True, error_action='ignore', suppress_warnings=True, seasonal=False)

    model.fit(df.set_index('idx'))


def find_best_arima_model(train, test):
    warnings.filterwarnings("ignore")
    train_df, test_df = prepare_data(train, test)
    
    # Extract training and testing data
    train_data = train_df['val'].values
    test_data = test_df['val'].values
    
    # Use auto_arima to find the best ARIMA model configuration
    arima_model = auto_arima(train_data, seasonal=False, trace=True)
    order = arima_model.order  # Best (p, d, q) order found by auto_arima
    
    # Fit the ARIMA model on the training data with the best order
    model = ARIMA(train_data, order=order)
    fitted_model = model.fit()
    
    # Forecast using the fitted model on the test data
    forecast= fitted_model.forecast(len(test_data))
    
    # Calculate RMSE (Root Mean Squared Error)
    rmse = sqrt(mean_squared_error(test_data, forecast))
    
    # Return the best ARIMA model configuration and RMSE
    return order, rmse


def generate_arima_models(train, test):
    warnings.filterwarnings("ignore")
    
    order_list = []

    for i in range(7):
        train_df, _ = prepare_data1(train, test, i)
    
        # Extract training and testing data
        train_data = train_df['val'].values
    
        # Use auto_arima to find the best ARIMA model configuration
        arima_model = auto_arima(train_data, seasonal=False, trace=True)
        order = arima_model.order  # Best (p, d, q) order found by auto_arima

        order_list.append(order)
    

    return order_list
