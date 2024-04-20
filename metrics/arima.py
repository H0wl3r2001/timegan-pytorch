import warnings
from math import sqrt
from statsmodels.tsa.arima.model import ARIMA
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