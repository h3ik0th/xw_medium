# %%
import xlwings as xw

import numpy as np
import pandas as pd

from pmdarima.pipeline import Pipeline
import pmdarima as pmd
from scipy.stats import normaltest

import warnings
warnings.filterwarnings("ignore")


# >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
# %%
# before we implement the more complicated SARIMAX function,
# let's start with a simple Python function that can be called from an Excel cell:
# it takes a single numerical argument, e.g. the value in another cell, and
# raises it to the third power

@xw.func
def py_cubed(x):
    if np.isnan(x):
        x = 0
    return x**3


# %%
# next, we define a simple function that will return pandas.describe() 
#   for the Excel range we provide as argument
# in the xw.arg decorator, we advise Python to interpret the Excel range
#   as a dataframe with column headings and an index 

@xw.func
@xw.arg("timeseries", pd.DataFrame, index=True, header=True)
def py_describe(timeseries):
    return timeseries.describe()


# %%
# >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
# main function for the SARIMAX forecast, callable from an Excel cell;
#   the main function will call a sequence of helper functions that contribute to the SARIMAX forecast
#   to make it callable from Excel, it is preceded by the decorator @xw.func
# in the second decorator, @xw.arg, we advise Python to interpret the Excel range
#   as a dataframe with column headings and an index 

@xw.func
@xw.arg("timeseries", pd.DataFrame, index=True, header=True)
@xw.arg("pqPQ", pd.Series, index=False, header=False)
def py_SARIMAX(timeseries, pqPQ, m=12, n_test=12, alpha=.05):
    
    df_TS = pd.DataFrame(timeseries)
    df_TS.columns = ["y", "X"] 

    # get order of differencing from ADF, KPSS, OCSB and CH tests
    d = diff_order(timeseries["y"].dropna(), alpha)
    D = diffseas_order(timeseries["y"].dropna(), m, alpha) 

    # repackage SARIMAX paramaters
    p = pqPQ[0]
    q = pqPQ[1]
    P = pqPQ[2]
    Q = pqPQ[3]
    pdq = (p, d, q)
    PDQm = (P, D, Q, m)
    
    n_y = df_TS["y"].count()                                    # count filled rows of y
    n_X = df_TS["X"].count()                                    # count filled rows of X = actual + forecast periods
    n_fc = n_X - n_y
    y_train = df_TS.iloc[:(n_y - n_test), 0].to_frame()        # testing: reserve last n rows for testing
    X_train = df_TS.iloc[:(n_y - n_test), 1].to_frame()   
    y_test = df_TS.iloc[(n_y - n_test):n_y, 0].to_frame()
    X_test = df_TS.iloc[(n_y - n_test):n_y, 1].to_frame()     
    X_fc = df_TS.iloc[(-n_fc - n_test):, 1].to_frame()


    applyBoxCox = need_transform(timeseries["y"].dropna(), alpha)

    res = sarimax_prep(pdq, PDQm, applyBoxCox)          # define SARIMAX model

    df_sum = model_train(res, y_train, X_train)         # fit the SARIMAX model to the training dataset 


    # get predictions for training, test, and forecast period
    yhat_train = predict_train(res, X_train, alpha)
    yhat_test = forecast(res, X_test, n_test, alpha)
    yhat_fc = forecast(res, X_fc, n_fc, alpha)


    # include forecast as column in dataframe
    yhat = np.hstack((yhat_train, yhat_fc))
    s_fc = pd.Series(yhat)
    s_fc.index = df_TS.index[:len(s_fc)]
    df_TS["forecast"] = s_fc.astype("float64")
    df_TS = df_TS.rename(columns={"y":"actual"})
    df_TS["actual"] = df_TS["actual"].astype("float64")
    df_TS["X"] = df_TS["X"].astype("float64")
    df_TS["variance"] = (df_TS["forecast"] - df_TS["actual"]).astype("float64")
    df_TS["perc.var"] = (df_TS["forecast"] / df_TS["actual"] - 1).astype("float64")
    cols = ["forecast", "actual", "variance", "perc.var", "X"]  
    df_TS = df_TS[cols]
    

    # calculate prediction accuracy and return as a third dataframe
    df_acc = accuracy(yhat_train, yhat_test, y_train, y_test)


    # generate a numerical index for all dataframes
    # then concatenate them side by side and show them in the worksheet
    df_TS.reset_index(inplace=True)
    df_sum.reset_index(inplace=True)
    df_acc.reset_index(inplace=True)

    # then concatenate them side by side and show them in the worksheet
    df = pd.concat([df_TS, df_sum, df_acc], axis=1)
    
    return df



# %%
# helper function: time series normally distributed? if not, apply Box-Cox transformation

def need_transform(y, alpha):
    p = normaltest(y)[1]
    if p <= 0.05:
        applyBoxCox = True
    else:
        applyBoxCox = False
    return applyBoxCox


# %%
# helper function: define tests for order of first differencing 

def diff_order(y, alpha=0.05):
    n_kpss = pmd.arima.ndiffs(y, alpha=alpha, test='kpss', max_d=3)
    n_adf = pmd.arima.ndiffs(y, alpha=alpha, test='adf', max_d=3)
    n_diff = max(n_adf, n_kpss)
    return n_diff


# helper function: define tests for order of seasonal differencing

def diffseas_order(y, mseas=12, alpha=0.05):
    n_ocsb = pmd.arima.OCSBTest(m=mseas).estimate_seasonal_differencing_term(y)
    n_ch = pmd.arima.CHTest(m=mseas).estimate_seasonal_differencing_term(y)
    ns_diff = max(n_ocsb, n_ch)
    return ns_diff



# %%
# helper function: set up the SARIMAX model

def sarimax_prep(pdq, PDQm, applyBoxCox):
    intercept = True
    
    if applyBoxCox:
        pipe = Pipeline([
                        ('boxcox', pmd.preprocessing.BoxCoxEndogTransformer(lmbda2=1e-6)),
                        ('arima', pmd.arima.ARIMA(order=pdq, seasonal_order=PDQm, with_intercept=intercept, suppress_warnings=True))
                        ])
    else:
        pipe = pmd.arima.ARIMA(order=pdq, seasonal_order=PDQm, with_intercept=intercept, suppress_warnings=True)

    return pipe



# %%
# helper function: fit the SARIMAX model to the training data set and return a summary report

def model_train(res, y, X):
    res.fit(y=y, X=X)
    sum = res.summary()

    # the summary() function in pmdarima and statsmodels consists of 3 tables
    # we convert each of them to html
    # convert html to dataframes
    # and concatenate the 3 dataframes 
    # to get a single dataframe we can show in an Excel range
    
    df = pd.DataFrame(columns=range(7))   

    html0 = sum.tables[0].as_html()
    df0 = pd.read_html(html0, index_col=0)[0]

    html1 = sum.tables[1].as_html()
    df1 = pd.read_html(html1, index_col=0)[0]
    for x in df1:
        try:
            x = float(x)
        except:
            pass

    html2 = sum.tables[2].as_html()
    df2 = pd.read_html(html2, index_col=0)[0]

    df = pd.concat([df0, df1, df2])

    return df



# %%
# helper function: get predicted values for training dataset

def predict_train(res, X, alpha):
    yhat, conf_int = res.predict_in_sample(
        X=X,
        return_conf_int=True, 
        alpha=alpha,
        inverse_transform=True)
    return yhat



# %%
# helper function: get predicted values for forecast period

def forecast(res, X, n_per, alpha):
    yhat, conf_int = res.predict(
        X=X,
        n_periods=n_per,                               
        return_conf_int=True, 
        alpha=alpha, 
        inverse_transform=True)
    return yhat



# %%
# helper function: define prediction accuracy metrics

def prediction_accuracy(y_hat, y_act):
    mae = np.mean(np.abs(y_hat - y_act))                             # MAE
    mape = np.mean(np.abs(y_hat - y_act)/np.abs(y_act))              # MAPE
    rmse = np.mean((y_hat - y_act)**2)**.5                           # RMSE
    corr = np.corrcoef(y_hat, y_act)[0,1]                            # correlation of prediction and actual
    return({'MAE':mae, 
            'MAPE':mape, 
            'RMSE':rmse, 
            'Corr':corr})


# %%
# helper function: calculate accuracy for training and test predictions

def accuracy(yhat_train, yhat_test, y_train, y_test):
    train_accuracy = prediction_accuracy(yhat_train, y_train["y"])
    df_train_accuracy = pd.DataFrame.from_dict(train_accuracy, orient="index")
    df_train_accuracy.columns = ["training: accuracy"]

    test_accuracy = prediction_accuracy(yhat_test, y_test["y"])
    df_test_accuracy = pd.DataFrame.from_dict(test_accuracy, orient="index")
    df_test_accuracy.columns = ["testing: accuracy"]

    df_accuracy = pd.concat([df_train_accuracy, df_test_accuracy], axis=1)
    df_accuracy["test vs training"] = df_accuracy["testing: accuracy"] - df_accuracy["training: accuracy"]

    return df_accuracy




# %% xlwings "main" function for possible actions in Excel; not used in our example

def main():
    wb = xw.Book.caller()
    sheet = wb.sheets[0]
    pass
