"""
# Created by ashish1610dhiman at 12/01/23
Contact at ashish1610dhiman@gmail.com
"""

import sys
import matplotlib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import datetime

from src.ad_hmm import sku_predict
from src.utils import *
from statsmodels.tsa.seasonal import STL
from statsmodels.tsa.seasonal import MSTL

from tsprial.forecasting import *
from xgboost import XGBRegressor
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor
from statsmodels.tsa.ar_model import AutoReg, ar_select_order
from statsmodels.tsa.arima_model import ARIMA
# from pmdarima import auto_arima
from scipy.stats import boxcox
from scipy.special import inv_boxcox
# from math import exp,log

class stl_model():
    def __init__(self,sku_id, train_test_sku, n_lags_y, n_lags_exog, pred_cols):
        self.sku_id = sku_id
        self.train_test_sku = train_test_sku
        self.ML_models = {"xgb_340":XGBRegressor(max_depth = 5, n_estimators=340),\
             "xgb_1000":XGBRegressor(max_depth = 5, n_estimators=1000)}
        self.n_lags_y = n_lags_y
        self.n_lags_exog = n_lags_exog
        self.train_sku = None
        self.validation_sku = None
        self.pred_cols = pred_cols
        self.model_dict = None
        self.y_pred_dict = None

    def update_ml_models(self, user_dict):
        self.ML_models = user_dict

    #Split train test:
    def _split_train_test(self):
        train_test_sku = self.train_test_sku
        train_sku = train_test_sku[(train_test_sku.validation == False) &\
                                   (train_test_sku.validation_clean == True)]
        validation_sku = train_test_sku[(train_test_sku.validation == True) &\
                                        (train_test_sku.validation_clean == True)]
        self.train_sku = train_sku
        self.validation_sku = validation_sku
        return (train_sku, validation_sku)

    #STL decompose
    def _stl(self, train_sku, make_plot=False):
        res = STL(train_sku["DAILY_UNITS"], period=365).fit()
        if make_plot:
            res.plot()
            plt.show()
        return (res)

    #MSTL decompose
    def _mstl(self,train_sku, make_plot = False):
        res_m= MSTL(train_sku["DAILY_UNITS"], periods=(7, 30, 365)).fit()
        if make_plot:
            res_m.plot()
            plt.show()
        return (res_m)


    def fit_decompose(self):
        train_sku,validation_sku = self._split_train_test()
        res = self._stl(train_sku)
        res_m = self._mstl(train_sku)
        train_sku["STL_resid"] = res.resid
        train_sku["STL_trend"] = res.trend
        train_sku["STL_seasonal"] = res.seasonal

        train_sku["MSTL_resid"] = res_m.resid
        train_sku["MSTL_trend"] = res_m.trend
        train_sku[["MSTL_seasonal_7", "MSTL_seasonal_30", "MSTL_seasonal_365"]] = res_m.seasonal
        self.train_sku = train_sku
        # return (train_sku)

    def fit_autoreg(self,y_train, X_train, X_test):
        lag_select = ar_select_order(y_train, exog=X_train, maxlag=30)
        model = AutoReg(y_train, exog=X_train, lags=lag_select.ar_lags)
        model_fit = model.fit()
        pred = model_fit.predict(start=len(y_train), end=len(y_train) + 6, dynamic=True, exog_oos=X_test)
        pred = pd.Series(pred.values, index=X_test.index)
        return (model_fit, pred)

    def _ad_rmse(self, x ,y):
        return (np.mean(x - y) ** 2) ** 0.5

    def _get_col_subset(self,x,df):
        return([col for col in df.columns if x in col])

    def fit_models(self):
        model_dict = {}  # dict to save model
        y_pred_dict = {}  # dict to save predictions on test
        self.fit_decompose() #Fit STL
        train_sku, validation_sku = self.train_sku, self.validation_sku
        X_train = train_sku[self.pred_cols].astype(np.float64)
        X_test = validation_sku[self.pred_cols].astype(np.float64)
        train_pred = pd.DataFrame(index=X_train.index) #df to hold results

        component_cols = self._get_col_subset("STL", train_sku) #both STL and MSTL

        for component in component_cols:
            sku_id = self.sku_id
            #Box Cox
            y_train = train_sku[component].astype(np.float64)
            if y_train.min() < 0.00:
                C = -y_train.min() + 1
            else:
                C = 0
            y_train_transf, lam = boxcox(y_train + C)
            y_train_transf = pd.Series(y_train_transf, index=y_train.index)
            y_trnsf_inv = pd.Series(inv_boxcox(y_train_transf, lam) - C, index=y_train.index)
            rmse_boxcox = np.abs(self._ad_rmse(y_train, y_trnsf_inv))
            assert rmse_boxcox <= 0.00001, f"Problem with Box Cox:{component}, rmse:{rmse_boxcox}"

            # Auto Reg Model
            model_name = "auto_reg"
            ar_model, ar_pred_test = self.fit_autoreg(y_train_transf, X_train, X_test)
            model_pred_train = inv_boxcox(ar_model.fittedvalues, lam) - C
            train_pred[component + "_" + model_name] = pd.Series(model_pred_train, index=X_train.index)
            y_pred_dict[component, model_name] = inv_boxcox(ar_pred_test, lam) - C
            rmse_train = (np.mean((y_train - model_pred_train) ** 2)) ** 0.5
            model_dict[(component, model_name)] = (ar_model, rmse_train)
            print(f"{sku_id,component, model_name}, Train RMSE = {rmse_train:.4}")

            # ML models
            for model_name, model_type in self.ML_models.items():
                model = ForecastingCascade(model_type,
                                           lags=range(1, self.n_lags_y), use_exog=True,\
                                           exog_lags=range(1, self.n_lags_exog),
                                           accept_nan=False)
                model.fit(X_train, y_train_transf);
                model_pred_train = pd.Series(inv_boxcox(model.predict(X_train), lam) - C, index=X_train.index)
                train_pred[component + "_" + model_name] = model_pred_train
                rmse_train = (np.mean((y_train - model_pred_train) ** 2)) ** 0.5
                model_dict[(component, model_name)] = (model, rmse_train)
                y_pred_dict[component, model_name] = pd.Series(inv_boxcox(model.predict(X_test), lam) - C,
                                                               index=X_test.index)
                print(f"{sku_id,component, model_name}, Train RMSE = {rmse_train:.4}")
            print()
        self.model_dict,self.y_pred_dict = model_dict, y_pred_dict
        return (model_dict,y_pred_dict)

    def subset_min_model(self,component,df):
        return df.loc[df.component == component, "model_name"].values[0]


    def get_result(self):
        model_dict_df = pd.DataFrame(self.model_dict).T.reset_index()
        model_dict_df.columns = ["component", "model_name", "model", "train_rmse"]
        model_dict_df["rnk"] = model_dict_df.groupby(["component"])["train_rmse"].rank(method="first")
        min_rmse_models = model_dict_df.loc[model_dict_df.rnk == 1]
        #make predictions
        y_pred_dict = self.y_pred_dict
        y_pred_STL = y_pred_dict[("STL_resid", self.subset_min_model("STL_resid", min_rmse_models)) \
                         ] + y_pred_dict[("STL_trend", self.subset_min_model("STL_resid", min_rmse_models)) \
            ] + y_pred_dict[("STL_seasonal", self.subset_min_model("STL_resid", min_rmse_models))]
        y_pred_MSTL = y_pred_dict[("MSTL_resid", self.subset_min_model("MSTL_resid", min_rmse_models)) \
                          ] + y_pred_dict[("MSTL_trend", self.subset_min_model("MSTL_trend", min_rmse_models)) \
            ] + y_pred_dict[("MSTL_seasonal_7", self.subset_min_model("MSTL_seasonal_7", min_rmse_models)) \
            ] + y_pred_dict[("MSTL_seasonal_30", self.subset_min_model("MSTL_seasonal_30", min_rmse_models)) \
            ] + y_pred_dict[("MSTL_seasonal_365", self.subset_min_model("MSTL_seasonal_365", min_rmse_models))]

        y_pred = pd.DataFrame([y_pred_STL, y_pred_MSTL]).T
        y_pred.columns = ["STL_prediction", "MSTL_prediction"]
        y_pred["Encoded_SKU_ID"] = [self.sku_id]*y_pred.shape[0]
        y_pred["actual"] = self.validation_sku["DAILY_UNITS"]
        return (min_rmse_models,y_pred)



