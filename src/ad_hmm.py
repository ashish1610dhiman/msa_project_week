import pandas as pd
from hmmlearn.hmm import GaussianHMM
import itertools
import numpy as np


class sku_predict():
    #initialise
    def __init__(self,train_test, sku_id):
        self.sku_id = sku_id
        train_test_sku = train_test[train_test.Encoded_SKU_ID == sku_id]
        train_test_sku.index = train_test_sku["SALES_DATE"]
        train_test_sku = train_test_sku.sort_index()
        self.train_test_sku = train_test_sku
        self.sales_data = None
        self.n_lags = None
        self.X = None
        self.model = None

    def get_features(self, n_lags = 3):
        self.n_lags = n_lags
        sales_data = self.train_test_sku[["DAILY_UNITS"]]
        for lag in range(1,n_lags+1):
            du_lag = f"DAILY_UNITS_lag{lag}"
            sales_data[du_lag] = sales_data["DAILY_UNITS"].shift(lag)
            sales_data[f"change_lag{lag}"] = (sales_data["DAILY_UNITS"] - sales_data[du_lag])
        print (f"Created {n_lags} lag features")
        self.sales_data = sales_data #featurised data
        return sales_data

    def split_train_test(self,start_dt):
        train1 = self.sales_data[:pd.to_datetime(start_dt)+pd.DateOffset(-1)]
        valid1 = self.sales_data[pd.to_datetime(start_dt):]
        return (train1,valid1)

    def fit_hmm(self,train, start_dt, n_components1 = 2):
        hmm = GaussianHMM(n_components=n_components1)
        # fit hmm to pct_change and DAILY_UNITS_lag1
        lag_price_cols = [f"DAILY_UNITS_lag{lag}" for lag in range(1,self.n_lags+1)]
        lag_change_cols = [f"change_lag{lag}" for lag in range(1,self.n_lags+1)]
        print ("Training on :",lag_price_cols,lag_change_cols)
        X = train[lag_price_cols + lag_change_cols]
        X = X[start_dt:]
        self.X = X
        hmm.fit(X.dropna())
        self.model = hmm

    def _compute_all_possible_outcomes(self,n_steps_pct,n_steps_price_lag):
        # pct_change_range = np.linspace(-0.61, 0.61, n_steps_pct)
        # change_range = np.linspace(self.X["change_lag1"].min(),self.X["change_lag1"].max(), n_steps_pct)
        #TODO
        DAILY_UNITS_range = np.unique(self.X.dropna()[[col for col in self.X.columns if "UNITS" in col]])
        change_range = np.unique(self.X.dropna()[[col for col in self.X.columns if "change" in col]])
        # DAILY_UNITS_range = np.linspace(self.X["DAILY_UNITS_lag1"].min(),\
        #                                      self.X["DAILY_UNITS_lag1"].quantile(0.75), n_steps_price_lag)
        change_list = [DAILY_UNITS_range]*self.n_lags + [change_range]*self.n_lags
        all_outcomes = np.array(list(itertools.product(*change_list)))
        return (all_outcomes)

    def _get_most_probable_outcome(self,prev_data,n_steps_pct,n_steps_price_lag):
        hmm = self.model
        all_outcomes = self._compute_all_possible_outcomes(n_steps_pct,n_steps_price_lag)
        outcome_score = list(map(lambda x: hmm.score(np.row_stack((prev_data, x))), \
                                 all_outcomes))
        most_probable_outcome = all_outcomes[np.argmax(outcome_score)]
        return (most_probable_outcome)

    def predict(self,valid1, pred_latency, n_steps_pct = 100,n_steps_price_lag = 100):
        prev_data_init = self.X[-pred_latency:]
        predict_df = pd.DataFrame()
        print ("Starting Prediction")
        for i, dt in enumerate(valid1.index):
            if i == 0:
                prev_data = prev_data_init
            else:
                # print(prev_data.shape)
                self.model.fit(prev_data[-pred_latency:])
            print (f"-----> Predicting for day:{i}")
            most_probable_outcome = self._get_most_probable_outcome(prev_data, 100, 100)
            temp = pd.DataFrame(most_probable_outcome).T
            temp.index = [dt]
            temp.columns = prev_data_init.columns
            prev_data = pd.concat([prev_data, temp])
            predict_df = pd.concat([predict_df, temp])
            # print(most_probable_outcome)
        predict_df["predicted1"] = predict_df["DAILY_UNITS_lag1"] + (predict_df["change_lag1"])
        predict_df["predicted2"] = predict_df["DAILY_UNITS_lag2"] + (predict_df["change_lag2"])
        predict_df["predicted"] = (0.55*predict_df["predicted1"] + 0.45*predict_df["predicted2"])
        return predict_df
