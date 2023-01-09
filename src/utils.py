"""
# Created by ashish1610dhiman at 07/01/23
Contact at ashish1610dhiman@gmail.com
"""
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

def rmse(y_act,y_pred):
    """
    y_act : pd df with Encoded_SKU_ID and SALES_DATE
    y_pred : pd df with Encoded_SKU_ID and SALES_DATE
    return --> rmse agg
    """
    assert all(y_act.dtypes.values == y_pred.dtypes.values), "Column types do not match"
    df_join = y_act.merge(y_pred,on=["Encoded_SKU_ID","SALES_DATE"])
    assert df_join.shape[0]==y_act.shape[0], "y_act and y_pred shapes do not match"
    rmse = np.sqrt(np.mean((df_join.actual-df_join.predicted)**2))
    return rmse

def rmse_sku(y_act,y_pred, asc_sort = False):
    """
    y_act : pd df with Encoded_SKU_ID and SALES_DATE
    y_pred : pd df with Encoded_SKU_ID and SALES_DATE
    return --> rmse for each sku
    """
    assert all(y_act.dtypes.values == y_pred.dtypes.values), "Column types do not match"
    df_join = y_act.merge(y_pred,on=["Encoded_SKU_ID","SALES_DATE"])
    assert df_join.shape[0]==y_act.shape[0], "y_act and y_pred shapes do not match"
    rmse_lambda = lambda x: np.sqrt(np.mean((x.actual-x.predicted)**2))
    rmse_skus = df_join.groupby("Encoded_SKU_ID").apply(lambda x: (rmse_lambda(x),x["actual"].mean()))
    rmse_skus1 = pd.DataFrame(rmse_skus.tolist(), index=rmse_skus.index)
    rmse_skus1.columns = ["rmse_du","mean_du"]
    rmse_skus1["pct_rmse"] = rmse_skus1["rmse_du"]/rmse_skus1["mean_du"]
    return rmse_skus1.sort_values(by=["rmse_du"],ascending = asc_sort)

def plot_pred_sku(train,y_act,y_pred, sku_id, start_dt = "2022-07-01"):
    """
    y_act : pd df with Encoded_SKU_ID and SALES_DATE
    y_pred : pd df with Encoded_SKU_ID and SALES_DATE
    return --> plot of actual vs predicted
    """
    assert all(y_act.dtypes.values == y_pred.dtypes.values), "Column types do not match"
    df_join = y_act.merge(y_pred,on=["Encoded_SKU_ID","SALES_DATE"])
    assert df_join.shape[0]==y_act.shape[0], "y_act and y_pred shapes do not match"
    train_sku = train[train.Encoded_SKU_ID==sku_id][["Encoded_SKU_ID","SALES_DATE","DAILY_UNITS"\
                                                ]].rename(columns={"DAILY_UNITS":'actual'})
    train_valid_sku = pd.concat([train_sku,y_act[y_act.Encoded_SKU_ID==sku_id]])
    y_pred_sku = y_pred[y_pred.Encoded_SKU_ID==sku_id]
    df_join = train_valid_sku.merge(y_pred_sku,on=["SALES_DATE"], how ="outer")
    df_join[df_join.SALES_DATE>=start_dt].plot(x="SALES_DATE",y=["actual","predicted"],color=["black","r"])
    plt.axvspan(y_pred_sku["SALES_DATE"].min(), y_pred_sku["SALES_DATE"].max(),\
            facecolor='pink', alpha=0.34, label ="Prediction week")
    plt.title (f"Actual/Predicted sales of sku:{sku_id} from {train_valid_sku.SALES_DATE.min():%Y-%m-%d} to {y_pred_sku.SALES_DATE.max():%Y-%m-%d}")
    plt.legend()
