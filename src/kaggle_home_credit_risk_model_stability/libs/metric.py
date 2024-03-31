import numpy as np
from sklearn.metrics import roc_auc_score

def calculate_gini_stability_metric(dataframe, w_fallingrate=88.0, w_resstd=-0.5):
    gini_in_time = dataframe \
        .sort_values("WEEK_NUM") \
        .groupby("WEEK_NUM")[["true", "predicted"]] \
        .apply(lambda x: 2 * roc_auc_score(x["true"], x["predicted"]) - 1).tolist()
    
    x = np.arange(len(gini_in_time))
    y = gini_in_time
    a, b = np.polyfit(x, y, 1)
    y_hat = a * x + b
    residuals = y - y_hat
    res_std = np.std(residuals)
    avg_gini = np.mean(gini_in_time)
    
    #print(avg_gini, min(0, a), res_std)
    
    return avg_gini + w_fallingrate * min(0, a) + w_resstd * res_std


def calculate_gini_stability_metric_in_worse_case(dataframe, w_fallingrate=88.0, w_resstd=-0.5):
    gini_in_time = dataframe \
        .sort_values("WEEK_NUM") \
        .groupby("WEEK_NUM")[["true", "predicted"]] \
        .apply(lambda x: 2 * roc_auc_score(x["true"], x["predicted"]) - 1).tolist()

    gini_in_time = list(reversed(sorted(gini_in_time)))
    x = np.arange(len(gini_in_time))
    y = gini_in_time
    a, b = np.polyfit(x, y, 1)
    y_hat = a * x + b
    residuals = y - y_hat
    res_std = np.std(residuals)
    avg_gini = np.mean(gini_in_time)
    
    #print(avg_gini, min(0, a), res_std)
    
    return avg_gini + w_fallingrate * min(0, a) + w_resstd * res_std