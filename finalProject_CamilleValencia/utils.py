# storage file for code that might be frequently used 
import torch
import numpy as np
import torch.nn as nn
import pandas as pd

# standardize numeric function 
def standardize_numeric(series: pd.Series, use_log: bool = False) -> pd.Series:
    if use_log:
        series = np.log(series)
   # Standardize the series
    standardized_series = (series - series.mean()) / series.std()
    return standardized_series


# mean-squared error function to be used in Model #3 
def mean_squared_error(predictions, targets):
    return np.mean((predictions - targets) ** 2)