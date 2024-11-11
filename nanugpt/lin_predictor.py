from typing import Tuple, Iterator, List, Dict, Callable, Union, Any, Optional
import numpy as np
import pandas as pd
import scipy
from scipy.optimize import curve_fit
from scipy.stats import linregress
from sklearn.linear_model import LinearRegression

def fit(x, y)->LinearRegression:
    xa = np.array(x).reshape(-1, 1)
    ya = np.array(y)

    model = LinearRegression().fit(xa, ya)

    # Coefficients (slope) and intercept
    # slope = model.coef_[0]
    # intercept = model.intercept_

    return model

def predict(model:LinearRegression, x): # x->(n_samples, n_features)
    xa = np.array(x).reshape(-1, 1)
    return model.predict(xa)    # -> (n_samples,)

def evaluate(model:LinearRegression, x, y):
    xa = np.array(x).reshape(-1, 1)
    y_pred = model.predict(xa)
    mae = np.mean(np.abs(y - y_pred))
    return mae