import numpy as np
import pandas as pd
import scipy
from scipy.optimize import curve_fit
from scipy.stats import linregress
from sklearn.linear_model import LinearRegression

from nanugpt import utils

data_filepath = utils.full_path('scripts/loss_curve_fitting/loss_data.txt')
results_filepath = utils.full_path('scripts/loss_curve_fitting/results_lin.txt')

with open(data_filepath, 'r') as f:
    lines = f.readlines()

data = []
for line in lines:
    step, loss = line.strip().split()
    data.append((int(step), float(loss)))

def curve_model(x, a, b):
    return a + b * x

def fit_and_evaluate(data, heldout):
    results = []

    n_data = 30
    for i in range(n_data, len(data)-1):
        # Fit model until i+1 data points
        x, y = zip(*data[i+1-n_data:i+1])
        x = np.array(x).reshape(-1, 1)
        y = np.array(y)

        model = LinearRegression().fit(x, y)

        # Coefficients (slope) and intercept
        slope = model.coef_[0]
        intercept = model.intercept_

        # Mean absolute error for all next points
        future_residuals = data[i+1:]
        fx, fy = zip(*future_residuals)
        fx, fy = np.array(fx), np.array(fy)
        fy_pred = model.predict(fx.reshape(-1, 1))
        mae_future = np.mean(np.abs(np.array(fy) - fy_pred))

        # Mean absolute error for last 5 points
        heldout_x, heldout_y = zip(*heldout)
        heldout_x, heldout_y = np.array(heldout_x), np.array(heldout_y)
        heldout_pred = model.predict(heldout_x.reshape(-1, 1))
        mae_last_5 = np.mean(np.abs(heldout_y - heldout_pred))
        last_actual, last_pred = heldout_y[-1], heldout_pred[-1]

        results.append([x[-1], slope, intercept, mae_future, mae_last_5, last_actual, last_pred])
    return results

# Fit and evaluate
n_heldout = 10
results = fit_and_evaluate(data[:-n_heldout], data[-n_heldout:])

# Save to file
df = pd.DataFrame(results, columns=["step", "slope", "intercept",
                                    #"c",
                                    # "confidence_a", "confidence_b",
                                    #"confidence_c",
                                    #"R^2",
                                    "MAE_all_next", "MAE_last_5",
                                    "last_actual", "last_pred"])
df.to_csv(results_filepath, sep="\t", index=False)
