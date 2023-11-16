import numpy as np
import pandas as pd
import scipy
from scipy.optimize import curve_fit
from scipy.stats import linregress

from nanugpt import utils

data_filepath = utils.full_path('scripts/loss_curve_fitting/loss_data.txt')
results_filepath = utils.full_path('scripts/loss_curve_fitting/results.txt')

with open(data_filepath, 'r') as f:
    lines = f.readlines()

data = []
for line in lines:
    step, loss = line.strip().split()
    data.append((int(step), float(loss)))

def exponential_decay(x, a, b, c):
    return a * np.exp(b * x) + c

def fit_and_evaluate(data, heldout):
    results = []

    params = (1, -1, 1) # initial guess

    for i in range(len(params), len(data)-1):
        # Fit model until i+1 data points
        x, y = zip(*data[:i+1])
        x = np.array(x)
        y = np.array(y)

        params, covariance = curve_fit(exponential_decay, x, y, p0=params)

        # Confidence intervals
        alpha = 0.05  # 95% confidence interval
        dof = max(0, len(y) - len(params))  # degrees of freedom
        t_val = scipy.stats.t.ppf(1.0-alpha/2., dof)
        confidence = np.sqrt(np.diag(covariance)) * t_val

        # Quality of fit - R^2
        residuals = y - exponential_decay(x, *params)
        ss_res = np.sum(residuals**2)
        ss_tot = np.sum((y - np.mean(y))**2)
        r_squared = 1 - (ss_res / ss_tot)

        # Mean absolute error for all next points
        future_residuals = data[i+1:]
        fx, fy = zip(*future_residuals)
        fx, fy = np.array(fx), np.array(fy)
        fy_pred = exponential_decay(np.array(fx), *params)
        mae_future = np.mean(np.abs(np.array(fy) - fy_pred))

        # Mean absolute error for last 5 points
        heldout_x, heldout_y = zip(*heldout)
        heldout_x, heldout_y = np.array(heldout_x), np.array(heldout_y)
        heldout_pred = exponential_decay(heldout_x, *params)
        mae_last_5 = np.mean(np.abs(heldout_y - heldout_pred))
        last_actual, last_pred = heldout_y[-1], heldout_pred[-1]

        results.append([x[-1], *params, *confidence, r_squared, mae_future, mae_last_5, last_actual, last_pred])
    return results

# Fit and evaluate
results = fit_and_evaluate(data[:-5], data[-5:])

# Save to file
df = pd.DataFrame(results, columns=["step", "a", "b", "c",
                                    "confidence_a", "confidence_b", "confidence_c",
                                    "R^2", "MAE_all_next", "MAE_last_5",
                                    "last_actual", "last_pred"])
df.to_csv(results_filepath, sep="\t", index=False)
