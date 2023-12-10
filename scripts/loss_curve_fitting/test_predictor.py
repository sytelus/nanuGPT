import numpy as np
from nanugpt import lin_predictor

x = [1, 2, 3, 4, 5]
y = np.array([1, 2, 3, 4, 5])*2+5

x_heldout = np.array([1, 2, 3, 4, 5])+10
y_heldout = x_heldout*2+5

model = lin_predictor.fit(x, y)
print(model.coef_)
print(model.intercept_)
print(lin_predictor.predict(model, [6]))
print(lin_predictor.evaluate(model, x_heldout, y_heldout))

