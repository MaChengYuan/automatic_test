import pickle
from sklearn.datasets import make_regression
from sklearn.metrics import mean_absolute_error
import numpy as np
import pandas as pd



model = pickle.load(open("models/model.pkl", "rb"))

X_test, y = make_regression(1000,n_features = 10)

# Test on the model
y_hat = model.predict(X_test)


result = mean_absolute_error(y_hat,y)


with open('result.txt',"w") as outfile:
     outfile.write(f"mean_absolute_error : {result}")