import pandas as pd
import numpy as np

# importing dataset
df = pd.read_csv("/home/gaurav/AI/Proiba_ML/Regression/RegressionData/headbrain.csv")

# creating X,Y
X = df["Head Size(cm^3)"].values
Y = df["Brain Weight(grams)"].values

# calculating mean
x_mean = np.mean(X)
y_mean = np.mean(Y)

# Calculating lenth of x to loop over all the records
n_epoch = len(X)

numerator = 0
denominator = 0

for i in range(n_epoch):
    numerator += (X[i] - x_mean) * (Y[i] - y_mean)
    denominator += (X[i] - x_mean) ** 2

m = numerator / denominator

# formula for c is
# y_mean = m * x_mean + c

c = y_mean - (m * x_mean)

print(m,c)

# to predict the Y value
y_pred = m * X[10] + c

print(y_pred)