from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_squared_error
from math import sqrt
import numpy as np
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('./IRIS.csv')
feature_columns = ['sepal_length', 'sepal_width',
                   'petal_length', 'petal_width']
X = dataset[feature_columns].values
y = dataset['species'].values
le = LabelEncoder()
y = le.fit_transform(y)
# Training the model
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=0)
knn_model = KNeighborsRegressor(n_neighbors=3)
knn_model.fit(X_train, y_train)
# Evaluating
train_preds = knn_model.predict(X_train)
mse = mean_squared_error(y_train, train_preds)
rmse = sqrt(mse)
print("Root-mean-square error:", rmse)
