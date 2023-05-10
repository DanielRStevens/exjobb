from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import numpy as np
import pandas as pd
# Importing the dataset
dataset = pd.read_csv('./input/IRIS.csv')
feature_columns = ['sepal_length', 'sepal_width',
                   'petal_length', 'petal_width', 'species']
X = dataset[feature_columns].values
y = dataset['species'].values
le = LabelEncoder()
y = le.fit_transform(y)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=0)
