
import time
import os
import psutil


def program(file, columns, class_column):
    import pandas as pd
    import numpy as np
    from math import sqrt
    from sklearn.metrics import mean_squared_error
    from sklearn.neighbors import KNeighborsRegressor
    from sklearn.preprocessing import LabelEncoder
    from sklearn.model_selection import train_test_split

    # Importing the dataset
    dataset = pd.read_csv(file)
    dataset = dataset.dropna(subset=columns)
    X = dataset[columns].values
    y = dataset[class_column].values
    le = LabelEncoder()
    y = le.fit_transform(y)
    # Processing the dataset
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


def process_memory():
    process = psutil.Process(os.getpid())
    mem_info = process.memory_info()
    return mem_info.rss


execution_time = []
memory_usage = []
for x in range(10):
    mem_before = process_memory()
    start_time = time.perf_counter_ns()
    # Uncomment the program you want to run.
    #program('./IRIS.csv',['sepal_length', 'sepal_width', 'petal_length', 'petal_width'], 'species')
    #program('WineQT.csv', ['fixed acidity', 'volatile acidity', 'citric acid', 'residual sugar', 'chlorides',
    #        'free sulfur dioxide', 'total sulfur dioxide', 'density', 'pH', 'sulphates', 'alcohol'], 'quality')
    #program('weatherAUS.csv', ['MinTemp', 'MaxTemp', 'Rainfall', 'Evaporation', 'Sunshine', 'WindGustSpeed', 'WindSpeed9am', 'WindSpeed3pm',
    #                           'Humidity9am', 'Humidity3pm', 'Pressure9am', 'Pressure3pm', 'Cloud9am', 'Cloud3pm', 'Temp9am', 'Temp3pm'], 'RainTomorrow')
    end_time = time.perf_counter_ns()
    mem_after = process_memory()
    execution_time.append(end_time - start_time)
    memory_usage.append(mem_after - mem_before)


print(f"The execution time is: {execution_time} nanoseconds")
print(f"The memory consumption is : {memory_usage} bytes")
