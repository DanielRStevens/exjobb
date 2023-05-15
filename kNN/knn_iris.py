
import time
import os
import psutil


def program():
    import pandas as pd
    import numpy as np
    from math import sqrt
    from sklearn.metrics import mean_squared_error
    from sklearn.neighbors import KNeighborsRegressor
    from sklearn.preprocessing import LabelEncoder
    from sklearn.model_selection import train_test_split

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


def process_memory():
    process = psutil.Process(os.getpid())
    mem_info = process.memory_info()
    return mem_info.rss


execution_time = []
memory_usage = []
for x in range(10):
    mem_before = process_memory()
    start_time = time.perf_counter_ns()
    program()
    end_time = time.perf_counter_ns()
    mem_after = process_memory()
    execution_time.append(end_time - start_time)
    memory_usage.append(mem_after - mem_before)


print(f"The execution time is: {execution_time} nanoseconds")
print(f"The memory consumption is : {memory_usage}")
