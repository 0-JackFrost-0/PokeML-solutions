import pandas as pd
from helper import read_csv, gradient_descent
import numpy as np

if __name__ == "__main__":
    file_path = "data/smol.csv"
    df = read_csv(file_path)

    X = df[['Temperature (F)', 'Humidity (%)', 'Rainfall (mm)']].to_numpy()
    y = df[['Mangoes (ton)', 'Oranges (ton)']].to_numpy()


    X = np.hstack((X, np.ones((X.shape[0], 1))))
    closed_w = np.linalg.inv(X.T @ X) @ X.T @ y
    
    print("Closed-form solution weights:")
    print(closed_w)

    learning_rate = 1e-5
    w = np.random.rand(X.shape[1], y.shape[1])
    print("Initial weights:")
    print(w)
    epsilon = 1e-8
    w = gradient_descent(X, y, w, learning_rate, epsilon)
    print("Gradient descent solution weights:")
    print(w)
    