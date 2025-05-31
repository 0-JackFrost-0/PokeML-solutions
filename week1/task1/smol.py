import pandas as pd
from helper import read_csv, gradient_descent, mse_loss_with_gradient
import numpy as np

if __name__ == "__main__":
    file_path = "data/smol.csv"
    df = read_csv(file_path)

    X = df[['Temperature (F)', 'Humidity (%)', 'Rainfall (mm)']].to_numpy()
    y = df[['Mangoes (ton)', 'Oranges (ton)']].to_numpy()

    X = np.hstack((X, np.ones((X.shape[0], 1))))
    closed_w = np.linalg.inv(X.T @ X) @ X.T @ y

    learning_rate = 1e-5
    w = np.random.rand(X.shape[1], y.shape[1])
    
    epsilon = 1e-8
    w = gradient_descent(X, y, w, learning_rate, epsilon, loss_fn=mse_loss_with_gradient)
    print("Gradient descent solution weights:")
    print(w)

    print("Closed-form solution weights:")
    print(closed_w)

    predictions = X @ w
    mse = np.mean((predictions - y) ** 2)
    mae = np.mean(np.abs(predictions - y))
    r2 = 1 - (np.sum((predictions - y) ** 2) / np.sum((y - np.mean(y, axis=0)) ** 2))
    print(f"Mean Squared Error: {mse:.2f}")
    print(f"Mean Absolute Error: {mae:.2f}")
    print(f"R^2 Score: {r2:.2f}")
