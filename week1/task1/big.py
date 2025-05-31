import pandas as pd
import re
import pandas as pd
from helper import gradient_descent, mse_loss_with_gradient
from sklearn.preprocessing import StandardScaler
# from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import numpy as np


def preprocess_floor_info(floor_series):
    floor_extracted = floor_series.str.extract(r'(\d+)|Ground', flags=re.IGNORECASE).fillna('0')
    floor_processed = floor_extracted[0].replace('Ground', '0', regex=True).astype(int)
    return floor_processed.to_numpy()

if __name__ == "__main__":
    file_path = "data/house_rent_dataset.csv"
    df = pd.read_csv(file_path)

    floor_info = preprocess_floor_info(df['Floor'])
    floor_info = floor_info.reshape(-1, 1)

    cols = df[['BHK', 'Size', 'Bathroom']].to_numpy()
    scaler = StandardScaler()

    X = np.hstack((floor_info, cols))
    X = scaler.fit_transform(X)
    y = df['Rent'].to_numpy().reshape(-1, 1)
    train_X,test_X, train_y, test_y = train_test_split(X, y, test_size=0.2, random_state=42)
    
    w = np.random.rand(X.shape[1], y.shape[1])
    learning_rate = 1e-4
    epsilon = 1e-8

    w = gradient_descent(train_X, train_y, w, learning_rate, epsilon, max_iterations=100000, loss_fn=mse_loss_with_gradient)
    # w = LinearRegression().fit(train_X, train_y).coef_.T
    print("Gradient descent solution weights:")
    print(w)

    predictions = test_X @ w
    mse = np.mean((predictions - test_y) ** 2)
    mae = np.mean(np.abs(predictions - test_y))
    r2 = 1 - (np.sum((predictions - test_y) ** 2) / np.sum((test_y - np.mean(test_y)) ** 2))
    print(f"Mean Squared Error on test set: {mse:.2f}")
    print(f"Mean Absolute Error on test set: {mae:.2f}")
    print(f"R^2 Score on test set: {r2:.2f}")