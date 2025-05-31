import matplotlib.pyplot as plt
from helper import read_csv, gradient_descent, sigmoid
import numpy as np

if __name__ == "__main__":
    train_data_path = 'data/task2/st1/train.csv'
    test_data_path = 'data/task2/st1/test.csv'

    train_df = read_csv(train_data_path)
    test_df = read_csv(test_data_path)
    
    train_df['y-x^2'] = train_df['y'] - (train_df['x'] ** 2)

    X_train = train_df[['x','y','y-x^2']].to_numpy()
    y_train = train_df['color'].to_numpy().reshape(-1, 1)
    w = np.random.rand(X_train.shape[1], y_train.shape[1])
    learning_rate = 1e-4
    epsilon = 1e-8
    w = gradient_descent(X_train, y_train, w, learning_rate, epsilon, max_iterations=100000, func=sigmoid)
    print("Gradient descent solution weights:")
    print(w)

    # Evaluate on test set
    test_df['y-x^2'] = test_df['y'] - (test_df['x'] ** 2)
    X_test = test_df[['x', 'y', 'y-x^2']].to_numpy()
    y_test = test_df['color'].to_numpy().reshape(-1, 1)
    predictions = (X_test @ w) > 0
    accuracy = np.mean(predictions.flatten() == y_test.flatten())
    print(f"Accuracy on test set: {accuracy:.2f}")

