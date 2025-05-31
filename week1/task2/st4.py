import matplotlib.pyplot as plt
from helper import read_csv, gradient_descent, relu, cross_entropy_loss_with_gradient
import numpy as np

if __name__ == "__main__":
    train_path = 'data/task2/st4/train.csv'
    test_path = 'data/task2/st4/test.csv'
    train_df = read_csv(train_path)
    test_df = read_csv(test_path)

    train_df['x^3'] = train_df['x'] ** 3
    train_df['y^3'] = train_df['y'] ** 3
    train_df['x^2y'] = (train_df['x'] ** 2) * train_df['y']
    train_df['xy^2'] = train_df['x'] * (train_df['y'] ** 2)
    train_df['xy'] = train_df['x'] * train_df['y']
    train_df['x^2'] = train_df['x'] ** 2
    train_df['y^2'] = train_df['y'] ** 2

    test_df['x^3'] = test_df['x'] ** 3
    test_df['y^3'] = test_df['y'] ** 3
    test_df['x^2y'] = (test_df['x'] ** 2) * test_df['y']
    test_df['xy^2'] = test_df['x'] * (test_df['y'] ** 2)
    test_df['xy'] = test_df['x'] * test_df['y']
    test_df['x^2'] = test_df['x'] ** 2
    test_df['y^2'] = test_df['y'] ** 2

    X_train = train_df[['x', 'y', 'x^3', 'y^3', 'x^2y', 'xy^2', 'xy', 'x^2', 'y^2']].to_numpy()
    y_train = train_df['color'].to_numpy().reshape(-1, 1)
    X_train = np.hstack((X_train, np.ones((X_train.shape[0], 1))))  # Add bias term
    
    w = np.random.rand(X_train.shape[1], y_train.shape[1])
    learning_rate = 1e-5
    epsilon = 1e-8

    X_test = test_df[['x', 'y', 'x^3', 'y^3', 'x^2y', 'xy^2', 'xy', 'x^2', 'y^2']].to_numpy()
    y_test = test_df['color'].to_numpy().reshape(-1, 1)
    X_test = np.hstack((X_test, np.ones((X_test.shape[0], 1))))  # Add bias term

    w = gradient_descent(X_train, y_train, w, learning_rate, epsilon, max_iterations=100000, func=relu, loss_fn=cross_entropy_loss_with_gradient)
    print("Gradient descent solution weights:")
    print(w)
    
    predictions = (X_test @ w) > 0
    accuracy = np.mean(predictions.flatten() == y_test.flatten())
    print(f"Accuracy on test set: {accuracy:.2f}")