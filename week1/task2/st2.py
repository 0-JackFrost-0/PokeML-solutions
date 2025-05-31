import matplotlib.pyplot as plt
from helper import read_csv, gradient_descent, encoder, softmax, cross_entropy_loss_with_gradient
import numpy as np


if __name__ == "__main__":
    train_path = 'data/task2/st2/train.csv'
    test_path = 'data/task2/st2/test.csv'

    train_df = read_csv(train_path)
    test_df = read_csv(test_path)

    color_list = ['red', 'green', 'blue', 'teal', 'orange', 'purple']

    # convert the below to one hot encoding
    train_df['label'] = train_df['color'].map(lambda x: encoder(x, color_list))
    test_df['label'] = test_df['color'].map(lambda x: encoder(x, color_list))

    X_train = train_df[['x', 'y']].to_numpy()
    y_train = np.array(train_df['label'].tolist())
    X_train = np.hstack((X_train, np.ones((X_train.shape[0], 1))))  # Add bias term
    w = np.random.rand(X_train.shape[1], y_train.shape[1])
    learning_rate = 0.1
    epsilon = 1e-8
    w = gradient_descent(X_train, y_train, w, learning_rate, epsilon, max_iterations=100000, func=softmax, loss_fn=cross_entropy_loss_with_gradient)
    print("Gradient descent solution weights:")
    print(w)

    # Evaluate on test set
    X_test = test_df[['x', 'y']].to_numpy()
    y_test = np.array(test_df['label'].tolist())
    X_test = np.hstack((X_test, np.ones((X_test.shape[0], 1))))  # Add bias term
    predictions = X_test @ w
    predicted_classes = np.argmax(predictions, axis=1)
    true_classes = np.argmax(y_test, axis=1)
    accuracy = np.mean(predicted_classes == true_classes)
    print(f"Accuracy on test set: {accuracy:.2f}")