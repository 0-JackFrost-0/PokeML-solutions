import numpy as np
from helper import read_csv, gradient_descent, sigmoid, cross_entropy_loss_with_gradient
from helper import encoder

if __name__ == '__main__':
    train_path = 'data/task2/st3/train.csv'
    test_path = 'data/task2/st3/test.csv'
    train_df = read_csv(train_path)
    test_df = read_csv(test_path)

    color_list = ['red', 'green', 'blue', 'teal', 'orange', 'purple']
    marker_list = ['+', '^', '*']
    print(train_df['marker'].value_counts())
    
    train_df['color'] = train_df['color'].map(lambda x: encoder(x, color_list))
    train_df['marker'] = train_df['marker'].map(lambda x: encoder(x, marker_list))
    test_df['color'] = test_df['color'].map(lambda x: encoder(x, color_list))
    test_df['marker'] = test_df['marker'].map(lambda x: encoder(x, marker_list))

    X_train = train_df[['x', 'y']].to_numpy()
    color_labels = np.array(train_df['color'].tolist())
    marker_labels = np.array(train_df['marker'].tolist())
    y_train = np.hstack((color_labels, marker_labels))
    X_train = np.hstack((X_train, np.ones((X_train.shape[0], 1))))  
    w = np.random.rand(X_train.shape[1], y_train.shape[1])
    learning_rate = 0.2
    epsilon = 1e-8

    # the difference here was to use sigmoid as the activation, instead of softmax, which helps for multi-label classification
    # alternatively, you could do a softmax, with 2 sets of weights, which beceomes a multi-output multi-class classification problem
    w = gradient_descent(X_train, y_train, w, learning_rate, epsilon, max_iterations=100000, func=sigmoid, loss_fn=cross_entropy_loss_with_gradient)
    print("Gradient descent solution weights:")
    print(w)

    # Evaluate on test set
    X_test = test_df[['x', 'y']].to_numpy()
    color_test = np.array(test_df['color'].tolist())
    marker_test = np.array(test_df['marker'].tolist())
    y_test = np.hstack((color_test, marker_test))
    X_test = np.hstack((X_test, np.ones((X_test.shape[0], 1))))
    predictions = sigmoid(X_test @ w)

    # converting one hot encoded predictions back to class labels
    predicted_colors = np.argmax(predictions[:, :len(color_list)], axis=1)
    predicted_markers = np.argmax(predictions[:, len(color_list):], axis=1)
    true_colors = np.argmax(y_test[:, :len(color_list)], axis=1)
    true_markers = np.argmax(y_test[:, len(color_list):], axis=1)

    color_accuracy = np.mean(predicted_colors == true_colors)
    marker_accuracy = np.mean(predicted_markers == true_markers)
    print(f"Color accuracy on test set: {color_accuracy:.2f}")
    print(f"Marker accuracy on test set: {marker_accuracy:.2f}")
