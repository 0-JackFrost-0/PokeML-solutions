from datasets import load_dataset
from helper import gradient_descent, softmax, cross_entropy_loss_with_gradient
import numpy as np
import matplotlib.pyplot as plt

if __name__ == "__main__":
    ds = load_dataset("ylecun/mnist")
    train_data = ds['train']
    test_data = ds['test']
    X_train, y_train = np.array(train_data['image']), np.array(train_data['label']).reshape(-1, 1)
    X_test, y_test = np.array(test_data['image']), np.array(test_data['label']).reshape(-1, 1)
    

    X_train = X_train.reshape(X_train.shape[0], -1)
    X_test = X_test.reshape(X_test.shape[0], -1)
    X_train = np.hstack((X_train, np.ones((X_train.shape[0], 1))))
    X_test = np.hstack((X_test, np.ones((X_test.shape[0], 1))))

    y_train = np.eye(10)[y_train.flatten()]
    y_test = np.eye(10)[y_test.flatten()] 
    print("Training data shape:", X_train.shape, y_train.shape)

    w = np.random.rand(X_train.shape[1], y_train.shape[1])

    learning_rate = 0.005
    epsilon = 1e-8
    w = gradient_descent(X_train, y_train, w, learning_rate, epsilon, max_iterations=300, func=softmax,loss_fn=cross_entropy_loss_with_gradient)
    print("Gradient descent solution weights:")
    print(w)
    predictions = X_test @ w
    predicted_classes = np.argmax(predictions, axis=1)
    true_classes = np.argmax(y_test, axis=1)
    accuracy = np.mean(predicted_classes == true_classes)
    print(f"Accuracy on test set: {accuracy:.2f}")

    # show 5 predictions
    fig, axes = plt.subplots(1, 5, figsize=(15, 3))
    for i in range(5):
        axes[i].imshow(X_test[i, :-1].reshape(28, 28), cmap='gray')
        axes[i].set_title(f"Predicted: {predicted_classes[i]}\nTrue: {true_classes[i]}")
        axes[i].axis('off')
    plt.tight_layout()
    plt.show()
    