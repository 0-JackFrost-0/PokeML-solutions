from datasets import load_dataset
from sklearn.model_selection import train_test_split
from helper import sigmoid, gradient_descent, cross_entropy_loss_with_gradient
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import io

if __name__ == "__main__":
    ds = load_dataset("microsoft/cats_vs_dogs")
    data = ds['train'].to_pandas()
    
    def bytes_to_np_array(entry):
        img = Image.open(io.BytesIO(entry['bytes'])).convert("RGB")
        return np.array(img)

    data['image'] = data['image'].apply(bytes_to_np_array)
    # resize the images to 64x64
    data['image'] = data['image'].apply(lambda x: np.array(Image.fromarray(x).resize((64, 64))))
    
    # convert the images to grayscale
    data['image'] = data['image'].apply(lambda x: np.mean(x, axis=2).astype(np.uint8))

    X = np.stack(data['image'].values)
    y = data['labels'].values
    y = y.reshape(-1, 1) 
    X = X.reshape(X.shape[0], -1)
    X = X / 255.0  # Normalize pixel values to [0, 1]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    w = np.random.randn(X_train.shape[1], y_train.shape[1])
    learning_rate = 0.01
    epsilon = 1e-6
    max_iterations = 300
    w = gradient_descent(X_train, y_train, w, learning_rate, epsilon, max_iterations, func=sigmoid, loss_fn=cross_entropy_loss_with_gradient)
    print("Training completed.")

    # Evaluate on test set
    predictions = sigmoid(X_test @ w)
    predictions = (predictions > 0.5).astype(int)
    accuracy = np.mean(predictions == y_test)
    print(f"Test Accuracy: {accuracy * 100:.2f}%")

    # show 5 images with their predictions
    plt.figure(figsize=(10, 10))
    for i in range(5):
        plt.subplot(1, 5, i + 1)
        plt.imshow(X_test[i].reshape(64, 64), cmap='gray')
        plt.title(f"Predicted: {'Dog' if predictions[i] == 1 else 'Cat'}")
        plt.axis('off')
    plt.tight_layout()
    plt.show()