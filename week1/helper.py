import pandas as pd
import numpy as np

def read_csv(file_path):
    try:
        df = pd.read_csv(file_path)
        return df
    except FileNotFoundError:
        print(f"Error: The file {file_path} was not found.")
        return None
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        return None
    
def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def softmax(z):
    exp_z = np.exp(z - np.max(z, axis=1, keepdims=True)) 
    return exp_z / np.sum(exp_z, axis=1, keepdims=True)

def cross_entropy_loss_with_gradient(predictions, targets):
    loss = -np.mean(np.sum(targets * np.log(predictions + 1e-15), axis=1))
    gradient = (predictions - targets) / targets.shape[0]
    return loss, gradient

def encoder(x, val_list):
    one_hot = np.zeros(len(val_list))
    one_hot[val_list.index(x)] = 1
    return one_hot

def relu(z):
    return np.maximum(0, z)

def mse_loss_with_gradient(predictions, targets):
    errors = predictions - targets
    loss = np.mean(errors ** 2)
    gradient = (2 * errors) / targets.shape[0]
    return loss, gradient

def gradient_descent(X, y, w, learning_rate, epsilon, max_iterations=10000, func=None, loss_fn=None):
    for i in range(max_iterations):
        predictions = X @ w
        if func is not None:
            predictions = func(predictions)

        if loss_fn is not None:
            loss, grad = loss_fn(predictions, y)
        else:
            loss, grad = mse_loss_with_gradient(predictions, y)
            
        print(f"Iteration {i+1}, Loss: {loss}")
        gradient = (X.T @ grad)
        w -= learning_rate * gradient
        
        if np.linalg.norm(gradient) < epsilon:
            print(f"Convergence reached after {i+1} iterations.")
            break

    return w