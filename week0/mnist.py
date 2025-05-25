from datasets import load_dataset
import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse.linalg import eigsh

EPS = 1e-10
def pca(arr, k=78):
    cen_arr = arr - arr.mean(axis=0, keepdims=True)
    cen_arr = cen_arr / (cen_arr.std(axis=0, keepdims=True) + EPS)
    cov = np.dot(cen_arr.T, cen_arr) / (cen_arr.shape[0] - 1)
    eigenvals, eigvecs = eigsh(cov, k=k, which='LM')
    sorted_indices = np.argsort(eigenvals)[::-1]
    return eigvecs[:, sorted_indices]

if __name__ == '__main__':
    ds = load_dataset("ylecun/mnist")
    data = ds['train']
    X = np.array(data['image'])
    fig, axes = plt.subplots(1, 2)
    X = X.reshape(X.shape[0], -1).astype(np.float32)
    vecs = pca(X)
    mean = X.mean(axis=0, keepdims=True)

    std_dev = X.std(axis=0, keepdims=True) + EPS
    X = (X - mean) / std_dev

    proj_X = X@vecs@vecs.T
    proj_X = proj_X * std_dev + mean
    X = X * std_dev + mean

    X = X.reshape(X.shape[0], 28, 28).astype(np.float32)
    print(X.shape)
    axes[0].imshow(X[0, :, :], cmap='gray')
    axes[0].set_title('Original')
    axes[0].axis('off')

    proj_X = proj_X.reshape(proj_X.shape[0], 28, 28).astype(np.float32)
    axes[1].imshow(proj_X[0, :, :], cmap='gray')
    axes[1].set_title('Reconstructed using 78 components')
    axes[1].axis('off')
    plt.show()

