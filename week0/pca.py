import matplotlib.pyplot as plt
from numpy.typing import NDArray
import numpy as np

def generate_dataset1() -> list[list[float], list[float]]:
    x = [np.random.rand() for i in range(1000)]
    y = [x[i] + 0.05*np.random.rand() for i in range(1000)]
    return [x, y]

def generate_dataset2() -> list[list[float], list[float]]:
    x : list[float] = [np.random.rand() for i in range(1000)]
    y : list[float] = [(0.5 - x[i])*(0.7 - x[i]) + 1 for i in range(1000)]
    return [x, y]


def pca(arr : NDArray) -> NDArray:
    arr = (arr - arr.mean(axis=1, keepdims=True))
    arr = arr/arr.std(axis=1, keepdims=True)
    cov = np.dot(arr, arr.T) / (arr.shape[1] - 1)
    print(cov.shape)
    eigvals, eigvecs = np.linalg.eig(cov)
    
    return eigvecs[:, np.argmax(eigvals)]

def plotter(vec: NDArray, arr: NDArray) -> None:
    point = np.mean(arr, axis=1) - eigvec/2
    point2 = vec+point
    plt.plot([point[0], point2[0]], [point[1], point2[1]], 'r-')
    plt.scatter(arr[0], arr[1])
    plt.show()
    
if __name__ == '__main__':
    [x, y] = generate_dataset1()
    [x2, y2] = generate_dataset2()

    arr1 : np.ndarray = np.array([x, y])
    arr2 : np.ndarray = np.array([x2, y2])

    eigvec = pca(arr1)
    plotter(eigvec, arr1)

    eigvec = pca(arr2)
    plotter(eigvec, arr2)

