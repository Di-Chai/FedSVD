import os
import pickle
import numpy as np
np.random.seed(101)
from paths import *
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import TfidfVectorizer

data_dir = 'dataset'


def load_mnist(n=100000):
    data = np.load(os.path.join(data_dir, 'mnist', 'mnist.npz'))
    x_train = data['x_train'].reshape([-1, 28 * 28])
    y_train = data['y_train']
    x_test = data['x_test'].reshape([-1, 28 * 28])
    y_test = data['y_test']
    x = np.concatenate([x_train, x_test], axis=0).astype(float)
    y = np.concatenate([y_train, y_test], axis=0).astype(float)
    return x[:n].T, y[:n]


def load_cifar10():
    with open(os.path.join(data_dir, 'cifar10', 'data_batch_1'), 'rb') as f:
        data = pickle.load(f, encoding='bytes')
    X, y = data[b'data'].T, data[b'labels']
    return X.reshape([3, 32, 32, -1]).transpose([1, 2, 0, 3]).reshape([32*32*3, -1]), y


def load_wine():
    with open(os.path.join(data_dir, 'wine', 'winequality-red.csv')) as f:
        wine_red = f.readlines()[1:]
        wine_red = [[float(e1) for e1 in e.strip('\n').split(';')] for e in wine_red]
    with open(os.path.join(data_dir, 'wine', 'winequality-white.csv')) as f:
        wine_white = f.readlines()[1:]
        wine_white = [[float(e1) for e1 in e.strip('\n').split(';')] for e in wine_white]
    wine = np.array(wine_red + wine_white)
    label = np.concatenate([np.zeros(len(wine_red)), np.ones(len(wine_white))])
    return wine.T, label


def load_synthetic(m, n, alpha):
    # Reference: https://github.com/andylamp/federated_pca/blob/master/synthetic_data_gen.m
    k = min(m, n)
    U, _ = np.linalg.qr(np.random.randn(m, m))
    Sigma = np.array(list(range(1, k+1))).astype(np.float32) ** -alpha
    V = np.random.randn(k, n)
    Y = (U @ np.diag(Sigma) @ V) / np.sqrt(n-1)
    yn = np.max(np.sqrt(np.sum(Y ** 2, axis=1)))
    Y /= yn
    return Y, None


if __name__ == '__main__':
    a = load_cifar10()