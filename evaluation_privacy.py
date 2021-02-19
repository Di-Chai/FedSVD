import os
import time
import scipy
import pickle
import logging
import datetime
import argparse
import matplotlib.pyplot as plt
import sys

from utils import *
from data_loader import *


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', '-d', type=str, default='cifar10')
    args = parser.parse_args()

    dataset = args.dataset
    num_images = 10000
    
    if dataset == 'cifar10':
        X, label = load_cifar10()
        X = X[:, :num_images]
        image_shape = [32, 32, 3]
        cmap = None
    elif dataset == 'mnist':
        X, label = load_mnist()
        X = X[:, :num_images]
        image_shape = [28, 28]
        cmap = 'gray'
    else:
        raise ValueError(dataset)

    m, n = X.shape
    P = generate_orthogonal_matrix(m)
    Q = []
    for b in [1, 2, 4, 8, 16, 32, 64, int(n/1000), int(n/100), int(n/10)]:
        Q.append(generate_orthogonal_matrix(n, block_reduce=b, reuse=False))

    num_plots = 10
    i = 100
    plot_image = [X[:, i:i+num_plots].T.reshape([-1] + image_shape)] +\
                 [(P @ X @ e)[:, i:i+num_plots].T.reshape([-1] + image_shape) for e in Q] # + \
                 # [(X @ e)[:, i:i+num_plots].T.reshape([-1] + image_shape) for e in Q]

    # TMP print information
    for i in range(1, len(plot_image)):
        print(num_images,
              np.mean(np.abs(plot_image[0] - (plot_image[i] - plot_image[i].min())
                             / (plot_image[i].max() - plot_image[i].min()) * 255)))

    for i in range(len(plot_image)):
        plot_image[i] = (plot_image[i] - plot_image[i].min()) / (plot_image[i].max() - plot_image[i].min())
    plot_image = np.array(plot_image)
    plot_n_times_m_images(plot_image, fname=os.path.join('images', dataset + '_%s_distortion.png' % num_images), cmap=cmap)

