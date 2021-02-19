import time
import logging
import datetime
import argparse
import matplotlib

from utils import *
from data_loader import *


def order_by_value(value, vectors):
    assert len(value) == len(vectors)
    zip_values = [[np.real(value[i]), np.real(vectors[i])] for i in range(len(value))]
    sorted_values = sorted(zip_values, key=lambda x: x[0], reverse=True)
    return np.array([e[0] for e in sorted_values]), np.array([e[1] for e in sorted_values])


if __name__ == '__main__':

    results_dir = 'results'

    fed_svd_results = [e for e in os.listdir(results_dir) if e.startswith('pca') and e.endswith('.pkl')]
    fed_svd_results = [e for e in fed_svd_results if 'mnist' in e or 'wine' in e]
    print(fed_svd_results)

    color_list = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd',
                  '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']

    scatter_plot = True
    line_plot = True
    distance = {}

    for file in fed_svd_results:

        distance[file] = []

        with open(os.path.join(results_dir, file), 'rb') as f:
            data = pickle.load(f)

        # Data label
        if data['Ys'] is not None:
            label = np.concatenate(data['Ys'])
        else:
            label = None

        # Raw data and label
        X = np.concatenate(data['Xs'], axis=-1)
        covariance = X @ X.T / X.shape[1]

        # Standard PCA
        lambda_normal, U_normal = np.linalg.eig(covariance)
        lambda_normal, U_normal = order_by_value(lambda_normal, U_normal.T)
        X_reduce_normal = U_normal[:2] @ X

        # FedSVD PCA result
        sigma_fed = data['fed_svd']['sigma']
        U_fed = data['fed_svd']['U']
        lambda_fed, U_fed = order_by_value(sigma_fed, U_fed.T)
        X_reduce_fed = U_fed[:2] @ X

        X_reduce_dp = []
        title_label = []
        # for i in range(10, 5, -2):
        for i in [10, 5, 1]:
            epsilon = 0.01 * i
            title_label.append(epsilon)
            delta = 0.1
            # DP based PCA epsilon=0.1, delta=0.1
            # (1) Add noise to each party
            covariance_dp = [(e @ e.T) / X.shape[1] + epsilon_delta_dp_noise(
                epsilon=epsilon, delta=delta, d=e.shape[0], n=e.shape[1]
            ) for e in data['Xs']]
            # (2) Lossless simulation of the merging operation
            covariance_dp = np.sum(covariance_dp, axis=0)
            lambda_dp, U_dp = np.linalg.eig(covariance_dp)
            lambda_dp, U_dp = order_by_value(lambda_dp, U_dp.T)
            X_reduce_dp.append(U_dp[:2] @ X)

        if scatter_plot and ('mnist' in file or 'wine' in file):

            matplotlib.rcParams.update({'font.size': 30})
            # plots
            x_plot = [X_reduce_normal, X_reduce_fed] + X_reduce_dp

            # TMP to rotate the plots
            x_plot[0][0] *= -1
            x_plot[1] *= -1

            title = ['Standalone PCA', 'FedSVD'] + ['DP-FedPCA epsilon=%s' % e for e in title_label]
            Nr, Nc = 1, len(x_plot)
            k = 10
            fig, axs = plt.subplots(Nr, Nc, figsize=(k*Nc, k*Nr))

            for i in range(Nc):
                axs[i].scatter(x=x_plot[i][0], y=x_plot[i][1], c=[color_list[int(e)] for e in label], marker='.')
                axs[i].set_xlabel('First PC')
                axs[i].set_title(title[i])
                if i == 0:
                    axs[i].set_ylabel('Second PC')

            figure_name = os.path.join('images', file.replace('pkl', 'png'))
            fig.tight_layout(h_pad=0.5, w_pad=0.5)
            if figure_name:
                plt.savefig(figure_name, type="png", dpi=300)
            else:
                plt.show()