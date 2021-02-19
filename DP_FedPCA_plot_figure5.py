import json
import time
import logging
import datetime
import argparse
import matplotlib

from paths import *
from utils import *
from data_loader import *
from sklearn.metrics.pairwise import cosine_similarity
from mpl_toolkits.axes_grid1.inset_locator import mark_inset
from mpl_toolkits.axes_grid1.inset_locator import inset_axes


def order_by_value(value, vectors):
    assert len(value) == len(vectors)
    zip_values = [[np.real(value[i]), np.real(vectors[i])] for i in range(len(value))]
    sorted_values = sorted(zip_values, key=lambda x: x[0], reverse=True)
    return np.array([e[0] for e in sorted_values]), np.array([e[1] for e in sorted_values])


if __name__ == '__main__':

    fed_svd_results = [e for e in os.listdir(results_dir) if e.startswith('pca') and e.endswith('.pkl')]
    print(fed_svd_results)

    color_list = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd',
                  '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']

    scatter_plot = True
    line_plot = True
    distance = {}

    for ratio in [0, 0.5, 0.9]:

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

            index_top_percent = None
            for i in range(1, len(lambda_normal)):
                if sum(lambda_normal[:i]) > (ratio * sum(lambda_normal)):
                    index_top_percent = i
                    break

            # FedSVD PCA result
            sigma_fed = data['fed_svd']['sigma']
            U_fed = data['fed_svd']['U']
            lambda_fed, U_fed = order_by_value(sigma_fed, U_fed.T)
            X_reduce_fed = U_fed[:2] @ X

            distance[file].append(
                np.mean(np.abs(np.diag(cosine_similarity(U_normal[:index_top_percent], U_fed[:index_top_percent]))))
            )

            X_reduce_dp = []
            # for i in range(1, 51):
            for i in [0.01, 0.05, 0.1, 0.5, 1]:
                # epsilon = 0.1 * i
                epsilon = i
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

                distance[file].append(
                    np.mean(np.abs(np.diag(cosine_similarity(U_normal[:index_top_percent], U_dp[:index_top_percent]))))
                )

        matplotlib.rcParams.update({'font.size': 15})

        x_ticks = [chr(949) + '=%s' % e for e in [0.01, 0.05, 0.1, 0.5, 1]]

        for key in distance:

            plt.close()

            fig, ax = plt.subplots(1, 1, figsize=[6, 4])

            ax.plot(list(range(len(x_ticks))),
                    [distance[key][0]] * len(x_ticks), label='FedSVD', color='green',
                    dashes=[5, 2, 1, 1])
            ax.bar(x=list(range(len(x_ticks))), height=distance[key][1:],
                   width=0.3, alpha=0.6, color='orange', label='DP-FedPCA')

            ax.set_ylim(0, 1.1)
            # ax.set_title(str(int(ratio * 100)) + '% variance explained' if ratio > 0 else '1st PC')
            # ax.set_xlabel('DP-FedPCA')
            ax.set_ylabel('Cosine Similarity')
            ax.set_ylabel('Cosine Similarity')
            ax.set_xticks(list(range(len(x_ticks))))
            ax.set_xticklabels(x_ticks)
            ax.legend(loc='center right')

            if os.path.isdir(os.path.join(images_dir, 'pca_hist')) is False:
                os.makedirs(os.path.join(images_dir, 'pca_hist'))

            plt.savefig(
                os.path.join(os.path.join(images_dir, 'pca_hist'),
                             key + '_' + (str(int(ratio * 100)) if ratio > 0 else '1st') + '.png'),
                type="png", dpi=100
            )
