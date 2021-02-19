import os
import pickle
import numpy as np
np.random.seed(101)
import matplotlib.pyplot as plt

from pympler import asizeof


def svd(x):
    m, n = x.shape
    if m >= n:
        return np.linalg.svd(x)
    else:
        u, s, v = np.linalg.svd(x.T)
        return v.T, s, u.T


def random(size):
    return np.random.randint(low=-10**5, high=10**5, size=size) + np.random.random(size)


def secure_aggregation(xs, only_evaluate_time):
    n = len(xs)
    size = xs[0].shape
    # Step 1 Generate random samples between each other
    perturbations = []
    for i in range(n):
        tmp = []
        for j in range(n):
            tmp.append(random(size))
        perturbations.append(tmp)
    perturbations = np.array(perturbations)
    perturbations -= np.transpose(perturbations, [1, 0, 2, 3])
    ys = [xs[i] - np.sum(perturbations[i], axis=0) for i in range(n)]
    if not only_evaluate_time:
        comm_size = get_object_size(perturbations) / (n**2) * (n-1) + get_object_size(ys) / n
    else:
        comm_size = 0
    results = np.sum(ys, axis=0)
    return results, comm_size


def generate_orthogonal_matrix(n, reuse=False, block_reduce=None):
    orthogonal_matrix_cache_dir = 'orthogonal_matrices'
    if os.path.isdir(orthogonal_matrix_cache_dir) is False:
        os.makedirs(orthogonal_matrix_cache_dir, exist_ok=True)
    file_list = os.listdir(orthogonal_matrix_cache_dir)
    existing = [e.split('.')[0] for e in file_list]

    file_name = str(n)
    if block_reduce is not None:
        file_name += '_blc%s' % block_reduce

    if reuse and file_name in existing:
        with open(os.path.join(orthogonal_matrix_cache_dir, file_name + '.pkl'), 'rb') as f:
            return pickle.load(f)
    else:
        if block_reduce is not None:
            qs = [int(n / block_reduce)] * block_reduce
            if n % block_reduce != 0:
                qs[-1] += (n - np.sum(qs))
            q = np.zeros([n, n])
            for i in range(len(qs)):
                sub_n = qs[i]
                tmp = generate_orthogonal_matrix(sub_n, reuse=False, block_reduce=None)
                index = int(np.sum(qs[:i]))
                q[index:index+sub_n, index:index+sub_n] += tmp
        else:
            q, _ = np.linalg.qr(np.random.randn(n, n), mode='full')
        if reuse:
            with open(os.path.join(orthogonal_matrix_cache_dir, file_name + '.pkl'), 'wb') as f:
                pickle.dump(q, f, protocol=4)
        return q


def get_object_size(x):
    return asizeof.asizeof(pickle.dumps(x, protocol=4)) / (2 ** 20)


def plot_n_times_m_images(x, fname=None, cmap=None):

    Nr, Nc = x.shape[:2]
    fig, axs = plt.subplots(Nr, Nc)

    if Nc > 1:
        for i in range(Nr):
            for j in range(Nc):
                try:
                    axs[i, j].axis('off')
                    axs[i, j].imshow(x[i][j], cmap=cmap)
                except Exception as e:
                    print(i, j, e)
    else:
        for i in range(Nr):
            axs[i].imshow(x[i][0], cmap=cmap)
            axs[i].axis('off')
    fig.tight_layout(h_pad=0.5, w_pad=0.5)
    if fname:
        plt.savefig(fname, type="png", dpi=300)
    else:
        plt.show()


# DP-noise for co-variance matrix, which can be used in PCA
def epsilon_delta_dp_noise(epsilon, delta, d, n):
    # Reference: https://arxiv.org/abs/1907.08059
    omega = (d+1)/(n*epsilon) * np.sqrt(2*np.log((d**2 + d) / (2*delta*np.sqrt(2*np.pi)))) + 1/(n * np.sqrt(epsilon))
    return np.random.normal(loc=0, scale=omega**2, size=(d, d))


def uniform_noise(m, n, alpha):
    return np.random.uniform(low=-alpha, high=alpha, size=[m, n])


def normal_noise(m, n, delta):
    return np.random.normal(loc=0, scale=delta, size=[m, n])


def parse_log_params(log):
    args = log.strip('\n').split(': ')[-1].replace("'", '')
    args = [e.split('=') for e in args[10:-1].split(', ')]
    return dict(args)


if __name__ == '__main__':
    test = epsilon_delta_dp_noise(0.1, 0.1, 100, 10000)
    print(test)