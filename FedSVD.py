import time
import copy
import logging
import datetime
import argparse

from utils import *
from data_loader import *
from paths import *


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', '-m', type=str, default='svd')
    parser.add_argument('--num_participants', '-p', type=int, default=10)
    parser.add_argument('--num_samples', '-s', type=int, default=1000)
    parser.add_argument('--dataset', '-d', type=str, default='load_synthetic')
    parser.add_argument('--block', '-b', type=int, default=1)
    parser.add_argument('--alpha', '-a', type=float, default=1)
    parser.add_argument('--num_feature', '-f', type=int, default=1000)
    parser.add_argument('--only_time', '-t', type=str, default='False')
    parser.add_argument('--output_pkl', '-o', type=str, default='False')
    parser.add_argument('--log_dir', '-l', type=str, default='')
    args = parser.parse_args()

    # Parameters
    num_participants = args.num_participants
    # Number of samples per participant
    num_samples = args.num_samples

    only_evaluate_time = True if args.only_time == 'True' else False
    save_pickle = True if args.output_pkl == 'True' else False

    # Load the data
    if args.dataset == 'load_synthetic':
        X, label = eval(args.dataset + '(m=%s, n=%s, alpha=%s)' % (
            args.num_feature, args.num_participants*args.num_samples, args.alpha))
        save_file_name = '_'.join([args.mode, args.dataset[5:], 'f%s' % args.num_feature, 'a%s' % args.alpha,
                                   'p%s' % args.num_participants,
                                   's%s' % (args.num_samples * args.num_participants), 'b%s' % args.block])
    else:
        X, label = eval(args.dataset + '()')
        save_file_name = '_'.join([args.mode, args.dataset[5:], 'p%s' % args.num_participants,
                                   's%s' % (args.num_samples * args.num_participants), 'b%s' % args.block])

    if args.mode == 'pca':
        X = X[:, :num_participants*num_samples]
        print('PCA mode, subtracting the mean')
        X = X.T
        X -= np.mean(X, axis=0)
        X = X.T

    # Split the data for participants
    Xs = [X[:, e * num_samples: e * num_samples + num_samples] for e in range(num_participants)]
    if label is not None:
        Ys = [label[e * num_samples: e * num_samples + num_samples] for e in range(num_participants)]
    else:
        Ys = None

    # Init the efficiency file
    log_file_name = os.path.join(log_dir, args.log_dir,
                                 'FedSVD_' + datetime.datetime.now().strftime('%Y%m%d%H%M%S') + '.log')
    logging.basicConfig(level=logging.INFO, format='%(asctime)s-%(levelname)s: %(message)s',
                        filename=log_file_name)
    logging.info(str(args))
    print(str(args))

    ground_truth = np.concatenate(Xs, axis=1)
    m, n = ground_truth.shape

    mape_denominator = copy.deepcopy(ground_truth)
    mape_denominator[np.where(mape_denominator == 0)] = 1e-10

    # Standalone SVD
    start = time.time()
    U_normal, sigma_normal, VT_normal = svd(ground_truth)
    time_standalone_svd = time.time() - start
    U_normal = U_normal[:, :min(m, n)]
    VT_normal = VT_normal[:min(m, n), :]
    Xs_normal = U_normal @ np.diag(sigma_normal) @ VT_normal
    reconstruct_error_normal_mape = np.mean(np.abs((Xs_normal - ground_truth) / mape_denominator))
    reconstruct_error_normal_mae = np.mean(np.abs(Xs_normal - ground_truth))
    logging.info('StandaloneSVD time %s' % time_standalone_svd)
    print('StandaloneSVD time %s' % time_standalone_svd)
    if 'synthetic' in args.dataset:
        logging.info('StandaloneSVD reconstruct error (MAPE) %s' % reconstruct_error_normal_mape)
        print('StandaloneSVD reconstruct error (MAPE) %s' % reconstruct_error_normal_mape)
    else:
        logging.info('StandaloneSVD reconstruct error (MAE) %s' % reconstruct_error_normal_mae)
        print('StandaloneSVD reconstruct error (MAE) %s' % reconstruct_error_normal_mae)

    # FedSVD, Start the simulation
    logging.info('Simulation Start!')

    comm_each_data_holder = 0

    # Masking Server: Generate random orthogonal matrix P and Q
    start = time.time()
    P = generate_orthogonal_matrix(n=X.shape[0], reuse=False)
    t1 = time.time()
    Q = generate_orthogonal_matrix(n=np.sum([e.shape[1] for e in Xs]), reuse=False, block_reduce=args.block)
    t2 = time.time()
    Qs = [Q[e * num_samples: e * num_samples + num_samples] for e in range(num_participants)]
    time_generate_orthogonal = t2 - start
    if not only_evaluate_time:
        comm_each_data_holder += get_object_size(P)
        comm_each_data_holder += (get_object_size(Qs) / num_participants)
    logging.info('Generate orthogonal matrix %s done. Using %s seconds.' % (P.shape[0], t1-start))
    logging.info('Generate orthogonal matrix %s done. Using %s seconds.' % (Q.shape[0], t2-start))
    print('Generate orthogonal matrix done. Using %s seconds.' % time_generate_orthogonal)

    # Data Holders & Factorization Server: SecureAggregation to get X'
    start = time.time()
    X_mask_partitions = []
    for i in range(num_participants):
        X_mask_partitions.append(P @ Xs[i] @ Qs[i])
    X_mask, comm_size = secure_aggregation(X_mask_partitions, only_evaluate_time)
    # The time consumption of applying random mask runs in parallel by all the participants
    time_apply_orthogonal = (time.time() - start) / num_participants
    comm_each_data_holder += comm_size
    logging.info('Apply distortion done. Using %s seconds.' % time_apply_orthogonal)
    print('Apply distortion done. Using %s seconds.' % time_apply_orthogonal)

    # Decrypt the distorted_X, and perform the SVD decomposition
    start = time.time()
    U_mask, sigma, VT_mask = svd(X_mask)
    time_svd = time.time() - start
    U_mask = U_mask[:, :min(m, n)]
    VT_mask = VT_mask[:min(m, n), :]
    logging.info('SVD done. Using %s seconds.' % time_svd)
    print('SVD done. Using %s seconds.' % time_svd)

    # Recover the real singular values and vectors
    start = time.time()
    # (1) singular values (No need to recover)
    sigma = sigma
    # (2) the shared left singular vector
    U = P.T @ U_mask
    # (3) the secret right singular vector
    VTs = []
    k = 1
    transferred_variables = []
    for i in range(num_participants):
        Q_i = Qs[i].T
        R1_i = np.random.random([n, k])
        R2_i = np.random.random([Q_i.shape[1] + k, Q_i.shape[1] + k])
        Q_i_mask = np.concatenate([Q_i, R1_i], axis=-1) @ R2_i
        VT_i_mask = VT_mask @ Q_i_mask
        VTs.append((VT_i_mask @ np.linalg.inv(R2_i))[:, :Q_i.shape[1]])
        transferred_variables.append([Q_i_mask, VT_i_mask])
    time_reconstruct = (time.time() - start) / num_participants
    if not only_evaluate_time:
        comm_each_data_holder += (get_object_size(U_mask) + get_object_size(sigma))
        comm_each_data_holder += (get_object_size(transferred_variables) / num_participants)
    logging.info('SVD recover done. Using %s seconds.' % time_reconstruct)

    # Evaluation (FedSVD): reconstruct to measure the precision
    U = np.array(U)
    VTs = np.concatenate(VTs, axis=1)
    Xs_fed = U[:, :min(m, n)] @ np.diag(sigma) @ VTs[:min(m, n), :]
    reconstruct_error_fed_mape = np.mean(np.abs((Xs_fed - ground_truth) / mape_denominator))
    reconstruct_error_fed_mae = np.mean(np.abs(Xs_fed - ground_truth))
    if 'synthetic' in args.dataset:
        logging.info('FedSVD reconstruct error (MAPE) %s' % reconstruct_error_fed_mape)
        print('FedSVD reconstruct error (MPAE) %s' % reconstruct_error_fed_mape)
    else:
        logging.info('FedSVD reconstruct error (MAE) %s' % reconstruct_error_fed_mae)
        print('FedSVD reconstruct error (MAE) %s' % reconstruct_error_fed_mae)
    logging.info('Total comm size %s' % comm_each_data_holder)

    # End Simulation
    logging.info('Finished!')
    # Collect the time consumption
    time_consumption = [time_generate_orthogonal, time_apply_orthogonal, time_svd, time_reconstruct]

    if save_pickle:
        # Save the results to pickle filw
        result = {
            'dataset': args.dataset,
            'num_participants': args.num_participants,
            'num_samples': args.num_samples,
            'Xs': Xs,
            'Ys': Ys,
            'block_reduce': args.block,
            'fed_svd': {
                'U': U,
                'sigma': sigma,
                'VTs': VTs,
            },
            'standalone_svd': {
                'U': U_normal,
                'sigma': sigma_normal,
                'VTs': VT_normal,
            },
            'time_consumption': time_consumption,
            'comm_size': comm_each_data_holder,
            'reconstruct_error_fed_mape': reconstruct_error_fed_mape,
            'reconstruct_error_fed_mae': reconstruct_error_fed_mae,
            'reconstruct_error_normal_mape': reconstruct_error_normal_mape,
            'reconstruct_error_normal_mae': reconstruct_error_normal_mae,
            'log': log_file_name
        }

        with open(os.path.join(results_dir, save_file_name + '.pkl'), 'wb') as f:
            pickle.dump(result, f)