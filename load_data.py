import numpy as np
import torch
import random
import scipy.io
from sklearn.datasets import load_svmlight_file
from sklearn.model_selection import train_test_split
from sklearn.datasets import fetch_openml

np.random.seed(123)

def preprocess(filepath, features):
    with open(filepath, 'r') as file:
        lines = file.readlines()
    x = np.zeros((len(lines), features))
    y = np.zeros(len(lines))
    i = 0
    for line in lines:
        data = line.strip().split()
        y[i] = data[0]
        data = data[1:]
        for dataFeat in data:
            feat = dataFeat.split(':')
            x[i, int(feat[0]) - 1] = feat[1]
        i += 1
    return x, y


def load_data(data_path):
    if data_path == './data/kin8nm' or data_path == './data/year':
        data = np.loadtxt(data_path, delimiter=',')
    else:
        data = np.loadtxt(data_path)

    if data_path == "./data/naval" or data_path == "./data/energy":
        X_input = data[:, range(data.shape[1] - 2)]
        y_input = data[:, data.shape[1] - 2:]  # num x 2
    else:
        X_input = data[:, range(data.shape[1] - 1)]
        y_input = data[:, data.shape[1] - 1:]  # num x 1

    train_ratio = 0.9
    permutation = np.arange(X_input.shape[0])
    random.shuffle(permutation)

    size_train = int(np.round(X_input.shape[0] * train_ratio))
    index_train = permutation[0: size_train]
    index_test = permutation[size_train:]

    X_train, y_train = X_input[index_train, :], y_input[index_train, :]
    X_test, y_test = X_input[index_test, :], y_input[index_test, :]

    X_train, y_train = torch.from_numpy(X_train).float(), torch.from_numpy(y_train).float()
    X_test, y_test = torch.from_numpy(X_test).float(), torch.from_numpy(y_test).float()

    size_dev = min(int(round(0.1 * X_train.shape[0])), 500)
    X_dev, y_dev = X_train[-size_dev:], y_train[-size_dev:]
    X_train, y_train = X_train[:-size_dev], y_train[:-size_dev]

    std_X_train = torch.std(X_train, dim=0)  # num x X_dim --> X_dim
    std_X_train[std_X_train == 0] = 1
    mean_X_train = torch.mean(X_train, dim=0)

    mean_y_train = torch.mean(y_train)  # scalar
    std_y_train = torch.std(y_train)  # scalar

    return X_train, y_train, X_test, y_test, X_dev, y_dev, mean_X_train, mean_y_train, std_X_train, std_y_train


def load_data_for_blr(dataset, random_state=1):
    # make target be {+1, -1}
    if dataset == 'covertype':
        data = scipy.io.loadmat('./data/covertype.mat')
        X_input = data['covtype'][:, 1:]  # N x d
        y_input = data['covtype'][:, 0]  # N
        y_input[y_input == 2] = -1

        N = X_input.shape[0]
        X_input = np.hstack([X_input, np.ones([N, 1])])

        # split the dataset into train, test, val
        X_train, X_test, y_train, y_test = train_test_split(X_input, y_input, test_size=0.2, random_state=random_state)
        X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.1, random_state=random_state)

        X_train = torch.from_numpy(X_train).float()  # train_N x D
        y_train = torch.from_numpy(y_train).unsqueeze(-1).float()  # train_N x 1
        X_test = torch.from_numpy(X_test).float()
        y_test = torch.from_numpy(y_test).unsqueeze(-1).float()
        X_val = torch.from_numpy(X_val).float()
        y_val = torch.from_numpy(y_val).unsqueeze(-1).float()

    elif dataset == 'w8a':
        X_train, y_train = load_svmlight_file('./data/w8a/w8a.txt')
        X_test, y_test = load_svmlight_file('./data/w8a/w8a.t.txt')
        X_train = X_train.toarray()
        X_test = X_test.toarray()
        y_test[y_test == -1] = -1
        y_train[y_train == -1] = -1
        N_train, N_test = X_train.shape[0], X_test.shape[0]
        X_train = np.hstack([X_train, np.ones([N_train, 1])])
        X_test = np.hstack([X_test, np.ones([N_test, 1])])

        # split the dataset into train, test, val
        X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.1, random_state=random_state)

        X_train = torch.from_numpy(X_train).float()  # train_N x D
        y_train = torch.from_numpy(y_train).unsqueeze(-1).float()  # train_N x 1
        X_test = torch.from_numpy(X_test).float()
        y_test = torch.from_numpy(y_test).unsqueeze(-1).float()
        X_val = torch.from_numpy(X_val).float()
        y_val = torch.from_numpy(y_val).unsqueeze(-1).float()

    elif dataset == 'a9a':
        X_train, y_train = preprocess('./data/a9a/a9a', features=123)
        X_test, y_test = preprocess('./data/a9a/a9a.t', features=123)
        N_train, N_test = X_train.shape[0], X_test.shape[0]
        X_train = np.hstack([X_train, np.ones([N_train, 1])])
        X_test = np.hstack([X_test, np.ones([N_test, 1])])

        y_test[y_test == -1] = -1
        y_train[y_train == -1] = -1

        # split the dataset into train, test, val
        X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.1, random_state=random_state)

        X_train = torch.from_numpy(X_train).float()  # train_N x D
        y_train = torch.from_numpy(y_train).unsqueeze(-1).float()  # train_N x 1
        X_test = torch.from_numpy(X_test).float()
        y_test = torch.from_numpy(y_test).unsqueeze(-1).float()
        X_val = torch.from_numpy(X_val).float()
        y_val = torch.from_numpy(y_val).unsqueeze(-1).float()


    elif dataset == 'bioresponse':
        dataset = fetch_openml(data_id=4134)
        X_input = dataset.data  # N x d
        y_input = dataset.target  # N
        y_input[y_input == '1'] = 1
        y_input[y_input == '0'] = -1
        y_input = y_input.astype(float)
        N = X_input.shape[0]
        X_input = np.hstack([X_input, np.ones([N, 1])])

        # split the dataset into train, test, val
        X_train, X_test, y_train, y_test = train_test_split(X_input, y_input, test_size=0.2, random_state=random_state)
        X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.1,random_state=random_state)

        X_train = torch.from_numpy(X_train).float()  # train_N x D
        y_train = torch.from_numpy(y_train).unsqueeze(-1).float()  # train_N x 1
        X_test = torch.from_numpy(X_test).float()
        y_test = torch.from_numpy(y_test).unsqueeze(-1).float()
        X_val = torch.from_numpy(X_val).float()
        y_val = torch.from_numpy(y_val).unsqueeze(-1).float()

    else:
        raise Exception('Invalid dataset')

    return X_train, y_train, X_test, y_test, X_val, y_val
