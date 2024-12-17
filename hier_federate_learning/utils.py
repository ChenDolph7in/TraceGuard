import copy
import torch
import tenseal as ts
import pandas as pd

from torchvision import datasets, transforms
from sampling import mnist_iid, mnist_noniid, mnist_noniid_unequal
from sampling import cifar_iid, cifar_noniid
from sampling import av_iid, av_noniid
from torch.utils.data import Dataset


class CustomTensorDataset(Dataset):
    """ Custom dataset class to mimic MNIST dataset structure (with `.data` and `.targets` attributes). """

    def __init__(self, data, targets):
        self.data = data  # Features will be accessible via .data
        self.targets = targets  # Labels will be accessible via .targets

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, index):
        return self.data[index], self.targets[index]


def get_dataset(args):
    """ Returns train and test datasets and a user group which is a dict where
    the keys are the user index and the values are the corresponding data for
    each of those users.
    """

    if args.dataset == 'cifar':
        data_dir = 'data/cifar/'
        apply_transform = transforms.Compose(
            [transforms.ToTensor(),
             transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

        train_dataset = datasets.CIFAR10(data_dir, train=True, download=True,
                                         transform=apply_transform)

        test_dataset = datasets.CIFAR10(data_dir, train=False, download=True,
                                        transform=apply_transform)

        # sample training data amongst users
        if args.iid:
            # Sample IID user data from Mnist
            user_groups = cifar_iid(train_dataset, args.num_users)
        else:
            # Sample Non-IID user data from Mnist
            if args.unequal:
                # Chose uneuqal splits for every user
                raise NotImplementedError()
            else:
                # Chose euqal splits for every user
                user_groups = cifar_noniid(train_dataset, args.num_users)

    elif args.dataset == 'mnist' or args.dataset == 'fmnist':
        if args.dataset == 'mnist':
            data_dir = 'data/mnist/'
        else:
            data_dir = './data/fmnist/'

        apply_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))])

        train_dataset = datasets.MNIST(data_dir, train=True, download=True,
                                       transform=apply_transform)

        test_dataset = datasets.MNIST(data_dir, train=False, download=True,
                                      transform=apply_transform)

        # sample training data amongst users
        if args.iid:
            # Sample IID user data from Mnist
            user_groups = mnist_iid(train_dataset, args.num_users)
        else:
            # Sample Non-IID user data from Mnist
            if args.unequal:
                # Chose uneuqal splits for every user
                user_groups = mnist_noniid_unequal(train_dataset, args.num_users)
            else:
                # Chose euqal splits for every user
                user_groups = mnist_noniid(train_dataset, args.num_users)
    elif args.dataset == 'av':
        if args.iid:
            train_file_path = 'data/AV_DATA/train_usa_car_over.csv'
            test_file_path = 'data/AV_DATA/test_usa_car_over.csv'
        else:
            train_file_path = 'data/AV_DATA/train_usa_car_guest.csv'
            test_file_path = 'data/AV_DATA/test_usa_car_guest.csv'

        train_dataset = load_ca_data(train_file_path)
        test_dataset = load_ca_data(test_file_path)

        # sample training data amongst users
        if args.iid:
            # Sample IID user data from av dataset
            user_groups = av_iid(train_dataset, args.num_users)
        else:
            # Sample Non-IID user data from av dataset
            if args.unequal:
                raise NotImplementedError()
            else:
                user_groups = av_noniid(train_dataset, args.num_users)

    return train_dataset, test_dataset, user_groups


def load_ca_data(file_path):
    data = pd.read_csv(file_path)

    labels = torch.tensor(data.iloc[:, 1].values, dtype=torch.long)
    features = torch.tensor(data.iloc[:, 2:].values, dtype=torch.float32)

    dataset = CustomTensorDataset(features, labels)

    return dataset


def average_weights(w):
    """
    Returns the average of the weights.
    """
    w_avg = copy.deepcopy(w[0])
    for key in w_avg.keys():
        for i in range(1, len(w)):
            w_avg[key] += w[i][key]
        w_avg[key] = w_avg[key].div(len(w))
    return w_avg


def average_weights_ckks(w, context):
    """
    Returns the average of the weights.
    """
    w_avg = copy.deepcopy(w[0])
    for key in w_avg.keys():
        for i in range(1, len(w)):
            w_avg[key] += w[i][key]
        decrypted_values = w_avg[key].decrypt()
        decrypted_list = decrypted_values.tolist()
        original_shape = decrypted_values.shape
        decrypted_tensor = torch.tensor(decrypted_list, dtype=torch.float32).reshape(original_shape)
        w_avg[key] = ts.ckks_tensor(context, decrypted_tensor / len(w))
    return w_avg


def exp_details(args):
    print('\nExperimental details:')
    print(f'    Model     : {args.model}')
    print(f'    Optimizer : {args.optimizer}')
    print(f'    Learning  : {args.lr}')
    print(f'    Global Rounds   : {args.epochs}\n')

    print('    Federated parameters:')
    if args.iid:
        print('    IID')
    else:
        print('    Non-IID')
    print(f'    Fraction of users  : {args.frac}')
    print(f'    Local Batch size   : {args.local_bs}')
    print(f'    Local Epochs       : {args.local_ep}\n')
    return
