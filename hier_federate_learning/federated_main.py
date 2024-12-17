import os
import sys

import copy
import time
import pickle
import numpy as np
import config as cfg
import torch

from tqdm import tqdm
from tensorboardX import SummaryWriter
from options import args_parser
from update import LocalUpdate, test_inference
from models import MLP, CNNMnist, CNNFashion_Mnist, CNNCifar, NN_Model
from utils import get_dataset, average_weights, exp_details

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../"))
sys.path.append(project_root)
from myUtils.WebSenderUtil import web_sender

if __name__ == '__main__':
    start_time = time.time()

    # define paths
    path_project = os.path.abspath('')
    logger = SummaryWriter('./logs/federated_main_iid')

    args = args_parser()
    exp_details(args)

    if args.gpu:
        torch.cuda.set_device(int(args.gpu))
    device = 'cuda' if args.gpu else 'cpu'

    # load dataset and user groups
    train_dataset, test_dataset, user_groups = get_dataset(args)

    # create init data to web
    nodes = {}
    for idx in range(args.num_users):
        node = {
            "status": "inactive",
            "local_epoch": args.local_ep,
            "current_epoch": 0,
            "total_dataset_size": len(user_groups[idx]),
            "current_dataset_size": 0,
            "loss": 'nan',
        }
        nodes[idx] = node
    data = {
        "total_epoch": args.epochs,
        "current_epoch": 0,
        "nodes": nodes,
        "avg_loss": 'nan',
        "avg_acc": 0,
    }
    web_sender(host=cfg.host, port=cfg.post, target=cfg.target, data=data)

    # BUILD MODEL
    if args.model == 'cnn':
        # Convolutional neural netork
        if args.dataset == 'mnist':
            global_model = CNNMnist(args=args)
        elif args.dataset == 'fmnist':
            global_model = CNNFashion_Mnist(args=args)
        elif args.dataset == 'cifar':
            global_model = CNNCifar(args=args)

    elif args.model == 'mlp':
        # Multi-layer preceptron
        img_size = train_dataset[0][0].shape
        len_in = 1
        for x in img_size:
            len_in *= x
            global_model = MLP(dim_in=len_in, dim_hidden=64,
                               dim_out=args.num_classes)
    elif args.model == 'nn':
        global_model = NN_Model()
    else:
        exit('Error: unrecognized model')

    # Set the model to train and send it to device.
    global_model.to(device)
    global_model.train()
    print(global_model)

    # copy weights
    global_weights = global_model.state_dict()

    # Training
    train_loss, train_accuracy = [], []
    val_acc_list, net_list = [], []
    cv_loss, cv_acc = [], []
    print_every = 2
    val_loss_pre, counter = 0, 0

    for epoch in tqdm(range(args.epochs)):
        local_weights, local_losses = [], []
        print(f'\n | Global Training Round : {epoch + 1} |\n')

        global_model.train()
        m = max(int(args.frac * args.num_users), 1)
        idxs_users = np.random.choice(range(args.num_users), m, replace=False)

        data["current_epoch"] = epoch + 1
        nodes = data["nodes"]
        for idx in nodes:
            if idx in idxs_users:
                nodes[idx]["status"] = 'active'
            else:
                nodes[idx]["status"] = 'inactive'
        web_sender(host=cfg.host, port=cfg.post, target=cfg.target, data=data)

        for idx in idxs_users:
            local_model = LocalUpdate(args=args, dataset=train_dataset,
                                      idxs=user_groups[idx], logger=logger, user_idx=idx)
            w, loss = local_model.update_weights(
                model=copy.deepcopy(global_model), global_round=epoch, data=data)
            local_weights.append(copy.deepcopy(w))
            local_losses.append(copy.deepcopy(loss))

        # update global weights
        global_weights = average_weights(local_weights)

        # update global weights
        global_model.load_state_dict(global_weights)

        loss_avg = sum(local_losses) / len(local_losses)
        logger.add_scalar('loss', loss_avg, epoch)
        train_loss.append(loss_avg)

        # Calculate avg training accuracy over all users at every epoch
        list_acc, list_loss = [], []
        global_model.eval()
        for idx in range(args.num_users):
            local_model = LocalUpdate(args=args, dataset=train_dataset,
                                      idxs=user_groups[idx], logger=logger, user_idx=idx)
            acc, loss = local_model.inference(model=global_model)
            list_acc.append(acc)
            list_loss.append(loss)
        train_accuracy.append(sum(list_acc) / len(list_acc))

        # print global training loss after every 'i' rounds
        if (epoch + 1) % print_every == 0:
            print(f' \nAvg Training Stats after {epoch + 1} global rounds:')
            print(f'Training Loss : {np.mean(np.array(train_loss))}')
            print('Train Accuracy: {:.2f}% \n'.format(100 * train_accuracy[-1]))
            data["avg_acc"] = train_accuracy[-1]
            data["avg_loss"] = np.mean(np.array(train_loss))
            web_sender(host=cfg.host, port=cfg.post, target=cfg.target, data=data)

    # Test inference after completion of training
    test_accuracy, test_precision, test_recall, test_loss = test_inference(args, global_model, test_dataset)

    print(f'\n Results after {args.epochs} global rounds of training:')
    print("|---- Avg Train Accuracy: {:.2f}%".format(100 * train_accuracy[-1]))
    print("|---- Test Accuracy: {:.2f}%".format(100 * test_accuracy))
    print("|---- Test Precision: {:.2f}%".format(100 * test_precision))
    print("|---- Test Recall: {:.2f}%".format(100 * test_recall))
    print("|---- Test Loss: {:.6f}".format(test_loss))

    # Saving the objects train_loss and train_accuracy:
    file_name = './save/objects/{}_{}_{}_C[{}]_iid[{}]_E[{}]_B[{}].pkl'. \
        format(args.dataset, args.model, args.epochs, args.frac, args.iid,
               args.local_ep, args.local_bs)

    with open(file_name, 'wb') as f:
        pickle.dump([train_loss, train_accuracy], f)

    print('\n Total Run Time: {0:0.4f}'.format(time.time() - start_time))

    # PLOTTING (optional)
    import matplotlib
    import matplotlib.pyplot as plt

    matplotlib.use('Agg')

    # Plot Loss curve
    plt.figure()
    plt.title('Training Loss vs Communication rounds')
    plt.plot(range(len(train_loss)), train_loss, color='r')
    plt.ylabel('Training loss')
    plt.xlabel('Communication Rounds')
    plt.savefig('./save/fed_{}_{}_{}_C[{}]_iid[{}]_E[{}]_B[{}]_loss.png'.
                format(args.dataset, args.model, args.epochs, args.frac,
                       args.iid, args.local_ep, args.local_bs))
    #
    # # Plot Average Accuracy vs Communication rounds
    plt.figure()
    plt.title('Average Accuracy vs Communication rounds')
    plt.plot(range(len(train_accuracy)), train_accuracy, color='k')
    plt.ylabel('Average Accuracy')
    plt.xlabel('Communication Rounds')
    plt.savefig('./save/fed_{}_{}_{}_C[{}]_iid[{}]_E[{}]_B[{}]_acc.png'.
                format(args.dataset, args.model, args.epochs, args.frac,
                       args.iid, args.local_ep, args.local_bs))
