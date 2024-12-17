import os
import time

import torch
import matplotlib.pyplot as plt

from tqdm import tqdm
from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter
from utils import get_dataset
from options import args_parser
from update import test_inference
from models import MLP, CNNMnist, CNNFashion_Mnist, CNNCifar, NN_Model

if __name__ == '__main__':
    start_time = time.time()
    path_project = os.path.abspath('')

    # Initialize the SummaryWriter for TensorBoard logging
    logger = SummaryWriter('./logs/baseline_main')

    args = args_parser()
    if args.gpu:
        torch.cuda.set_device(int(args.gpu))
    device = 'cuda' if args.gpu else 'cpu'

    # load datasets
    train_dataset, test_dataset, _ = get_dataset(args)

    # BUILD MODEL
    if args.model == 'cnn':
        # Convolutional neural network
        if args.dataset == 'mnist':
            global_model = CNNMnist(args=args)
        elif args.dataset == 'fmnist':
            global_model = CNNFashion_Mnist(args=args)
        elif args.dataset == 'cifar':
            global_model = CNNCifar(args=args)

    elif args.model == 'mlp':
        # Multi-layer perceptron
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

    # Set optimizer and criterion
    if args.optimizer == 'sgd':
        optimizer = torch.optim.SGD(global_model.parameters(), lr=args.lr,
                                    momentum=0.5)
    elif args.optimizer == 'adam':
        optimizer = torch.optim.Adam(global_model.parameters(), lr=args.lr,
                                     weight_decay=1e-4)

    trainloader = DataLoader(train_dataset, batch_size=64, shuffle=True)

    if args.dataset == 'av':
        # Binary classification
        criterion = torch.nn.BCELoss().to(device)
    else:
        # Multi-class classification
        criterion = torch.nn.NLLLoss().to(device)

    epoch_loss = []

    for epoch in tqdm(range(args.epochs)):
        batch_loss = []

        correct = 0  # 初始化正确预测的数量
        total = 0  # 初始化总样本数

        for batch_idx, (images, labels) in enumerate(trainloader):
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = global_model(images)

            if args.dataset == 'av':
                # For binary classification
                outputs = outputs.squeeze()
                loss = criterion(outputs, labels.float())
                # 将输出转换为 0 或 1
                predicted = (outputs > 0.5).float()
            else:
                # For multi-class classification
                loss = criterion(outputs, labels)
                # 获取预测的最大概率类别
                _, predicted = torch.max(outputs, 1)

            loss.backward()
            optimizer.step()

            # 计算精度
            correct += (predicted == labels).sum().item()
            total += labels.size(0)

            if batch_idx % 50 == 0:
                accuracy = 100. * correct / total
                print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}\tAccuracy: {:.2f}%'.format(
                    epoch + 1, batch_idx * len(images), len(trainloader.dataset),
                    100. * batch_idx / len(trainloader), loss.item(), accuracy))

            batch_loss.append(loss.item())

        loss_avg = sum(batch_loss) / len(batch_loss)
        print('\nTrain loss:', loss_avg)
        epoch_loss.append(loss_avg)

        # Log epoch loss to TensorBoard
        logger.add_scalar('loss', loss_avg, epoch)

    # Testing
    test_accuracy, test_precision, test_recall, test_loss = test_inference(args, global_model, test_dataset)
    print('Test on', len(test_dataset), 'samples')
    print("Test Accuracy: {:.2f}%".format(100 * test_accuracy))
    print("Test Precision: {:.2f}%".format(100 * test_precision))
    print("Test Recall: {:.2f}%".format(100 * test_recall))
    print("Test Loss: {:.6f}".format(test_loss))

    print('\n Total Run Time: {0:0.4f}'.format(time.time() - start_time))

    # Plot loss
    plt.figure()
    plt.plot(range(len(epoch_loss)), epoch_loss)
    plt.xlabel('epochs')
    plt.ylabel('Train loss')
    plt.savefig('./save/nn_{}_{}_{}.png'.format(args.dataset, args.model,
                                                args.epochs))

    # Close the TensorBoard logger
    logger.close()
