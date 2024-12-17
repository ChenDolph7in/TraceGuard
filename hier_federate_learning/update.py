import os
import sys

import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../"))
sys.path.append(project_root)
from myUtils.WebSenderUtil import web_sender


class DatasetSplit(Dataset):
    """An abstract Dataset class wrapped around Pytorch Dataset class.
    """

    def __init__(self, dataset, idxs):
        self.dataset = dataset
        self.idxs = [int(i) for i in idxs]

    def __len__(self):
        return len(self.idxs)

    def __getitem__(self, item):
        image, label = self.dataset[self.idxs[item]]
        return torch.tensor(image), torch.tensor(label)


class LocalUpdate(object):
    def __init__(self, args, dataset, idxs, logger, user_idx):
        self.args = args
        self.logger = logger
        self.user_idx = user_idx
        self.trainloader, self.validloader, self.testloader = self.train_val_test(
            dataset, list(idxs))
        self.device = 'cuda' if args.gpu else 'cpu'
        # Default criterion set to NLL loss function
        if args.dataset == 'av':
            # Binary classification
            self.criterion = torch.nn.BCELoss().to(self.device)
        else:
            # Multi-class classification
            self.criterion = torch.nn.NLLLoss().to(self.device)

    def train_val_test(self, dataset, idxs):
        """
        Returns train, validation and test dataloaders for a given dataset
        and user indexes.
        """
        # split indexes for train, validation, and test (80, 10, 10)
        idxs_train = idxs[:int(0.8 * len(idxs))]
        idxs_val = idxs[int(0.8 * len(idxs)):int(0.9 * len(idxs))]
        idxs_test = idxs[int(0.9 * len(idxs)):]

        trainloader = DataLoader(DatasetSplit(dataset, idxs_train),
                                 batch_size=self.args.local_bs, shuffle=True)
        validloader = DataLoader(DatasetSplit(dataset, idxs_val),
                                 batch_size=max(int(len(idxs_val) / 10), 1), shuffle=False)
        testloader = DataLoader(DatasetSplit(dataset, idxs_test),
                                batch_size=max(int(len(idxs_test) / 10), 1), shuffle=False)
        return trainloader, validloader, testloader

    def update_weights(self, model, global_round, data):
        # get data of node
        node = data["nodes"][self.user_idx]

        # Set model to train mode
        model.train()
        epoch_loss = []

        # Set optimizer for the local updates
        if self.args.optimizer == 'sgd':
            optimizer = torch.optim.SGD(model.parameters(), lr=self.args.lr,
                                        momentum=0.5)
        elif self.args.optimizer == 'adam':
            optimizer = torch.optim.Adam(model.parameters(), lr=self.args.lr,
                                         weight_decay=1e-4)

        for iter in range(self.args.local_ep):
            batch_loss = []
            for batch_idx, (images, labels) in enumerate(self.trainloader):
                images, labels = images.to(self.device), labels.to(self.device)

                model.zero_grad()
                log_probs = model(images)
                if self.args.dataset == 'av':
                    # For binary classification
                    outputs = log_probs.squeeze()
                    loss = self.criterion(outputs, labels.float())
                else:
                    # For multi-class classification
                    loss = self.criterion(log_probs, labels)
                loss.backward()
                optimizer.step()

                if self.args.verbose and (batch_idx % 10 == 0):
                    print('| Global Round : {} |user {} |Local Epoch : {} | [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                        global_round, self.user_idx, iter, batch_idx * len(images),
                        len(self.trainloader.dataset), 100. * batch_idx / len(self.trainloader), loss.item()))
                    node["current_epoch"] = iter + 1
                    node["total_dataset_size"] = len(self.trainloader)
                    node["current_dataset_size"] = batch_idx
                    node["loss"] = loss.item()
                # self.logger.add_scalar('loss', loss.item(), global_round * self.args.local_ep + iter)
                batch_loss.append(loss.item())
            epoch_loss.append(sum(batch_loss) / len(batch_loss))

        return model.state_dict(), sum(epoch_loss) / len(epoch_loss)

    def inference(self, model):
        """ Returns the inference accuracy and loss.
        """

        model.eval()
        loss, total, correct = 0.0, 0.0, 0.0

        for batch_idx, (images, labels) in enumerate(self.testloader):
            images, labels = images.to(self.device), labels.to(self.device)

            # Inference
            outputs = model(images)
            if self.args.dataset == 'av':
                outputs = outputs.squeeze()
                batch_loss = self.criterion(outputs, labels.float())
                pred_labels = (outputs > 0.5).float()
            else:
                batch_loss = self.criterion(outputs, labels)
                _, pred_labels = torch.max(outputs, 1)
                pred_labels = pred_labels.view(-1)
            loss += batch_loss.item()
            correct += torch.sum(torch.eq(pred_labels, labels)).item()
            total += len(labels)

        accuracy = correct / total
        return accuracy, loss


def test_inference(args, model, test_dataset):
    """ Returns the test accuracy, precision, and recall.
    """
    model.eval()
    loss, total, correct = 0.0, 0.0, 0.0
    true_positive, false_positive, false_negative = 0.0, 0.0, 0.0
    device = 'cuda' if args.gpu else 'cpu'

    if args.dataset == 'av':
        # Binary classification
        criterion = torch.nn.BCELoss().to(device)
    else:
        # Multi-class classification
        criterion = torch.nn.NLLLoss().to(device)

    testloader = DataLoader(test_dataset, batch_size=128, shuffle=False)

    for batch_idx, (images, labels) in enumerate(testloader):
        images, labels = images.to(device), labels.to(device)

        # Inference
        outputs = model(images)
        if args.dataset == 'av':
            # Binary classification
            outputs = outputs.squeeze()
            batch_loss = criterion(outputs, labels.float())
            pred_labels = (outputs > 0.5).float()

            # Calculate TP, FP, FN for binary classification
            true_positive += torch.sum((pred_labels == 1) & (labels == 1)).item()
            false_positive += torch.sum((pred_labels == 1) & (labels == 0)).item()
            false_negative += torch.sum((pred_labels == 0) & (labels == 1)).item()

        else:
            # Multi-class classification
            batch_loss = criterion(outputs, labels)
            _, pred_labels = torch.max(outputs, 1)
            pred_labels = pred_labels.view(-1)

            # Calculate TP, FP, FN for each class
            for cls in range(args.num_classes):
                true_positive += torch.sum((pred_labels == cls) & (labels == cls)).item()
                false_positive += torch.sum((pred_labels == cls) & (labels != cls)).item()
                false_negative += torch.sum((pred_labels != cls) & (labels == cls)).item()

        loss += batch_loss.item()
        correct += torch.sum(torch.eq(pred_labels, labels)).item()
        total += len(labels)

    # Accuracy
    accuracy = correct / total

    # Precision and Recall
    if true_positive + false_positive > 0:
        precision = true_positive / (true_positive + false_positive)
    else:
        precision = 0.0

    if true_positive + false_negative > 0:
        recall = true_positive / (true_positive + false_negative)
    else:
        recall = 0.0

    return accuracy, precision, recall, loss
