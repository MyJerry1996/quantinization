import torch
import torchvision
from torch import optim
from model import LeNet
import torch.nn as nn
import torch.nn.functional as F
import torchvision.datasets as dataset
import torchvision.utils as utils 
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
import os


train_data = dataset.MNIST(root = "mnist",
                           train = True,
                           transform = transforms.ToTensor(),
                           download = True)

train_loader = DataLoader(dataset=train_data, 
                          batch_size=2, 
                          shuffle=True)

test_data = dataset.MNIST(root = "mnist",
                           train = False,
                           transform = transforms.ToTensor(),
                           download = True)

test_loader = torch.utils.data.DataLoader(dataset = test_data,
                                  batch_size = 64,
                                  shuffle = True)


def train_epoch(net, optimizer, criterion):
    epoch_loss = 0.0
    epoch_acc = 0.0
    for i, (img, label) in tqdm(enumerate(train_loader)):
        img, label = img.to(device), label.to(device)
        out = net(img)
        optimizer.zero_grad()
        loss = criterion(out, label)
        loss.backward()
        optimizer.step()
        pred = torch.argmax(out, dim=1)
        acc = torch.sum(pred == label)
        epoch_loss += loss.item()
        epoch_acc += acc.item()
        
    epoch_acc /= len(train_data)
    epoch_loss /= len(train_loader)
    print("epoch loss:{:6f} epoch acc:{:6f}".format(epoch_loss, epoch_acc))
    return epoch_acc, epoch_loss, net


def valid(net, criterion):
    with torch.no_grad():
        test_loss = 0.0
        test_acc = 0.0
        for i, (img, label) in tqdm(enumerate(test_loader)):
            img, label = img.to(device), label.to(device)
            out = net(img)
            loss = criterion(out, label)
            pred = torch.argmax(out, dim=1)
            acc = torch.sum(pred == label)
            test_loss += loss.item()
            test_acc += acc.item()
            
        test_acc /= len(test_data)
        test_loss /= len(test_loader)
        print("test loss:{:6f} test acc:{:6f}".format(test_loss, test_acc))
        return test_acc, test_loss


def init_train(net):
    checkpoint = "./checkpoints/"
    if os.path.exists(os.path.join(checkpoint, "best.pth")):
        save_model = torch.load(os.path.join(checkpoint, "best.pth"));
        net.load_state_dict(save_model['net'])
        if save_model['best_accuracy'] > 0.9:
            print('break init train')
            return
        best_accuracy = save_model['best_accuracy']
        best_loss = save_model['best_loss']
    else:
        best_accuracy = 0.0
        best_loss = 10.0
    
#     writer = SummaryWriter('logs/')
    criteron = torch.nn.CrossEntropyLoss()
    
    optimizer = optim.SGD(net.parameters(), lr=0.001, weight_decay=0.0001, momentum=0.9)
    num_epoch = 100
    lr = 0.001
    for epoch in range(num_epoch):
        print('epoch: {}'.format(epoch))
        epoch_acc, epoch_loss, net = train_epoch(net, optimizer, criteron)
#         writer.add_scalar('epoch_acc', epoch_acc,
#                           sum([e[0] for e in init_epoch_lr[:i]]) + epoch)
#         writer.add_scalar('epoch_loss', epoch_loss,
#                           sum([e[0] for e in init_epoch_lr[:i]]) + epoch)

        test_acc, test_loss = valid(net, criteron)
        if test_loss <= best_loss:
            if test_acc >= best_accuracy:
                best_accuracy = test_acc

            best_loss = test_loss
            best_model_weights = net.state_dict().copy()
            best_model_params = optimizer.state_dict().copy()
            torch.save(
                {
                    'net': best_model_weights,
                    'optimizer': best_model_params,
                    'best_accuracy': best_accuracy,
                    'best_loss': best_loss
                },
                os.path.join(checkpoint, 'best_model.pth')
            )

#         writer.add_scalar('test_acc', test_acc,
#                           sum([e[0] for e in init_epoch_lr[:i]]) + epoch)
#         writer.add_scalar('test_loss', test_loss,
#                           sum([e[0] for e in init_epoch_lr[:i]]) + epoch)

#     writer.close()
    return net


if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    net = LeNet().to(device)
    init_train(net)