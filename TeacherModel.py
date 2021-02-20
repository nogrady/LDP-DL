import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.autograd import Variable
import torchvision.models as models
import time
import random

device = torch.device("cuda:3" if torch.cuda.is_available() else "cpu")
# specific the dataset
dataset = "CIFAR10"

if dataset == "CIFAR10":
    imgTransform = transforms.Compose([transforms.Scale((224, 224)),
                               transforms.ToTensor(),
                               transforms.Normalize((0.4914, 0.4822, 0.4465),
                                                    (0.2023, 0.1994, 0.2010))])
    trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=imgTransform)
    testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=imgTransform)
    classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
    loadIndexPath = "./label_index_" + dataset +"/"
    saveModelPath = "./teacherModels_" + dataset + "/cifar_private_net_"
    print("Training teacher models on " + str(dataset))

elif dataset == "MNIST":
    imgTransform = transforms.Compose([transforms.Scale((224, 224)),  # Grayscale image
                                       transforms.Grayscale(num_output_channels=3), transforms.ToTensor(),
                                       transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                            std=[0.229, 0.224, 0.225])])

    trainset = torchvision.datasets.MNIST(root='./MNISTDataset', train=True, download=True, transform=imgTransform)
    testset = torchvision.datasets.MNIST(root='./MNISTDataset', train=False, download=True, transform=imgTransform)
    classes = ('0', '1', '2', '3', '4', '5', '6', '7', '8', '9')

    loadIndexPath = "./label_index_" + dataset +"/"
    saveModelPath = "./teacherModels_" + dataset + "/mnist_private_net_"
    print("Training teacher models on " + str(dataset))

elif dataset == "FashionMNIST":
    imgTransform = transforms.Compose([transforms.Scale((224, 224)),
                                       transforms.Grayscale(num_output_channels=3), transforms.ToTensor(),
                                       transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                            std=[0.229, 0.224, 0.225])])

    trainset = torchvision.datasets.FashionMNIST(root='./FashionMNISTataset', train=True, download=True,
                                                 transform=imgTransform)
    testset = torchvision.datasets.FashionMNIST(root='./FashionMNISTataset', train=False, download=True,
                                                transform=imgTransform)
    loadIndexPath = "./label_index_" + dataset + "/"
    saveModelPath = "./teacherModels_" + dataset + "/FashionMnist_private_net_"
    print("Training teacher models on " + str(dataset))

n_private = 200  # n_private data owners, 1 data user

def train_teacher_models():
    for j in range(50, n_private):
        print("model", j)
        start_time = time.time()

        # load training data
        one_index = np.loadtxt(loadIndexPath + "private_train" + str(j), dtype='str', delimiter=",")
        indices = []

        for oi in one_index:
            indices.append(int(oi))    # removed "" from input str then append to list

        random.shuffle(indices)

        sampler = torch.utils.data.SubsetRandomSampler(indices)
        trainloader = torch.utils.data.DataLoader(trainset, batch_size=32, sampler=sampler, shuffle=False, num_workers=2)

        # load testing data
        one_index = np.loadtxt(loadIndexPath + "private_test" + str(j), dtype='str', delimiter=",")
        indices = []

        for oi in one_index:
            indices.append(int(oi))

        random.shuffle(indices)

        sampler = torch.utils.data.SubsetRandomSampler(indices)
        testloader = torch.utils.data.DataLoader(testset, batch_size=32, sampler=sampler, shuffle=False, num_workers=2)

        # train teacher models
        net = models.resnet50(pretrained=True)
        num_ftrs = net.fc.in_features
        net.fc = nn.Linear(num_ftrs, 10)
        net.to(device)
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

        for epoch in range(50):  # loop over the dataset multiple times
            running_loss = 0.0
            for i, data in enumerate(trainloader, 0):
                # get the inputs; data is a list of [inputs, labels]
                inputs, labels = data[0].to(device), data[1].to(device)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward + backward + optimize
                outputs = net(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

                # print statistics
                running_loss += loss.item()
                if i % 10 == 9:  # print every 10 mini-batches
                    # print('[%d, %5d] loss: %.8f' %
                    #       (epoch + 1, i + 1, running_loss / 10))
                    running_loss = 0.0

        print('Finished Training')
        # save the trained teacher model
        PATH = saveModelPath + str(j) + '.pth'
        torch.save(net.state_dict(), PATH)

        # Make a pass over the validation data.
        correct = 0.0
        cum_loss = 0.0
        counter = 0
        net.eval()  # This is important to call before evaluating!
        for (i, data) in enumerate(testloader):
            inputs, labels = data[0].to(device), data[1].to(device)
            # Wrap inputs, and targets into torch.autograd.Variable types.
            inputs = Variable(inputs)
            labels = Variable(labels)

            # Forward pass:
            outputs = net(inputs)
            loss = criterion(outputs, labels)

            # logging information.
            cum_loss += loss.item()
            max_scores, max_labels = torch.max(outputs.data, 1)
            correct += (max_labels == labels).sum().item()
            counter += labels.size(0)
        # print('Accuracy of the network on the ' + str(len(testsets[-1])) + ' test images: %d %%' % (
        #             100 * correct / counter))
        print('Accuracy of the network on the ' + str(len(indices)) + ' test images: %d %%' % (
                     100 * correct / counter))
        print("--- %s seconds ---" % (time.time() - start_time))

if __name__ == "__main__":
    train_teacher_models()