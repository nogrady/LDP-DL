#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from torch.autograd import Variable
import torchvision.models as models
import math
from joblib import Parallel, delayed
from scipy.special import comb
import random

n_para_jobs = 4

# LDP mechanism
# Laplace mechanism
def getNoisyAns_Lap(x_train, epsilon):
    loc = 0
    n_features = x_train.shape[0]
    scale = 2 * n_features / epsilon
    s = np.random.laplace(loc, scale, x_train.shape)
    x_train_noisy = x_train + s
    return x_train_noisy

# Duchi's mechanism for one-dimensional and multi-dimensional data
def Duchi_1d(t_i, eps, t_star):
    p = (math.exp(eps) - 1) / (2 * math.exp(eps) + 2) * t_i + 0.5
    coin = np.random.binomial(1, p)
    if coin == 1:
        return t_star[1]
    else:
        return t_star[0]

def getNoisyAns_Duchi_1d(x_train, eps):
    assert np.amin(x_train) >= -1.0 and np.amax(x_train) <= 1.0
    assert x_train.ndim == 1
    tmp = (math.exp(eps) + 1) / (math.exp(eps) - 1)
    t_star = [-tmp, tmp]
    x_train_Duchi_1d_list = Parallel(n_jobs=n_para_jobs)(
        delayed(Duchi_1d)(t_i, eps, t_star) for t_i in x_train)
    x_train_Duchi_1d_array = np.asarray(x_train_Duchi_1d_list)
    return x_train_Duchi_1d_array

def Duchi_md(t_i, eps):
    n_features = len(t_i)
    if n_features % 2 != 0:
        C_d = pow(2, n_features - 1) / comb(n_features - 1, (n_features - 1) / 2)
    else:
        C_d = (pow(2, n_features - 1) + 0.5 * comb(n_features, n_features / 2)) / comb(n_features - 1, n_features / 2)

    B = C_d * (math.exp(eps) + 1) / (math.exp(eps) - 1)
    v = []
    for tmp in t_i:
        tmp_p = 0.5 + 0.5 * tmp
        tmp_q = 0.5 - 0.5 * tmp
        v.append(np.random.choice([1, -1], p=[tmp_p, tmp_q]))
    bernoulli_p = math.exp(eps) / (math.exp(eps) + 1)
    coin = np.random.binomial(1, bernoulli_p)

    t_star = np.random.choice([-B, B], len(t_i), p=[0.5, 0.5])
    v_times_t_star = np.multiply(v, t_star)
    sum_v_times_t_star = np.sum(v_times_t_star)
    if coin == 1:
        while sum_v_times_t_star <= 0:
            t_star = np.random.choice([-B, B], len(t_i), p=[0.5, 0.5])
            v_times_t_star = np.multiply(v, t_star)
            sum_v_times_t_star = np.sum(v_times_t_star)
    else:
        while sum_v_times_t_star > 0:
            t_star = np.random.choice([-B, B], len(t_i), p=[0.5, 0.5])
            v_times_t_star = np.multiply(v, t_star)
            sum_v_times_t_star = np.sum(v_times_t_star)
    return t_star.reshape(-1)

def getNoisyAns_Duchi_md(x_train, eps):
    assert np.amin(x_train) >= -1.0 and np.amax(x_train) <= 1.0
    assert x_train.ndim > 1
    n_features = x_train.shape[1]
    if n_features % 2 != 0:
        C_d = pow(2, n_features - 1) / comb(n_features - 1, (n_features - 1) / 2)
    else:
        C_d = (pow(2, n_features - 1) + 0.5 * comb(n_features, n_features / 2)) / comb(n_features - 1, n_features / 2)

    B = C_d * (math.exp(eps) + 1) / (math.exp(eps) - 1)
    x_train_Duchi_md_list = Parallel(n_jobs=n_para_jobs)(
        delayed(Duchi_md)(t_i, eps, B) for t_i in x_train)
    x_train_Duchi_md_array = np.asarray(x_train_Duchi_md_list)
    return x_train_Duchi_md_array

# Piecewise mechanism for one-dimensional and multi-dimensional data
def PM_1d(t_i, eps):
    C = (math.exp(eps / 2) + 1) / (math.exp(eps / 2) - 1)
    l_t_i = (C + 1) * t_i / 2 - (C - 1) / 2
    r_t_i = l_t_i + C - 1
    # provide 'size' parameter in uniform() would result in a ndarray
    x = np.random.uniform(0, 1)
    threshold = math.exp(eps / 2) / (math.exp(eps / 2) + 1)
    if x < threshold:
        t_star = np.random.uniform(l_t_i, r_t_i)
    else:
        tmp_l = np.random.uniform(-C, l_t_i)
        tmp_r = np.random.uniform(r_t_i, C)
        w = np.random.randint(2)
        t_star = (1 - w) * tmp_l + w * tmp_r
    # print("PM_1d t-star: %.3f" % t_star)
    return t_star

def PM_md(t_i, eps):
    n_features = len(t_i)
    k = max(1, min(n_features, int(eps / 2.5)))
    rand_features = np.random.randint(0, n_features, size=k)
    res = np.zeros(t_i.shape)
    for j in rand_features:
        res[j] = (n_features * 1.0 / k) * PM_1d(t_i[j], eps / k)
    return res

def getNoisyAns_PM_1d(x_train, eps):
    assert np.amin(x_train) >= -1.0 and np.amax(x_train) <= 1.0
    assert x_train.ndim == 1
    x_train_PM_1d_list = Parallel(n_jobs=n_para_jobs)(
        delayed(PM_1d)(t_i, eps) for t_i in x_train)
    x_train_PM_1d_array = np.asarray(x_train_PM_1d_list)
    return x_train_PM_1d_array

def getNoisyAns_PM_md(x_train, eps):
    assert np.amin(x_train) >= -1.0 and np.amax(x_train) <= 1.0
    assert x_train.ndim > 1
    x_train_PM_md_list = Parallel(n_jobs=n_para_jobs)(
        delayed(PM_md)(t_i, eps) for t_i in x_train)
    x_train_PM_md_array = np.asarray(x_train_PM_md_list)
    return x_train_PM_md_array

# specific running device
device = torch.device("cuda:2" if torch.cuda.is_available() else "cpu")

# specific dataset
dataset = "CIFAR10"

if dataset == "CIFAR10":
    imgTransform = transforms.Compose([transforms.Scale((224, 224)),
                               transforms.ToTensor(),
                               transforms.Normalize((0.4914, 0.4822, 0.4465),
                                                    (0.2023, 0.1994, 0.2010))])
    trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=imgTransform)
    testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=imgTransform)
    classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
    loadIndexPath = "./label_index_" + dataset + "/"
    saveModelPath = "./teacherModels_" + dataset + "/cifar_private_net_"
    print("Test on CIFAR10 dataset")

elif dataset == "MNIST":
    imgTransform = transforms.Compose([transforms.Scale((224, 224)),  # Grayscale image
                                       transforms.Grayscale(num_output_channels=3), transforms.ToTensor(),
                                       transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                            std=[0.229, 0.224, 0.225])])
    trainset = torchvision.datasets.MNIST(root='./MNISTDataset', train=True, download=True, transform=imgTransform)
    testset = torchvision.datasets.MNIST(root='./MNISTDataset', train=False, download=True, transform=imgTransform)
    loadIndexPath = "./label_index_" + dataset + "/"
    saveModelPath = "./teacherModels_" + dataset + "/mnist_private_net_"
    classes = ('0', '1', '2', '3', '4', '5', '6', '7', '8', '9')
    print("Test on MNIST dataset")

elif dataset == 'FashionMNIST':
    imgTransform = transforms.Compose([transforms.Scale((224, 224)),
                                       transforms.Grayscale(num_output_channels=3), transforms.ToTensor(),
                                       transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                            std=[0.229, 0.224, 0.225])])
    trainset = torchvision.datasets.FashionMNIST(root='./FashionMNISTataset', train=True, download=True, transform=imgTransform)
    testset = torchvision.datasets.FashionMNIST(root='./FashionMNISTataset', train=False, download=True, transform=imgTransform)
    classes = ('0', '1', '2', '3', '4', '5', '6', '7', '8', '9')
    loadIndexPath = "./label_index_" + dataset + "/"
    saveModelPath = "./teacherModels_" + dataset + "/FashionMnist_private_net_"
    print("Test on FashionMNIST dataset")

public_rate = 0.02
num_train = len(trainset)
num_test = len(testset)
num_train_public = int(num_train * public_rate)
num_test_public = int(num_test * public_rate)
num_train_private = num_train - num_train_public
num_test_private = num_test - num_test_public

#n_private = 30;  # n_private data owners, 1 data user
#
# num_sets_train = np.empty(n_private + 1)
# num_sets_train.fill(int(num_train_private / n_private))
# num_sets_train[n_private - 1] = int(num_train_private / n_private) + num_train_private - int(
#     num_train_private / n_private) * n_private
# num_sets_train[-1] = num_train_public
#
# num_sets_test = np.empty(n_private + 1)
# num_sets_test.fill(int(num_test_private / n_private))
# num_sets_test[n_private - 1] = int(num_test_private / n_private) + num_test_private - int(
#     num_test_private / n_private) * n_private
# num_sets_test[-1] = num_test_public

def fetch_teacher_outputs(teacher_model, dataloader):
    # set teacher_model to evaluation mode
    teacher_model.eval()
    teacher_outputs = []

    teacher_outputs_labels = []

    criterion = nn.CrossEntropyLoss()
    correct = 0.0
    cum_loss = 0.0
    counter = 0

    for i, data in enumerate(dataloader, 0):
        data_batch, labels_batch = data[0].to(device), data[1].to(device)
        data_batch, labels_batch = Variable(data_batch), Variable(labels_batch)
        output_teacher_batch = teacher_model(data_batch)

        loss = criterion(output_teacher_batch, labels_batch)
        cum_loss += loss.item()
        max_scores, max_labels = torch.max(output_teacher_batch.data, 1)
        teacher_outputs_labels.append(max_labels.data.cpu().numpy())

        correct += (max_labels == labels_batch).sum().item()
        counter += labels_batch.size(0)
        output_teacher_batch = output_teacher_batch.data.cpu().numpy()
        teacher_outputs.append(output_teacher_batch)
    #print('Accuracy of the network on the public train images: %d %%' % (100 * correct / counter))
    return teacher_outputs, teacher_outputs_labels

def loss_fn_kd(outputs, labels, teacher_outputs, teacher_outputs_labels, alpha, T, beta):
    """
    Compute the knowledge-distillation (KD) loss given outputs, labels.
    "Hyperparameters": temperature and alpha

    NOTE: the KL Divergence for PyTorch comparing the softmaxs of teacher
    and student expects the input tensor to be log probabilities! See Issue #2
    """
    KD_loss_1 = nn.KLDivLoss()(F.log_softmax(outputs / T, dim=1), F.softmax(teacher_outputs / T, dim=1)) * (
                T * T) * beta

    return KD_loss_1 * alpha

# training student model:
"""
# n_private_total: number of total data owner (teacher) participated
# n_private: number of data owner (teacher) queried each iteration
# eps: total privacy budget
# alg: LDP mechanism (Lap, Duchi, PM)
# it: number of training iteration
# batchsize: training batchsize
# aqsit: number of iteration of active query sampling
# num_data_per_aqs: number of iteration of active query sampling
"""
def train_student_models_AQS(n_private_total, n_private, eps, alg, it, batchsize, aqsit, num_data_per_aqs):

    weight_train = np.empty(len(trainset))
    weight_train.fill(0)

    one_index = np.loadtxt(loadIndexPath + "public_train", dtype='str', delimiter=",")
    indices = []

    for oi in one_index:
        indices.append(int(oi))

    allindices = indices
    # randomly select first aqs training data
    trainIndices = np.random.choice(allindices, num_data_per_aqs, replace=False).tolist()
    resIndices = [k for k in allindices if k not in trainIndices]

    random.shuffle(trainIndices)
    sampler = torch.utils.data.SubsetRandomSampler(trainIndices)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batchsize, sampler=sampler, shuffle=False, num_workers=2)

    weight_test = np.empty(len(testset))
    weight_test.fill(0)
    one_index = np.loadtxt(loadIndexPath + "public_test", dtype='str', delimiter=",")
    indices = []

    for oi in one_index:
        indices.append(int(oi))

    random.shuffle(indices)
    sampler = torch.utils.data.SubsetRandomSampler(indices)
    testloader = torch.utils.data.DataLoader(testset, batch_size=batchsize, sampler=sampler, shuffle=False, num_workers=2)

    beta = 10
    alpha = 1.0
    T = 1.0
    eps = eps

    snet = models.resnet18(pretrained=True)
    num_ftrs = snet.fc.in_features
    snet.fc = nn.Linear(num_ftrs, 10)
    snet.to(device)

    # train student model using active query sampling
    while aqsit != 0:
        for j in range(0, 1):
            rnd = np.random.randint(0, high=n_private_total, size=1, dtype='l')
            PATH = saveModelPath + str(rnd[0]) + '.pth'

            net = models.resnet50(pretrained=True)
            num_ftrs = net.fc.in_features
            net.fc = nn.Linear(num_ftrs, 10)
            net.to(device)
            net.load_state_dict(torch.load(PATH))

            net.to(device)
            teacher_outputs, teacher_outputs_labels = fetch_teacher_outputs(net, trainloader)

            teacher_outputs_labels_vote = []
            for i in range(len(teacher_outputs_labels)):
                teacher_outputs_labels_vote_single_batch = []
                for k in range(len(teacher_outputs_labels[i])):
                    dic = np.zeros(10)
                    dic[teacher_outputs_labels[i][k]] = 1
                    teacher_outputs_labels_vote_single_batch.append(dic.tolist())
                teacher_outputs_labels_vote.append(np.array(teacher_outputs_labels_vote_single_batch, dtype=float))

            # apply LDP mechanism
            for i in range(len(teacher_outputs)):
                for k in range(len(teacher_outputs[i])):
                    max_v = np.amax(teacher_outputs[i][k])
                    min_v = np.amin(teacher_outputs[i][k])
                    teacher_outputs[i][k] = (teacher_outputs[i][k] - min_v) / (max_v - min_v) * 2.0 - 1.0
                    if alg == "PM":
                        teacher_outputs[i][k] = PM_md(teacher_outputs[i][k], eps)
                    if alg == "Lap":
                        teacher_outputs[i][k] = getNoisyAns_Lap(teacher_outputs[i][k], eps)
                    if alg == "Duchi":
                        teacher_outputs[i][k] = Duchi_md(teacher_outputs[i][k], eps)
                    teacher_outputs[i][k] = (1.0 + teacher_outputs[i][k]) / 2.0 * (max_v - min_v) + min_v

        if n_private > 1:
            rndlist = np.random.randint(0, high=n_private_total, size=n_private-1, dtype='l')
            while rnd in rndlist:
                rndlist = np.random.randint(0, high=n_private_total, size=n_private-1, dtype='l')
            for j in rndlist:
                PATH = saveModelPath + str(int(j)) + '.pth'
                net = models.resnet50(pretrained=True)

                num_ftrs = net.fc.in_features
                net.fc = nn.Linear(num_ftrs, 10)
                net.to(device)
                net.load_state_dict(torch.load(PATH))

                net.to(device)
                teacher_outputs_, teacher_outputs_labels_ = fetch_teacher_outputs(net, trainloader)

                # apply LDP mechanism
                for i in range(len(teacher_outputs_)):
                    for k in range(len(teacher_outputs_[i])):
                        max_v = np.amax(teacher_outputs_[i][k])
                        min_v = np.amin(teacher_outputs_[i][k])
                        teacher_outputs_[i][k] = (teacher_outputs_[i][k] - min_v) / (max_v - min_v) * 2.0 - 1.0
                        if alg == "PM":
                            teacher_outputs_[i][k] = PM_md(teacher_outputs_[i][k], eps)
                        if alg == "Lap":
                            teacher_outputs_[i][k] = getNoisyAns_Lap(teacher_outputs_[i][k], eps)
                        if alg == "Duchi":
                            teacher_outputs_[i][k] = Duchi_md(teacher_outputs_[i][k], eps)
                        teacher_outputs_[i][k] = (1.0 + teacher_outputs_[i][k]) / 2.0 * (max_v - min_v) + min_v

                for i in range(len(teacher_outputs)):
                    for k in range(len(teacher_outputs[i])):
                        teacher_outputs[i][k] = teacher_outputs[i][k] + teacher_outputs_[i][k]

                for i in range(len(teacher_outputs_labels_)):
                    for k in range(len(teacher_outputs_labels_[i])):
                        teacher_outputs_labels_vote[i][k][teacher_outputs_labels_[i][k]] = \
                        teacher_outputs_labels_vote[i][k][teacher_outputs_labels_[i][k]] + 1.

        for i in range(len(teacher_outputs)):
            for k in range(len(teacher_outputs[i])):
                teacher_outputs[i][k] = teacher_outputs[i][k] / float(n_private)

        teacher_outputs_labels_vote_scalar = []
        for i in range(len(teacher_outputs_labels_vote)):
            teacher_outputs_labels_vote_scalar_batch = []
            for k in range(len(teacher_outputs_labels_vote[i])):
                teacher_outputs_labels_vote_scalar_batch.append(np.argmax(teacher_outputs_labels_vote[i][k], axis=0))
            teacher_outputs_labels_vote_scalar.append(np.array(teacher_outputs_labels_vote_scalar_batch))

        # train student model
        snet = models.resnet18(pretrained=True)
        num_ftrs = snet.fc.in_features
        snet.fc = nn.Linear(num_ftrs, 10)
        snet.to(device)

        snet.train()
        optimizer = optim.SGD(snet.parameters(), lr=0.001, momentum=0.9)

        for epoch in range(it):  # loop over the dataset multiple times
            running_loss = 0.0
            for i, data in enumerate(trainloader, 0):
                # get the inputs; data is a list of [inputs, labels]
                inputs, labels = data[0].to(device), data[1].to(device)
                inputs, labels = Variable(inputs), Variable(labels)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward + backward + optimize
                outputs = snet(inputs)

                output_teacher_batch = torch.from_numpy(teacher_outputs[i])
                output_teacher_batch_labels = torch.from_numpy(teacher_outputs_labels_vote_scalar[i])

                output_teacher_batch = output_teacher_batch.to(device)
                output_teacher_batch_labels = output_teacher_batch_labels.to(device)

                output_teacher_batch = Variable(output_teacher_batch, requires_grad=False)
                output_teacher_batch_labels = Variable(output_teacher_batch_labels, requires_grad=False)

                loss = loss_fn_kd(outputs, labels, output_teacher_batch, output_teacher_batch_labels, alpha, T, beta)
                loss.backward()
                optimizer.step()

        print('Finished Training with ' + str(len(trainIndices)) + " data")

        # test the rest of public training data on student model
        random.shuffle(resIndices)
        sampler = torch.utils.data.SubsetRandomSampler(resIndices)
        tmp_testloader = torch.utils.data.DataLoader(trainset, batch_size=batchsize, sampler=sampler, shuffle=False, num_workers=2)

        snet.eval()  # This is important to call before evaluating!
        student_outputs_softmax = []
        criterion = nn.CrossEntropyLoss()

        for (i, data) in enumerate(tmp_testloader):

            inputs, labels = data[0].to(device), data[1].to(device)
            # Wrap inputs, and targets into torch.autograd.Variable types.
            inputs = Variable(inputs)
            labels = Variable(labels)

            # Forward pass:
            outputs = snet(inputs)
            m = nn.Softmax(dim=1)
            output_softmax = m(outputs)             #softmax function to probability, sum to 1

            output_softmax = output_softmax.data.cpu().numpy()
            student_outputs_softmax.append(output_softmax)

        # add next active query sampling data into training data for retrain
        allConfidence = []
        for i in range(len(student_outputs_softmax)):
            for k in range(len(student_outputs_softmax[i])):
                confident = 0
                maxP = max(student_outputs_softmax[i][k])     #largest value
                confidentVector = student_outputs_softmax[i][k]
                for j in range(len(confidentVector)):
                    confident = confident + maxP - confidentVector[j]
                allConfidence.append(confident/(len(confidentVector)-1))

        topScoreIndex = sorted(range(len(allConfidence)), key=lambda i: allConfidence[i])[:num_data_per_aqs]

        for i in topScoreIndex:
            trainIndices.append(resIndices[i])

        resIndices = [k for k in allindices if k not in trainIndices]

        #print("len(trainIndices) for next train", len(trainIndices))
        random.shuffle(trainIndices)
        sampler = torch.utils.data.SubsetRandomSampler(trainIndices)
        trainloader = torch.utils.data.DataLoader(trainset, batch_size=batchsize, sampler=sampler, shuffle=False, num_workers=2)
        aqsit = aqsit - 1

    # evaluate the final student model
    correct = 0.0
    cum_loss = 0.0
    counter = 0
    snet.eval()  # This is important to call before evaluating!
    criterion = nn.CrossEntropyLoss()
    for (i, data) in enumerate(testloader):
        inputs, labels = data[0].to(device), data[1].to(device)
        # Wrap inputs, and targets into torch.autograd.Variable types.
        inputs = Variable(inputs)
        labels = Variable(labels)

        # Forward pass:
        outputs = snet(inputs)
        loss = criterion(outputs, labels)

        # logging information.
        cum_loss += loss.item()
        max_scores, max_labels = torch.max(outputs.data, 1)
        correct += (max_labels == labels).sum().item()
        counter += labels.size(0)

    return (100 * correct / counter)

def main():
    # define the parameter
    alg = "PM"
    n_private_total = 55
    n_private = 10
    total_Budget = 5
    num_dataowner = 10000
    qurey_perdataowner = n_private * num_train_public / num_dataowner
    eps = total_Budget / qurey_perdataowner
    it = 20
    batchsize = 16
    aqsit = 5
    num_data_per_aqs = 200

    acc = train_student_models_AQS(n_private_total, n_private, eps, alg, it, batchsize, aqsit, num_data_per_aqs)
    print("student model acc:" + str(acc))

if __name__ == "__main__":
    main()