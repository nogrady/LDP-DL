
import torchvision
import torchvision.transforms as transforms
import numpy as np

dataset = "CIFAR10"

outputPath = "./label_index_" + dataset + "/"

n_private = 200  # number of data owners, 1 data user
TrainSize = 4000 # private training size
TestSize = 800   # private testing size
public_rate = 0.1 # ratio of public training and testing data

if dataset == "FashionMNIST":
    imgTransform = transforms.Compose([transforms.Scale((224, 224)),
                                       transforms.Grayscale(num_output_channels=3), transforms.ToTensor(),
                                       transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                            std=[0.229, 0.224, 0.225])])

    trainset = torchvision.datasets.FashionMNIST(root='./FashionMNISTataset', train=True, download=True, transform=imgTransform)
    testset = torchvision.datasets.FashionMNIST(root='./FashionMNISTataset', train=False, download=True, transform=imgTransform)

if dataset == "CIFAR10":
    imgTransform = transforms.Compose([transforms.Scale((224, 224)),
                                       transforms.ToTensor(),
                                       transforms.Normalize((0.4914, 0.4822, 0.4465),
                                                            (0.2023, 0.1994, 0.2010))])
    trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=imgTransform)
    testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=imgTransform)

if dataset == "MNIST":
    imgTransform = transforms.Compose([transforms.Scale((224, 224)),                                #Grayscale image
                                       transforms.Grayscale(num_output_channels=3), transforms.ToTensor(),
                                       transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                            std=[0.229, 0.224, 0.225])])

    trainset = torchvision.datasets.MNIST(root='./MNISTDataset', train=True, download=True, transform=imgTransform)
    testset = torchvision.datasets.MNIST(root='./MNISTDataset', train=False, download=True, transform=imgTransform)

labeltoIndex_train = dict()
labeltoIndex_test = dict()

n_class = 10

for i in range(len(trainset)):
    label = trainset.__getitem__(i)[1]
    if label in labeltoIndex_train:
        labeltoIndex_train[label].append(i)
    else:
        labeltoIndex_train[label] = [i]

# print("label for each train class")
# for key in labeltoIndex_train.keys():
#     print(len(labeltoIndex_train[key]))

for i in range(len(testset)):
    label = testset.__getitem__(i)[1]
    if label in labeltoIndex_test:
        labeltoIndex_test[label].append(i)
    else:
        labeltoIndex_test[label] = [i]


num_train = len(trainset)
num_test = len(testset)
num_train_public = int(num_train * public_rate)
num_test_public = int(num_test * public_rate)
num_train_private = num_train - num_train_public
num_test_private = num_test - num_test_public

##select public data
print("Sampling private data")
selection = []
for key in labeltoIndex_train.keys():
    tmp = list(np.random.choice(labeltoIndex_train[key], size=int(num_train_public/n_class), replace=False))
    res = [i for i in labeltoIndex_train[key] if i not in tmp]
    selection = selection + tmp
    labeltoIndex_train[key] = res

print("num of public training:", len(selection))

with open(outputPath + "public_train", 'w') as myfile:
    for k in range(len(selection)):
        if k == len(selection) - 1:
            myfile.write(str(selection[k]))
        else:
            myfile.write(str(selection[k]) + ",")

selection = []
for key in labeltoIndex_test.keys():
    tmp = list(np.random.choice(labeltoIndex_test[key], size=int(num_test_public/n_class), replace=False))
    res = [i for i in labeltoIndex_test[key] if i not in tmp]
    selection = selection + tmp
    labeltoIndex_test[key] = res

print("num of public testing:", len(selection))

with open(outputPath + "public_test", 'w') as myfile:
    for k in range(len(selection)):
        if k == len(selection)-1:
           myfile.write(str(selection[k]))
        else:
           myfile.write(str(selection[k]) + ",")

##select private data, overlapped
print("Sampling private data")

for i in range(n_private):
    selection = []
    for key in labeltoIndex_train.keys():
        tmp = list(np.random.choice(labeltoIndex_train[key], size=int(TrainSize/n_class), replace=False))
        selection = selection + tmp

    with open(outputPath+"private_train" + str(i), 'w') as myfile:
        for k in range(len(selection)):
            if k == len(selection) - 1:
                myfile.write(str(selection[k]))
            else:
                myfile.write(str(selection[k]) + ",")

    selection = []
    for key in labeltoIndex_test.keys():
        tmp = list(np.random.choice(labeltoIndex_test[key], size=int(TestSize/n_class), replace=False))
        selection = selection + tmp

    with open(outputPath+"private_test" + str(i), 'w') as myfile:
        for k in range(len(selection)):
            if k == len(selection) - 1:
                myfile.write(str(selection[k]))
            else:
                myfile.write(str(selection[k]) + ",")

print("ALL public&private data has sampled. Training data and testing data are no overlapped")