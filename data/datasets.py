import os
import torchvision
from torch.utils.data import DataLoader
from torchvision.datasets import CIFAR10, MNIST, FashionMNIST
import torchvision.transforms as transforms

def Dataset(dataset_name, batch_size=128, data_dir='data'):

    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
    if dataset_name == 'cifar10':
        print('Dataset: CIFAR10.')
        trainset = CIFAR10(root=os.path.join(data_dir, 'cifar10'), train=True, download=True, transform=transforms.Compose([
                    transforms.RandomHorizontalFlip(),
                    transforms.RandomCrop(32, 4),
                    transforms.ToTensor(),
                    normalize]))


        testset = CIFAR10(root=os.path.join(data_dir, 'cifar10'), train=False, download=True, transform=transforms.Compose([
                    transforms.ToTensor(),
                    normalize]))
    elif dataset_name == 'fashion_mnist':
        print('Dataset: FashionMNIST.')
        trainset = FashionMNIST(root=os.path.join(data_dir, 'fashion_mnist'), train=True, download=True, transform=transforms.Compose([
                    transforms.Grayscale(3),
                    transforms.Resize(32),
                    transforms.RandomHorizontalFlip(),
                    transforms.RandomCrop(32, 4),
                    transforms.ToTensor(),
                    normalize]))


        testset = FashionMNIST(root=os.path.join(data_dir, 'fashion_mnist'), train=False, download=True, transform=transforms.Compose([
                    transforms.Grayscale(3),
                    transforms.Resize(32),
                    transforms.ToTensor(),
                    normalize]))
    elif dataset_name == 'imagenet':
        print('Dataset: ImageNet')
        train_sampler = None
        data_dir = ""
        train_dir = os.path.join(data_dir, 'train')
        test_dir = os.path.join(data_dir, 'val')
        transform_train = transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),])

        transform_test = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),])

        trainset = torchvision.datasets.ImageFolder(root=train_dir, transform=transform_train)
        testset = torchvision.datasets.ImageFolder(root=test_dir, transform=transform_test)
    else:
        raise NotImplementedError("Only cifar10, imagenet, fashion mnist are allowed.")


    trainloader = DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=4)
    testloader = DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=4)
    return trainloader, testloader
