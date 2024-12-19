import torch
from torchvision import datasets, transforms
from torch.utils.data import Dataset, DataLoader, random_split, ConcatDataset
from torchvision.datasets.folder import default_loader
from torchvision.datasets.utils import download_url
from sklearn.model_selection import train_test_split
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import cv2
from PIL import Image
from datasets import load_dataset


class CustomMNIST(Dataset):
    def __init__(self, root, mean, std, train=True, transform='densenet', augment=True):

        self.mnist_data = datasets.MNIST(root=root, train=train, download=True)
        self.transform = transform
        self.mean = mean 
        self.std = std 

        if self.transform in ['densenet', 'vit', 'resnet']:
            self.transform = transforms.Compose([
                transforms.Resize(224),
                transforms.Grayscale(num_output_channels=3), 
                transforms.ToTensor(),
                transforms.Normalize(self.mean, self.std)
            ])
        else:
            raise ValueError('Backbone not implemented yet :(')

    def __len__(self):
        return len(self.mnist_data)

    def __getitem__(self, idx):
        image, label = self.mnist_data[idx]
        transformed_image = self.transform(image)
        even_odd = int(label % 2 == 0)  
        digit = np.zeros(10, dtype=int)
        digit[label] = 1        
        return transformed_image, even_odd, digit
    
    
def MNIST_loader(batch_size, val_size=0.2, backbone='densenet', seed = 42, join=False, dataset='./datasets/', num_workers=3, pin_memory=True, augment=True, shuffle=True):

    mean = (0.4914, 0.4822, 0.4465)
    std = (0.247, 0.243, 0.261)
    generator = torch.Generator().manual_seed(seed) 

    train_dataset = CustomMNIST(root=dataset, mean=mean, std=std, train=True, transform=backbone, augment=augment)
    test_dataset = CustomMNIST(root=dataset, mean=mean, std=std, train=False, transform=backbone)

    if join==False:
        val_size = int(len(train_dataset) * val_size)
        train_size = len(train_dataset) - val_size
        test_size = len(test_dataset)
        train_dataset, val_dataset = random_split(train_dataset, [train_size, val_size], generator=generator)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers, pin_memory=pin_memory)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=pin_memory)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=pin_memory)
        print(f"Train dataset size: {len(train_dataset)}")
        print(f"Validation dataset size: {len(val_dataset)}")
        print(f"Test dataset size: {len(test_dataset)}")
        return train_loader, val_loader, test_loader, test_dataset, mean, std
    else:
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers, pin_memory=pin_memory)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=pin_memory)
        print(f"Train dataset size: {len(train_dataset)}")
        print(f"Test dataset size: {len(test_dataset)}")
        return train_loader, test_loader, test_dataset, mean, std

    
class CustomCIFAR10(Dataset):
    def __init__(self, root, mean, std, train=True, transform='densenet', augment=True):

        self.cifar_data = datasets.CIFAR10(root=root, train=train, download=True)
        self.transform = transform
        self.mean = mean 
        self.std = std 

        if self.transform in ['densenet', 'vit', 'resnet']:
            if train and augment:
                    self.transform = transforms.Compose([
                        transforms.RandomHorizontalFlip(),
                        transforms.RandomRotation(degrees=10), 
                        transforms.Resize(280),  # image_size + 1/4 * image_size
                        transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),
                        transforms.ToTensor(),
                        transforms.Normalize(self.mean, self.std)  
                    ])                 
            else:
                self.transform = transforms.Compose([
                    transforms.Resize(224), 
                    transforms.ToTensor(),
                    transforms.Normalize(self.mean, self.std)  
                ])                
        else:
            raise ValueError('Backbone not yet implemented :(')

    def __len__(self):
        return len(self.cifar_data)

    def __getitem__(self, idx):
        image, label = self.cifar_data[idx]
        transformed_image = self.transform(image)
        alternative_label = np.zeros(15, dtype=int)
        alternative_label[label] = 1 
        return transformed_image, label, alternative_label  # the second label is useless for this dataset but it's returned just to be compliant with the training functions


def CIFAR10_loader(batch_size, val_size=0.2, backbone='densenet', seed = 42, join=False, dataset='./datasets/', num_workers=3, pin_memory=True, augment=True, shuffle=True):

    mean = (0.4914, 0.4822, 0.4465)
    std = (0.247, 0.243, 0.261)
    generator = torch.Generator().manual_seed(seed)  
        
    train_dataset = CustomCIFAR10(root=dataset, mean=mean, std=std, train=True, transform=backbone, augment=augment)
    test_dataset = CustomCIFAR10(root=dataset, mean=mean, std=std, train=False, transform=backbone)

    if join==False:
        val_size = int(len(train_dataset) * val_size)
        train_size = len(train_dataset) - val_size
        test_size = len(test_dataset)
        train_dataset, val_dataset = random_split(train_dataset, [train_size, val_size], generator=generator)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers, pin_memory=pin_memory)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=pin_memory)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=pin_memory)
        print(f"Train dataset size: {len(train_dataset)}")
        print(f"Validation dataset size: {len(val_dataset)}")
        print(f"Test dataset size: {len(test_dataset)}")
        return train_loader, val_loader, test_loader, test_dataset, mean, std
    else:
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers, pin_memory=pin_memory)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=pin_memory)
        print(f"Train dataset size: {len(train_dataset)}")
        print(f"Test dataset size: {len(test_dataset)}")
        return train_loader, test_loader, test_dataset, mean, std


def sparse2coarse(targets):
    """Convert Pytorch CIFAR100 sparse targets to coarse targets"""
    
    coarse_labels = np.array([ 4,  1, 14,  8,  0,  6,  7,  7, 18,  3,  
                               3, 14,  9, 18,  7, 11,  3,  9,  7, 11,
                               6, 11,  5, 10,  7,  6, 13, 15,  3, 15,  
                               0, 11,  1, 10, 12, 14, 16,  9, 11,  5, 
                               5, 19,  8,  8, 15, 13, 14, 17, 18, 10, 
                               16, 4, 17,  4,  2,  0, 17,  4, 18, 17, 
                               10, 3,  2, 12, 12, 16, 12,  1,  9, 19,  
                               2, 10,  0,  1, 16, 12,  9, 13, 15, 13, 
                              16, 19,  2,  4,  6, 19,  5,  5,  8, 19, 
                              18,  1,  2, 15,  6,  0, 17,  8, 14, 13])
    return coarse_labels[targets]
    
    
class CustomCIFAR100(Dataset):
    def __init__(self, root, mean, std, train=True, transform='densenet', augment=True):

        self.cifar_data = datasets.CIFAR100(root=root, train=train, download=True)
        self.transform = transform
        self.mean = mean 
        self.std = std 
        
        if self.transform in ['densenet', 'vit', 'resnet']:
            if train and augment:
                self.transform = transforms.Compose([
                    transforms.RandomHorizontalFlip(),
                    transforms.RandomRotation(degrees=10), 
                    transforms.Resize(280),  # image_size + 1/4 * image_size
                    transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),
                    transforms.ToTensor(),
                    transforms.Normalize(self.mean, self.std)  
                ])
            else:
                self.transform = transforms.Compose([
                    transforms.Resize(224), 
                    transforms.ToTensor(),
                    transforms.Normalize(self.mean, self.std)  
                ])  
        else:
            raise ValueError('Backbone not yet implemented :(')

    def __len__(self):
        return len(self.cifar_data)

    def __getitem__(self, idx):
        image, label = self.cifar_data[idx]
        transformed_image = self.transform(image)
        superclass = sparse2coarse(label)
        alternative_label = np.zeros(20, dtype=int)
        alternative_label[superclass] = 1 
        return transformed_image, label, alternative_label
    
    
def CIFAR100_loader(batch_size, val_size=0.2, backbone='densenet', seed=42, join=False, dataset='./datasets/', num_workers=3, pin_memory=True, augment=True, shuffle=True):

    mean = (0.4914, 0.4822, 0.4465)
    std = (0.247, 0.243, 0.261)
    generator = torch.Generator().manual_seed(seed) 

    train_dataset = CustomCIFAR100(root=dataset, mean=mean, std=std, train=True, transform=backbone, augment=augment)
    test_dataset = CustomCIFAR100(root=dataset, mean=mean, std=std, train=False, transform=backbone)

    if join==False:
        val_size = int(len(train_dataset) * val_size)
        train_size = len(train_dataset) - val_size
        test_size = len(test_dataset)
        train_dataset, val_dataset = random_split(train_dataset, [train_size, val_size], generator=generator)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers, pin_memory=pin_memory)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=pin_memory)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=pin_memory)
        print(f"Train dataset size: {len(train_dataset)}")
        print(f"Validation dataset size: {len(val_dataset)}")
        print(f"Test dataset size: {len(test_dataset)}")
        return train_loader, val_loader, test_loader, test_dataset, mean, std
    else:
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers, pin_memory=pin_memory)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=pin_memory)
        print(f"Train dataset size: {len(train_dataset)}")
        print(f"Test dataset size: {len(test_dataset)}")
        return train_loader, test_loader, test_dataset, mean, std
    

class CustomDataset(Dataset):
    def __init__(self, mean, std, images, labels, digits, transform=None, augment=True):
        
        tensor = [] #np.zeros((len(images), 28, 56)) #torch.Tensor(len(images), 28, 56)
        for i in range(len(images)):
            tmp = torch.Tensor(images[i]).numpy()
            tmp = np.clip(tmp * 255, 0, 255).astype(np.uint8)
            tmp = Image.fromarray(tmp)
            tensor.append(tmp)
            
        self.images = tensor               
        self.labels = labels
        self.transform = transform
        self.mean = mean
        self.std = std
        self.digits = digits
        
        if self.transform in ['densenet', 'vit', 'resnet']:
            self.transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.Grayscale(num_output_channels=3), 
                transforms.ToTensor(),
                transforms.Normalize(self.mean, self.std)
            ])
        else:
            raise ValueError('Backbone not implemented yet :(')

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = self.images[idx]
        label = self.labels[idx]
        image = self.transform(image)   
        digits = np.zeros(10, dtype=int)
        digits[self.digits[idx]] = 1 
        return image, label, digits


def MNIST_addition_loader(batch_size, val_size=0.2, backbone='densenet', seed=42, join=False, dataset='./datasets/', num_workers=3, pin_memory=True, augment=True, shuffle=True):

    mean = (0.4914, 0.4822, 0.4465)
    std = (0.247, 0.243, 0.261)
    
    # fix the seed for both pytorch generator and numpy.random
    generator = torch.Generator().manual_seed(seed) 
    np.random.seed(seed)

    train_dataset = datasets.MNIST(root='./datasets/', train=True, download=True)
    test_dataset = datasets.MNIST(root='./datasets/', train=False)

    # Create composed training-set
    unique_pairs = [str(x)+str(y) for x in range(10) for y in range(10)]
    X_train = []
    y_train = []
    c_train = []
    y_train_lab = np.array([x[1] for x in train_dataset])
    y_test_lab = np.array([x[1] for x in test_dataset])
    y_digits = np.array([x[1] for x in test_dataset])
    samples_per_permutation = 1000
    for train_set_pair in unique_pairs:
        for _ in range(samples_per_permutation):
            rand_i = np.random.choice(np.where(y_train_lab == int(train_set_pair[0]))[0])
            rand_j = np.random.choice(np.where(y_train_lab == int(train_set_pair[1]))[0])
            temp_image = np.zeros((28,56), dtype="uint8")
            temp_image[:,:28] = train_dataset[rand_i][0]
            temp_image[:,28:] = train_dataset[rand_j][0]
            X_train.append(temp_image)
            y_train.append(y_train_lab[rand_i] + y_train_lab[rand_j])
            c_train.append([y_train_lab[rand_i], y_train_lab[rand_j]])  
    
    # Create composed test-set
    X_test = []
    y_test = []
    c_test = []
    samples_per_permutation = 100
    for test_set_pair in unique_pairs:
        for _ in range(samples_per_permutation):
            rand_i = np.random.choice(np.where(y_test_lab == int(test_set_pair[0]))[0])
            rand_j = np.random.choice(np.where(y_test_lab == int(test_set_pair[1]))[0])
            temp_image = np.zeros((28,56), dtype="uint8")
            temp_image[:,:28] = test_dataset[rand_i][0]
            temp_image[:,28:] = test_dataset[rand_j][0]
            X_test.append(temp_image)
            y_test.append(y_test_lab[rand_i] + y_test_lab[rand_j])
            c_test.append([y_test_lab[rand_i], y_test_lab[rand_j]])
      
    
    '''
    # Create the composed dataset (two images concatenated over the x-axis)
    unique_pairs = [str(x)+str(y) for x in range(10) for y in range(10)]
    test_set_pairs = []
    while(len(test_set_pairs) < 10):
        pair_to_add = np.random.choice(unique_pairs)
        if pair_to_add not in test_set_pairs:
            test_set_pairs.append(pair_to_add)
    train_set_pairs = list(set(unique_pairs) - set(test_set_pairs))
    assert(len(test_set_pairs) == 10)
    assert(len(train_set_pairs) == 90)
    for test_set in test_set_pairs:
        assert(test_set not in train_set_pairs)
        print("%s not in training set." % test_set)
    X_train = []
    y_train = []
    c_train = []
    y_train_lab = np.array([x[1] for x in train_dataset])
    y_test_lab = np.array([x[1] for x in test_dataset])
    y_digits = np.array([x[1] for x in test_dataset])
    samples_per_permutation = 1000
    for train_set_pair in train_set_pairs:
        for _ in range(samples_per_permutation):
            rand_i = np.random.choice(np.where(y_train_lab == int(train_set_pair[0]))[0])
            rand_j = np.random.choice(np.where(y_train_lab == int(train_set_pair[1]))[0])
            temp_image = np.zeros((28,56), dtype="uint8")
            temp_image[:,:28] = train_dataset[rand_i][0]
            temp_image[:,28:] = train_dataset[rand_j][0]
            X_train.append(temp_image)
            y_train.append(y_train_lab[rand_i] + y_train_lab[rand_j])
            c_train.append([y_train_lab[rand_i], y_train_lab[rand_j]])
    X_test = []
    y_test = []
    c_test = []
    for test_set_pair in test_set_pairs:
        for _ in range(samples_per_permutation):
            rand_i = np.random.choice(np.where(y_test_lab == int(test_set_pair[0]))[0])
            rand_j = np.random.choice(np.where(y_test_lab == int(test_set_pair[1]))[0])
            temp_image = np.zeros((28,56), dtype="uint8")
            temp_image[:,:28] = test_dataset[rand_i][0]
            temp_image[:,28:] = test_dataset[rand_j][0]
            X_test.append(temp_image)
            y_test.append(y_test_lab[rand_i] + y_test_lab[rand_j])
            c_test.append([y_test_lab[rand_i], y_test_lab[rand_j]])
    '''
    
    
    if join==False:
        # X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, stratify=y_train, test_size=val_size, random_state=seed)
        train_dataset = CustomDataset(mean, std, X_train, y_train, c_train, transform=backbone, augment=augment)        
        # val_dataset = CustomDataset(X_val, y_val, transform=transform)
        test_dataset = CustomDataset(mean, std, X_test, y_test, c_test, transform=backbone)        
        val_size = int(len(train_dataset) * val_size)
        train_size = len(train_dataset) - val_size
        train_dataset, val_dataset = random_split(train_dataset, [train_size, val_size], generator=generator)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers, pin_memory=pin_memory)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=pin_memory)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=pin_memory)
        print(f"Train dataset size: {len(train_dataset)}")
        print(f"Validation dataset size: {len(val_dataset)}")
        print(f"Test dataset size: {len(test_dataset)}")
        return train_loader, val_loader, test_loader, test_dataset, mean, std
    else:
        train_dataset = CustomDataset(mean, std, X_train, y_train, transform=backbone, augment=augment)        
        test_dataset = CustomDataset(mean, std, X_test, y_test, transform=backbone)        
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers, pin_memory=pin_memory)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=pin_memory)
        print(f"Train dataset size: {len(train_dataset)}")
        print(f"Test dataset size: {len(test_dataset)}")
        return train_loader, test_loader, test_dataset, mean, std
    
    

class CustomTinyImagenet(Dataset):
    def __init__(self, mean, std, train=True, transform='densenet', augment=True):

        if train:
            self.tiny_data =  load_dataset("zh-plus/tiny-imagenet")['train']
        else:
            self.tiny_data =  load_dataset("zh-plus/tiny-imagenet")['valid']
                        
        self.mean = mean 
        self.std = std 
        
        if transform in ['densenet', 'vit', 'resnet']:
            if train and augment:
                self.transform = transforms.Compose([
                    transforms.RandomHorizontalFlip(),
                    transforms.RandomRotation(degrees=10), 
                    transforms.Resize(280),  # image_size + 1/4 * image_size
                    transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),
                    transforms.ToTensor(),
                    transforms.Normalize(self.mean, self.std)  
                ])
            else:
                self.transform = transforms.Compose([
                    transforms.Resize(224), 
                    transforms.ToTensor(),
                    transforms.Normalize(self.mean, self.std)  
                ])  
        else:
            raise ValueError('Backbone not yet implemented :(')        
        
    def __len__(self):
        return len(self.tiny_data)

    def __getitem__(self, idx):
        image = self.tiny_data[idx]['image']
        if image.mode == 'L':  # Check if the image is grayscale
            image = image.convert("RGB")  # Convert grayscale to RGB
        label = self.tiny_data[idx]['label']
        transformed_image = self.transform(image)
        alternative_label = np.zeros(30, dtype=int)
        return transformed_image, label, alternative_label
    
    
def TinyImagenet_loader(batch_size, val_size=0.2, backbone='densenet', seed=42, join=False, dataset='./datasets/', num_workers=3, pin_memory=True, augment=True, shuffle=True):

    mean = (0.4914, 0.4822, 0.4465)
    std = (0.247, 0.243, 0.261)
    generator = torch.Generator().manual_seed(seed) 

    train_dataset = CustomTinyImagenet(mean=mean, std=std, train=True, transform=backbone, augment=augment)
    test_dataset = CustomTinyImagenet(mean=mean, std=std, train=False, transform=backbone)

    if join==False:
        val_size = int(len(train_dataset) * val_size)
        train_size = len(train_dataset) - val_size
        test_size = len(test_dataset)
        train_dataset, val_dataset = random_split(train_dataset, [train_size, val_size], generator=generator)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers, pin_memory=pin_memory)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=pin_memory)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=pin_memory)
        print(f"Train dataset size: {len(train_dataset)}")
        print(f"Validation dataset size: {len(val_dataset)}")
        print(f"Test dataset size: {len(test_dataset)}")
        return train_loader, val_loader, test_loader, test_dataset, mean, std
    else:
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers, pin_memory=pin_memory)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=pin_memory)
        print(f"Train dataset size: {len(train_dataset)}")
        print(f"Test dataset size: {len(test_dataset)}")
        return train_loader, test_loader, test_dataset, mean, std
