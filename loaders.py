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
import os, re
import requests
import tarfile

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


def MNIST_addition_loader(batch_size, val_size=0.2, backbone='densenet', seed=42, join=False, dataset='./datasets/', num_workers=3, pin_memory=True, augment=True, shuffle=True, incomplete=False):

    mean = (0.4914, 0.4822, 0.4465)
    std = (0.247, 0.243, 0.261)
    
    # fix the seed for both pytorch generator and numpy.random
    generator = torch.Generator().manual_seed(seed) 
    np.random.seed(seed)

    train_dataset = datasets.MNIST(root='./datasets/', train=True, download=True)
    test_dataset = datasets.MNIST(root='./datasets/', train=False)

    if not incomplete:
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
    else:
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


class CustomSkinDataset(Dataset):
    def __init__(self, root, mean, std, phase='train', transform='densenet', augment=True):
        self.root = os.path.join(root, phase)
        self.transform = transform
        self.mean = mean
        self.std = std
 
        # Define the mapping of concepts to classes
        self.class_map = {
            'Healthy': ['Healthy'],
            'Benign': ['Actinic keratoses', 'Benign keratosis-like lesions', 'Dermatofibroma',
                       'Melanocytic nevi', 'Vascular lesions'],
            'Malignant': ['Basal cell carcinoma', 'Melanoma', 'Squamous cell carcinoma'],
            'Infectious Diseases': ['Chickenpox', 'Cowpox', 'HFMD', 'Measles', 'Monkeypox']
        }
 
        # Define the ordered list of concepts
        self.concepts = [
            'Healthy',
            'Actinic keratoses',
            'Benign keratosis-like lesions',
            'Dermatofibroma',
            'Melanocytic nevi',
            'Vascular lesions',
            'Basal cell carcinoma',
            'Melanoma',
            'Squamous cell carcinoma',
            'Chickenpox',
            'Cowpox',
            'HFMD',  # Hand, Foot, and Mouth Disease
            'Measles',
            'Monkeypox'
        ]
 
        # Map concepts to indices
        self.concept_to_idx = {concept: idx for idx, concept in enumerate(self.concepts)}
 
        # Map classes to indices
        self.class_to_idx = {cls: idx for idx, cls in enumerate(self.class_map)}
 
        # List all images and their respective concepts
        self.data = []
        for concept in self.concepts:
            concept_dir = os.path.join(self.root, concept)
            if os.path.isdir(concept_dir):
                for img_name in os.listdir(concept_dir):
                    img_path = os.path.join(concept_dir, img_name)
                    self.data.append((img_path, concept))
 
        # Define transformations
        if self.transform in ['densenet', 'vit', 'resnet']:
            if phase == 'train':
                self.transform = transforms.Compose([
                    transforms.RandomHorizontalFlip(),
                    transforms.RandomRotation(degrees=10),
                    transforms.Resize((280, 280)),  # image_size + 1/4 * image_size
                    transforms.RandomResizedCrop((224,224)),
                    transforms.ToTensor(),
                    transforms.Normalize(self.mean, self.std)  
                ])
            else:
                self.transform = transforms.Compose([
                    transforms.Resize((224, 224)),
                    transforms.ToTensor(),
                    transforms.Normalize(self.mean, self.std)
                ])
 
        else:
            raise ValueError('Backbone not implemented yet :(')
 
    def __len__(self):
        return len(self.data)
 
    def __getitem__(self, idx):
        img_path, concept = self.data[idx]
        image = Image.open(img_path).convert('RGB')
        transformed_image = self.transform(image)
 
        # Get the class label using class_map
        class_label = None
        for cls, concepts in self.class_map.items():
            if concept in concepts:
                class_label = self.class_to_idx[cls]
                break
 
        if class_label is None:
            raise ValueError(f"Concept '{concept}' not found in class_map!")
 
        # Create the concept one-hot array
        subclass_labels = np.zeros(len(self.concepts), dtype=int)
        concept_index = self.concept_to_idx[concept]
        subclass_labels[concept_index] = 1
 
        return transformed_image, class_label, subclass_labels
 

    
def SkinDatasetLoader(batch_size, backbone='resnet', dataset='./dataset/', num_workers=3, pin_memory=True, augment=True, shuffle=True):
    mean = (0.4914, 0.4822, 0.4465)
    std = (0.247, 0.243, 0.261)
 
    
    train_dataset = CustomSkinDataset(root=dataset, mean=mean, std=std, phase='train', transform=backbone, augment=augment)
    val_dataset = CustomSkinDataset(root=dataset, mean=mean, std=std, phase='val', transform=backbone)
    test_dataset = CustomSkinDataset(root=dataset, mean=mean, std=std, phase='test', transform=backbone)
 
    # Create loaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers, pin_memory=pin_memory)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=pin_memory)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=pin_memory)
 
    print(f"Train dataset size: {len(train_dataset)}")
    print(f"Validation dataset size: {len(val_dataset)}")
    print(f"Test dataset size: {len(test_dataset)}")
 
    return train_loader, val_loader, test_loader, test_dataset, mean, std
    

def SkinDatasetLoader(batch_size, backbone='resnet', dataset='./dataset/', num_workers=3, pin_memory=True, augment=True, shuffle=True):
    mean = (0.4914, 0.4822, 0.4465)
    std = (0.247, 0.243, 0.261)

    train_dataset = CustomSkinDataset(root=dataset, mean=mean, std=std, phase='train', transform=backbone, augment=augment)
    val_dataset = CustomSkinDataset(root=dataset, mean=mean, std=std, phase='val', transform=backbone)
    test_dataset = CustomSkinDataset(root=dataset, mean=mean, std=std, phase='test', transform=backbone)
    
    # Create loaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers, pin_memory=pin_memory)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=pin_memory)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=pin_memory)

    print(f"Train dataset size: {len(train_dataset)}")
    print(f"Validation dataset size: {len(val_dataset)}")
    print(f"Test dataset size: {len(test_dataset)}")

    return train_loader, val_loader, test_loader, test_dataset, mean, std


class CUBDataset(Dataset):
    def __init__(self, root_dir, train=False, mean=None, std=None):
        SELECTED_CONCEPTS = [1, 4, 6, 7, 10, 14, 15, 20, 21, 23, 25, 29, 30, 35, 36, 38, 40, 44, 
                             45, 50, 51, 53, 54, 56, 57, 59, 63, 64, 69, 70, 72, 75, 80, 84, 90,
                             91, 93, 99, 101, 106, 110, 111, 116, 117, 119, 125, 126, 131, 132, 
                             134, 145, 149, 151, 152, 153, 157, 158, 163, 164, 168, 172, 178, 179, 
                             181, 183, 187, 188, 193, 194, 196, 198, 202, 203, 208, 209, 211, 212, 
                             213, 218, 220, 221, 225, 235, 236, 238, 239, 240, 242, 243, 244, 249,
                             253, 254, 259, 260, 262, 268, 274, 277, 283, 289, 292, 293, 294, 298,
                             299, 304, 305, 308, 309, 310, 311]

        self.root_dir = root_dir
        self.train = train

        self.mean = mean
        self.std = std

        
        if self.train:
            self.transform = transforms.Compose([
                        transforms.RandomHorizontalFlip(),
                        transforms.RandomRotation(degrees=10), 
                        transforms.Resize((280, 280)),  # image_size + 1/4 * image_size
                        transforms.RandomResizedCrop((224, 224)),
                        transforms.ToTensor()
                    ])
        else:
            self.transform = transforms.Compose([
                    transforms.Resize((224, 224)), 
                    transforms.ToTensor()
                ]) 
        
        if os.path.isdir(self.root_dir + "/CUB_200_2011") is False:
            url = "https://data.caltech.edu/records/65de6-vp158/files/CUB_200_2011.tgz?download=1"
            file_name = "CUB_200_2011.tgz"
            self._download_file(url, file_name)
            self._extract_file(file_name, root_dir)
            os.remove(file_name)

        # Parse the dataset files
        dataset_dir = root_dir + "/CUB_200_2011"
        self.image_paths_train = []
        self.labels_train = []
        self.image_ids_train = []
        self.image_paths_test = []
        self.labels_test = []
        self.image_ids_test = []

        with open(os.path.join(dataset_dir, "images.txt"), "r") as img_file:
            image_lines = img_file.readlines()
        with open(os.path.join(dataset_dir, "image_class_labels.txt"), "r") as label_file:
            label_lines = label_file.readlines()
        with open(os.path.join(dataset_dir, "train_test_split.txt"), "r") as split_file:
            split_lines = split_file.readlines()

        # Initialize a dictionary to hold the boolean arrays for each image
        self.image_attributes = {}
        with open(os.path.join(dataset_dir, "./attributes/image_attribute_labels.txt"), "r") as file:
            for line in file:
                matches = re.findall(r"\d+\.\d+|\d+", line)
                image_id, attribute_id, is_present = matches[0], matches[1], matches[2] #line.strip().split(" ")
                image_id = int(image_id)
                attribute_id = int(attribute_id)
                is_present = int(is_present)
                if image_id not in self.image_attributes:
                    cnt = 0
                    self.image_attributes[image_id] = np.zeros(len(SELECTED_CONCEPTS), dtype=float)
                if attribute_id in SELECTED_CONCEPTS:
                    self.image_attributes[image_id][cnt] = float(is_present)
                    cnt += 1


        # Extract image paths and labels
        for img_line, label_line, split_line in zip(image_lines, label_lines, split_lines):
            img_id, img_path = img_line.strip().split(" ")
            label_id, label = label_line.strip().split(" ")
            img2_id, split_id = split_line.strip().split(" ")
            assert img_id == label_id == img2_id # Ensure consistent IDs
            if split_id == '1':
                self.image_ids_train.append(int(img_id))
                self.image_paths_train.append(os.path.join(dataset_dir, "images", img_path))
                self.labels_train.append(int(label) - 1)  # Convert to zero-based index
            else:
                self.image_ids_test.append(int(img_id))
                self.image_paths_test.append(os.path.join(dataset_dir, "images", img_path))
                self.labels_test.append(int(label) - 1)  # Convert to zero-based index


    def __len__(self):
        if self.train:
            return len(self.image_paths_train)
        return len(self.image_paths_test)
    
    # Step 1: Download the file
    def _download_file(self, url, file_name):
        print(f"Downloading {file_name}...")
        response = requests.get(url, stream=True)
        response.raise_for_status()  # Raise an exception for HTTP errors
        with open(file_name, "wb") as file:
            for chunk in response.iter_content(chunk_size=8192):
                file.write(chunk)
        print(f"Downloaded {file_name} successfully.")

    # Step 2: Extract the tar.gz file
    def _extract_file(self, file_name, output_dir):
        print(f"Extracting {file_name}...")
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        with tarfile.open(file_name, "r:gz") as tar:
            tar.extractall(path=output_dir)
        print(f"Extracted files to {output_dir}.")

    def __getitem__(self, idx):
        if self.train:
            img_path = self.image_paths_train[idx]
            label = self.labels_train[idx]
            concepts = torch.from_numpy(self.image_attributes[self.image_ids_train[idx]])
        else:
            img_path = self.image_paths_test[idx]
            label = self.labels_test[idx]
            concepts = torch.from_numpy(self.image_attributes[self.image_ids_test[idx]])
        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image, label, concepts

def CUB200_loader(batch_size, val_size=0.1, seed = 42, dataset='./datasets/', num_workers=3, pin_memory=True, augment=True, shuffle=True):
    generator = torch.Generator().manual_seed(seed) 

    mean = (0.4914, 0.4822, 0.4465)
    std = (0.247, 0.243, 0.261)

    train_dataset = CUBDataset(root_dir=dataset, train=True, mean=mean, std=std)
    test_dataset = CUBDataset(root_dir=dataset, train=False, mean=mean, std=std)

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
