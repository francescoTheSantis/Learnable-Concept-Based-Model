import argparse
import torch
import torch.nn as nn
import os
import matplotlib.pyplot as plt
import torch.nn.functional as F
from itertools import product, permutations
import numpy as np
from sklearn.metrics import confusion_matrix, classification_report
import random
import seaborn as sns
import json
from utilities import *
from models import *
from training import *
from loaders import *

class CustomDataset(torch.utils.data.Dataset):
    def __init__(self, original_dataset):
        self.original_dataset = original_dataset

    def __len__(self):
        return len(self.original_dataset)

    def __getitem__(self, idx):
        image, concept, label = self.original_dataset[idx]
        return image, concept, label
    
def add_concept_scores_to_loader(loader):
    original_dataset = loader.dataset
    new_dataset = CustomDataset(original_dataset)
    new_loader = torch.utils.data.DataLoader(new_dataset, batch_size=loader.batch_size, num_workers=loader.num_workers)
    return new_loader

def main(args):
  
    # device check
    if torch.cuda.is_available():
        device = args.device
    else:
        device = "CPU"
        
    print(f'Device: {device}')
    
    # create path for experiment
    path = f"{args.results_folder_name}/results/e2e/{args.backbone}/{args.dataset}/{args.seed}"   
    
    if not os.path.exists(path):
        os.makedirs(path)
        print(f"Path created: {path}")
    else:
        print("Path already exists!")
        
    # Load dataset
    if args.dataset=='CIFAR10':
        train_loader, val_loader, test_loader, test_dataset, _, _ = CIFAR10_loader(args.batch_size, args.val_size, args.backbone, num_workers=args.num_workers)
        classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
    elif args.dataset=='MNIST_even_odd':
        train_loader, val_loader, test_loader, test_dataset, _, _ = MNIST_loader(args.batch_size, args.val_size, args.backbone, num_workers=args.num_workers)
        classes = ('odd', 'even')
    elif args.dataset=='CIFAR100':
        train_loader, val_loader, test_loader, test_dataset, _, _ = CIFAR100_loader(args.batch_size, args.val_size, args.backbone, num_workers=args.num_workers)
        classes = None 
    elif args.dataset=='imagenet':
        train_loader, val_loader, test_loader, test_dataset, _, _ = TinyImagenet_loader(args.batch_size, args.val_size, args.backbone, num_workers=args.num_workers)
        classes = None
    elif args.dataset=='MNIST_sum':
        train_loader, val_loader, test_loader, test_dataset, _, _ = MNIST_addition_loader(args.batch_size, args.val_size, args.backbone, num_workers=args.num_workers, incomplete=False)
        classes = None
    elif args.dataset=='MNIST_sum_incomplete':
        train_loader, val_loader, test_loader, test_dataset, _, _ = MNIST_addition_loader(args.batch_size, args.val_size, args.backbone, num_workers=args.num_workers, incomplete=True)
        classes = None
    elif args.dataset=='Skin':
        data_root = os.path.join(os.getcwd(), "datasets/skin_lesions")
        train_loader, val_loader, test_loader, test_dataset, _, _ = SkinDatasetLoader(args.batch_size, args.backbone, data_root, num_workers=args.num_workers)
        classes = None
    elif args.dataset=='CUB200':
        train_loader, val_loader, test_loader, test_dataset, _, _ = CUB200_loader(args.batch_size, args.val_size, num_workers=args.num_workers)
        classes = None
    else:
        raise ValueError('Dataset not yet implemented!')
    
    # set the seed and initialize concept attention
    model = e2e_model(args.n_labels, args.backbone, device, args.fine_tune).to(device)

    print('Total number of models\'s parameters:', sum([p.numel() for p in model.parameters()]))
    print('Total number of models\'s trainable parameters:', sum([p.numel() for p in model.parameters() if p.requires_grad]))

    params = {
        "model": model,
        "train_loader": train_loader, 
        "val_loader": val_loader, 
        "test_loader": test_loader,
        "n_labels": args.n_labels,
        "lr": args.lr,
        "num_epochs" : args.num_epochs,
        "step_size" : args.step_size,
        "gamma" : args.gamma,
        "device" : device,
        "train" : True,
        "accumulation": args.accumulation,
        "folder": f"{path}",
        "verbose": args.verbose
    }

    # train concept attention
    e2e, task_losses, task_losses_val = train_e2e(**params)
    
    # generate training curves
    dim = (22,5)
    plot_training_curves(task_losses, task_losses_val, None, None, None, None, dim, path, 1)

    
    # make predictions over the test-set
    params["train"] = False
    y_preds, y_true = train_e2e(**params)  
        
    # generate classification report and confusion matrix
    y_true = y_true.cpu().numpy()
    y_preds = y_preds.argmax(-1).detach().cpu().numpy()
    cm = confusion_matrix(y_true, y_preds)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=range(cm.shape[1]), yticklabels=range(cm.shape[0]))
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.title('Confusion Matrix')
    plt.savefig(f'{path}/confusion_matrix.pdf')
    plt.show()
    
    print(classification_report(y_true, y_preds))
    pd.DataFrame(classification_report(y_true, y_preds, output_dict=True)).to_csv(f'{path}/classification_report.csv')
        
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="A script that trains the Concept Attention Model")

    # parameters relate dto the concept attention
    parser.add_argument('--seed', type=int, help="Seed of the experiment")
    parser.add_argument('--n_labels', type=int, help='Number of classes in the dataset')
    parser.add_argument('--backbone', type=str, help="Backbone used for the visual feature extraction")

    # parameters related to the preprocessing and training
    parser.add_argument('--batch_size', type=int, help="Size of the batch")
    parser.add_argument('--val_size', type=float, default=0.1, help="Percentage of training used as validation")
    parser.add_argument('--size', type=int, default=224, help="Input image hape of the Backbone")
    parser.add_argument('--channels', type=int, default=3, help="Image number of channels of the Backbone")
    parser.add_argument('--dataset', type=str, help="Name of the dataset used for the experiment")
    parser.add_argument('--lr', type=float, help="Learning rate")
    parser.add_argument('--num_epochs', type=int, default=40, help="Number of epochs")
    parser.add_argument('--step_size', type=int, help="Step size of the optimizer")
    parser.add_argument('--gamma', type=float, help="Gamma of the optimizer")
    parser.add_argument('--verbose', type=int, help="Verbosity")
    parser.add_argument('--accumulation', type=int, default=0, help="Perform gradient accumulation. If >0 it specify the number of batches to accumulate")
    parser.add_argument('--num_workers', type=int, default=3, help="Specifies the number of workers used by the data loader.")
    parser.add_argument('--results_folder_name', type=str, default='results', help="Name of the Foldr where to store the results") 
    parser.add_argument('--fine_tune', action='store_true', default=False, help='Whether to fine tuning or not the pre-trained backbone')
    parser.add_argument('--device', type=int, default=1, help='Device to use for the training')

    args = parser.parse_args()  
    main(args)
