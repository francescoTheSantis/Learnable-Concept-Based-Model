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


def main():
    parser = argparse.ArgumentParser(description="A script that generates concept prototypes and samples explanation.")
    parser.add_argument('--seed', type=int, help="Seed of the experiment")
    parser.add_argument('--dataset', type=str, help="Name of the dataset used for the experiment")
    parser.add_argument('--backbone', type=str, help="Backbone used to process the image")
    parser.add_argument('--n_samples', type=int, help="Number of prototypes to generate for each concept")
    
    args = parser.parse_args()    

    # device check
    device = torch.cuda.current_device() if torch.cuda.is_available() else "CPU"
    print(f'Device: {device}')
    
    path = f"results/concept_attention/{args.backbone}/{args.dataset}/{args.seed}"
    print('Path:', path)
    
    # load concept attention
    if os.path.exists(f"{path}/concept_attention.pth"):
        print("Loading pre-trained concept attention")
    else:
        raise ValueError('There is no pre-trained concept attention!')
        
    concept_attention = torch.load(f"{path}/concept_attention.pth")
    concept_queries = concept_attention.get_prototypes()
 
    with open(f'{path}/concept_attention_config.json', 'r') as file:
        saved_args = json.load(file)   
    
    # Load dataset
    val_size = 0.1

    if args.dataset=='CIFAR10':
        train_loader, val_loader, test_loader, test_dataset, _, _ = CIFAR10_loader(saved_args['batch_size'], val_size, args.backbone, args.seed, num_workers=saved_args['num_workers'])
        classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
    elif args.dataset=='MNIST_even_odd':
        train_loader, val_loader, test_loader, test_dataset, _, _ = MNIST_loader(saved_args['batch_size'], val_size, args.backbone, args.seed, num_workers=saved_args['num_workers'])
        classes = ('odd', 'even')
    elif args.dataset=='CIFAR100':
        train_loader, val_loader, test_loader, test_dataset, _, _ = CIFAR100_loader(saved_args['batch_size'], val_size, args.backbone, args.seed, num_workers=saved_args['num_workers'])
        classes = None 
    elif args.dataset=='imagenet':
        train_loader, val_loader, test_loader, test_dataset, _, _ = TinyImagenet_loader(saved_args['batch_size'], val_size, args.backbone, args.seed, num_workers=saved_args['num_workers'])
        classes = None
    elif args.dataset=='MNIST_sum':
        train_loader, val_loader, test_loader, test_dataset, _, _ = MNIST_addition_loader(saved_args['batch_size'], val_size, args.backbone, args.seed, num_workers=saved_args['num_workers'])
        classes = None
    else:
        raise ValueError('Dataset not yet implemented!')       

    params = {
        "concept_attention": concept_attention,
        "train_loader": train_loader, 
        "val_loader": val_loader, 
        "test_loader": test_loader,
        "n_concepts": saved_args['n_concepts'],
        "n_labels": saved_args['n_labels'],
        "lr": saved_args['lr'],
        "num_epochs" : saved_args['num_epochs'],
        "step_size" : saved_args['step_size'],
        "gamma" : saved_args['gamma'],
        "alpha" : saved_args['alpha'],
        "lambda_task" : saved_args['lambda_task'],
        "lambda_gate" : saved_args['lambda_gate'],
        "lambda_recon" : saved_args['lambda_recon'],
        "verbose" : saved_args['verbose'], 
        "device" : device,
        "train" : False,
        "binarization_step" : saved_args['binarization_step'],
        "KL_penalty": saved_args['KL_penalty'],
        "accumulation": saved_args['accumulation'],
        "reconstruct_embedding": saved_args['reconstruct_embedding'],
        "folder": f"{path}",
        "warm_up": saved_args['warm_up']
    }
    
    c_preds, y_preds, y_true, c_logits, _, _, _ = train_concept_attention(**params)  

    y_true = y_true.cpu().numpy()
    y_preds = y_preds.argmax(-1).detach().cpu().numpy()

    # generate table with images the lead to the highest concept activation
    learned_concepts(concept_attention, c_logits, test_dataset, args.n_samples, saved_args["n_concepts"], 
                     folder=path, dim=(8,8), device='cuda', alpha=0, classes=classes, concept_order=None)
    

    # distribution of concept activation x class
    #concept_dist_per_label(saved_args["n_labels"], y_preds, c_preds, folder=path, dim=(25,4))

    exp_path = f"{path}/explanations"
    if not os.path.exists(exp_path):
        os.makedirs(exp_path)   
    
    image_idxs = [3175, 5220, 6595, 8723, 9030, 7980]+list(np.random.choice(len(test_dataset), 20, replace=False))
    for i, image_idx in enumerate(image_idxs):
        sample_explanation(concept_attention, c_logits, test_dataset, saved_args["n_concepts"], 
                           image_idx, exp_path, dim=(5,3), classes=classes)    
    
if __name__ == '__main__':
    main()