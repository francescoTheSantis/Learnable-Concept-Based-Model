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
from transformers import CLIPProcessor, CLIPModel

def compute_scores(loader, processor, model, concepts, device):
    all_scores = []
    for images, _ in loader:
        images = images.to(device)
        inputs = processor(text=concepts, images=images, return_tensors="pt", padding=True).to(device)
        outputs = model(**inputs)
        logits_per_image = outputs.logits_per_image
        scores = logits_per_image.softmax(dim=1).detach().cpu().numpy()
        all_scores.append(scores)
    return np.concatenate(all_scores, axis=0)

def compute_clip_concept_scores(loader, concepts, device):
    model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    train_scores = compute_scores(loader, processor, model, concepts, device)
    return train_scores

class CustomDataset(torch.utils.data.Dataset):
    def __init__(self, original_dataset, scores):
        self.original_dataset = original_dataset
        self.scores = scores

    def __len__(self):
        return len(self.original_dataset)

    def __getitem__(self, idx):
        image, concept, label = self.original_dataset[idx]
        score = self.scores[idx]
        return image, concept, label, score
    
def add_concept_scores_to_loader(loader, concept_names):
    original_dataset = loader.dataset
    scores = compute_clip_concept_scores(loader, concept_names, 'cuda')
    new_dataset = CustomDataset(original_dataset, scores)
    new_loader = torch.utils.data.DataLoader(new_dataset, batch_size=loader.batch_size, shuffle=loader.shuffle, num_workers=loader.num_workers)
    return new_loader


def main(args):
  
    # device check
    device = torch.cuda.current_device() if torch.cuda.is_available() else "CPU"
    print(f'Device: {device}')
    
    # create path for experiment
    path = f"{args.root_path}/results/cbm/{args.backbone}/{args.dataset}/{args.seed}"

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
        data_root = os.path.join(args.root_path, "datasets/skin_lesions")
        train_loader, val_loader, test_loader, test_dataset, _, _ = SkinDatasetLoader(args.batch_size, args.backbone, data_root, num_workers=args.num_workers)
        classes = None
    elif args.dataset=='CUB200':
        train_loader, val_loader, test_loader, test_dataset, _, _ = CUB200_loader(args.batch_size, args.val_size, num_workers=args.num_workers)
        classes = None
    else:
        raise ValueError('Dataset not yet implemented!')
            
    # Load concept names
    with open(f'{args.root_path}/concepts/{args.dataset}.json') as f:
        concepts = json.load(f)

    # Add concept scores to the loader
    train_loader = add_concept_scores_to_loader(train_loader, concepts)
    val_loader = add_concept_scores_to_loader(val_loader, concepts)
    test_loader = add_concept_scores_to_loader(test_loader, concepts)

    # set the seed and initialize concept attention
    model = cbm_model(args.n_labels, args.backbone, device).to(device)

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
        "folder": f"{path}"
    }

    # train concept attention
    cbm, task_losses, task_losses_val, concept_losses, concept_losses_val = train_cbm(**params)
    
    # generate training curves
    dim = (22,5)
    plot_training_curves(task_losses, task_losses_val, concept_losses, concept_losses_val, None, None, dim, path, 1)

        # load the best concept attention
    best_cbm = torch.load(f'{path}/cbm.pth')
    
    # make predictions over the test-set
    params["train"] = False
    params["concept_attention"] = best_cbm
    c_preds, y_preds, y_true, c_true = train_cbm(**params)  
        
    # store the tensors which are useful for computing the additional metrics
    tensors = [c_preds, y_preds, y_true, c_true]
    names = ['concept_predictions', 'task_predictions','task_ground_truth','concept_ground_truth']
    for name, tensor in zip(names, tensors):
        if tensor.is_cuda:
            tensor = tensor.cpu()
        array = tensor.numpy()
        np.save(f'{path}/{name}.npy', array)
        
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
    cr = classification_report(y_true, y_preds, output_dict=True)
    pd.DataFrame(cr).to_csv(f'{path}/classification_report.csv')
        
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
    parser.add_argument('--root_path', type=str, default='./data', help="Specifies the root directory of the dataset.")

    args = parser.parse_args()  
    main(args)
