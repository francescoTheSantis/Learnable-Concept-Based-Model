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
from tqdm import tqdm

def compute_scores(loader, processor, model, concepts, device):
    all_scores = []
    for images, _, _ in tqdm(loader):
        images = images.to(device)
        images = (images - images.min()) / (images.max() - images.min())
        inputs = processor(text=concepts, images=images, return_tensors="pt", padding=True).to(device)
        outputs = model(**inputs)
        logits_per_image = outputs.logits_per_image
        scores = logits_per_image.softmax(dim=1).detach().cpu().numpy()
        all_scores.append(scores)
    return np.concatenate(all_scores, axis=0)

def compute_clip_concept_scores(loader, concepts, device):
    model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
    for param in model.parameters():
        param.requires_grad = False
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    scores = compute_scores(loader, processor, model, concepts, device)
    # normalize scores
    mean = np.mean(scores, axis=0, keepdims=True)
    std = np.std(scores, axis=0, keepdims=True)
    scores -= mean
    scores /= std
    return scores

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
    
def add_concept_scores_to_loader(loader, concept_names, device):
    original_dataset = loader.dataset
    scores = compute_clip_concept_scores(loader, concept_names, device)
    new_dataset = CustomDataset(original_dataset, scores)
    new_loader = torch.utils.data.DataLoader(new_dataset, batch_size=loader.batch_size, num_workers=loader.num_workers)
    return new_loader


def main(args):
  
    # device check
    if torch.cuda.is_available():
        device = args.device
    else:
        device = "CPU"
    
    # create path for experiment
    path = f"{os.getcwd()}/{args.results_folder_name}/lf_cbm/{args.backbone}/{args.dataset}/{args.seed}/"        

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
    with open(f'{os.getcwd()}/concept_lists/{args.dataset}.json') as f:
        concepts = json.load(f)['selected_names']

    n_concepts = len(concepts)

    # Add concept scores to the loader
    train_loader = add_concept_scores_to_loader(train_loader, concepts, args.device)
    val_loader = add_concept_scores_to_loader(val_loader, concepts, args.device)
    test_loader = add_concept_scores_to_loader(test_loader, concepts, args.device)

    # set the seed and initialize concept attention
    label_free_approach = not args.supervised
    model = cbm_model(args.n_labels, n_concepts, args.backbone, device, label_free_approach, True, args.fine_tune).to(device)

    print('Total number of models\'s parameters:', sum([p.numel() for p in model.parameters()]))
    print('Total number of models\'s trainable parameters:', sum([p.numel() for p in model.parameters() if p.requires_grad]))

    params = {
        'cbm': model,
        'train_loader': train_loader,
        'val_loader': val_loader,
        'test_loader': test_loader,
        'n_concepts': n_concepts,
        'n_labels': args.n_labels,
        'lr': args.lr,
        'num_epochs': args.num_epochs,
        'step_size': args.step_size,
        'gamma': args.gamma,
        'verbose': args.verbose,
        'device': device,
        'train': True,
        "folder": f"{path}",
        "supervised": args.supervised,
    }

    # train concept attention
    cbm, task_losses, concept_losses, task_losses_val, concept_losses_val = train_cbm(**params)
    
    # generate training curves
    dim = (22,5)
    #plot_training_curves(task_losses, task_losses_val, concept_losses, concept_losses_val, np.zeros(len(concept_losses_val)), np.zeros(len(concept_losses_val)), dim, path, 1)

        # load the best concept attention
    best_cbm = torch.load(f'{path}/cbm.pth', weights_only=False)
    
    # make predictions over the test-set
    params["train"] = False
    params["cbm"] = best_cbm
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
    parser.add_argument('--seed', type=int, default=0, help="Seed of the experiment")
    parser.add_argument('--n_labels', type=int, default=200, help='Number of classes in the dataset')
    parser.add_argument('--backbone', type=str, default='resnet', help="Backbone used for the visual feature extraction")

    # parameters related to the preprocessing and training
    parser.add_argument('--batch_size', type=int, default=128, help="Size of the batch")
    parser.add_argument('--val_size', type=float, default=0.1, help="Percentage of training used as validation")
    parser.add_argument('--size', type=int, default=224, help="Input image hape of the Backbone")
    parser.add_argument('--channels', type=int, default=3, help="Image number of channels of the Backbone")
    parser.add_argument('--dataset', type=str, default='CUB200', help="Name of the dataset used for the experiment")
    parser.add_argument('--lr', type=float, default=1e-4, help="Learning rate")
    parser.add_argument('--num_epochs', type=int, default=40, help="Number of epochs")
    parser.add_argument('--step_size', type=int, default=10, help="Step size of the optimizer")
    parser.add_argument('--gamma', type=float, help="Gamma of the optimizer")
    parser.add_argument('--verbose', type=int, default=1, help="Verbosity")
    parser.add_argument('--accumulation', type=int, default=0, help="Perform gradient accumulation. If >0 it specify the number of batches to accumulate")
    parser.add_argument('--num_workers', type=int, default=3, help="Specifies the number of workers used by the data loader.")
    parser.add_argument('--results_folder_name', type=str, default='results', help="Name of the Foldr where to store the results")       
    parser.add_argument('--supervised', action='store_true', default=False, help="Specifies if the model is trained in a supervised way (over the concepts).")
    parser.add_argument('--device', type=int, default=0, help='Device to use for the training')
    parser.add_argument('--fine_tune', action='store_true', default=False, help='Whether to fine tuning or not the pre-trained backbone')

    args = parser.parse_args()  
    main(args)
