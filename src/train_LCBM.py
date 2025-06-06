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

def main(args):

    # device check
    if torch.cuda.is_available():
        device = args.device
    else:
        device = "CPU"
        
    print(f'Device: {device}')
    
    # set the seed for torch
    set_seed(args.seed)
    
    # create path for experiment
    path = f"{os.getcwd()}/{args.results_folder_name}/concept_attention/{args.backbone}/{args.dataset}/{args.seed}/"        

    if not os.path.exists(path):
        os.makedirs(path)
        print(f"Path created: {path}")
    else:
        print("Path already exists!")

    # store configuration
    with open(f'{path}/concept_attention_config.json', 'w') as f:
        json.dump(vars(args), f, indent=4)
    
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
    
    # initialize concept attention
    concept_attention = Concept_Attention(args.n_concepts, 
                                          args.concept_emb_size, 
                                          args.n_labels, 
                                          args.size, 
                                          args.channels, 
                                          args.reconstruct_embedding, 
                                          args.backbone, 
                                          device, 
                                          args.deep_parameterization, 
                                          args.use_bias,
                                          args.bound,
                                          args.multi,
                                          args.concept_encoder,
                                          args.expand_recon_bottleneck,
                                          args.fine_tune).to(device)
    
    print('Total parameters Concept Attention:', sum([p.numel() for p in concept_attention.parameters()]))
    print('Trainable parameters Concept Attention:', sum([p.numel() for p in concept_attention.parameters() if p.requires_grad]))
    print('Average number of active concepts:', args.alpha*args.n_concepts) 

    params = {
        "concept_attention": concept_attention,
        "train_loader": train_loader, 
        "val_loader": val_loader, 
        "test_loader": test_loader,
        "n_concepts": args.n_concepts,
        "n_labels": args.n_labels,
        "lr": args.lr,
        "num_epochs" : args.num_epochs,
        "step_size" : args.step_size,
        "gamma" : args.gamma,
        "alpha" : args.alpha,
        "lambda_task" : args.lambda_task,
        "lambda_gate" : args.lambda_gate,
        "lambda_recon" : args.lambda_recon,
        "verbose" : args.verbose, 
        "device" : device,
        "train" : True,
        "binarization_step" : args.binarization_step,
        "KL_penalty": args.KL_penalty,
        "accumulation": args.accumulation,
        "reconstruct_embedding": args.reconstruct_embedding,
        "folder": path,
        "warm_up": args.warm_up
    }

    # train concept attention
    concept_attention, task_losses, gate_penalty_losses, reconstruction_losses, task_losses_val, gate_penalty_losses_val, reconstruction_losses_val = train_concept_attention(**params)
    
    # generate training curves
    dim = (22,5)
    plot_training_curves(task_losses, task_losses_val, gate_penalty_losses, gate_penalty_losses_val, 
                     reconstruction_losses, reconstruction_losses_val, dim, path, 1)

    # load the best concept attention
    best_concept_attention = torch.load(f'{path}/concept_attention.pth', weights_only=False)
    
    # make predictions over the test-set
    params["train"] = False
    params["concept_attention"] = best_concept_attention
    c_preds, y_preds, y_true, c_logits, c_true, reconstruction, c_embs = train_concept_attention(**params)  
        
    # store the tensors which are useful for computing the additional metrics
    tensors = [c_preds, y_preds, y_true, c_true, c_embs]
    names = ['concept_predictions', 'task_predictions','task_ground_truth','concept_ground_truth','concept_embeddings']
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
    cr['mse'] = reconstruction
    pd.DataFrame(cr).to_csv(f'{path}/classification_report.csv')
        
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="A script that trains the Concept Attention Model")

    parser.add_argument('--seed', type=int, default=2, help="Seed of the experiment")
    parser.add_argument('--n_concepts', type=int, default=112, help="Number of concepts")
    parser.add_argument('--concept_emb_size', type=int, default=128, help="Dimension of the concept embeddings")
    parser.add_argument('--n_labels', type=int, default=200, help='Number of classes in the dataset')
    parser.add_argument('--backbone', type=str, default='resnet', help="Backbone used for the visual feature extraction")
    parser.add_argument('--alpha', type=float, default=0.1, help="Concept activatin probability")
    parser.add_argument('--deep_parameterization', action='store_true', default=False, help="Use DNN to compute the weight given the concept embedding")
    parser.add_argument('--reconstruct_embedding', action='store_true', default=False, help="Use a decoder that reconstruc the embedding rather than the entire input (image)")
    parser.add_argument('--use_bias', action='store_true', default=False, help="Use bias to make predictions")
    parser.add_argument('--batch_size', type=int, default=128, help="Size of the batch")
    parser.add_argument('--val_size', type=float, default=0.1, help="Percentage of training used as validation")
    parser.add_argument('--size', type=int, default=224, help="Input image hape of the Backbone")
    parser.add_argument('--channels', type=int, default=3, help="Image number of channels of the Backbone")
    parser.add_argument('--dataset', type=str, default='CUB200', help="Name of the dataset used for the experiment")
    parser.add_argument('--lr', type=float, default=1e-4, help="Learning rate")
    parser.add_argument('--num_epochs', type=int, default=40, help="Number of epochs")
    parser.add_argument('--step_size', type=int, default=10, help="Step size of the optimizer")
    parser.add_argument('--gamma', type=float, default=0.1, help="Gamma of the optimizer")
    parser.add_argument('--lambda_task', type=float, default=1, help="Lambda coefficient of the task term in the loss function")
    parser.add_argument('--lambda_gate', type=float, default=1, help="Lambda coefficient of the KL penalty term in the loss function")
    parser.add_argument('--lambda_recon', type=float, default=1, help="Lambda coefficient of the reconstruction term in the loss function")
    parser.add_argument('--verbose', type=int, default=10, help="Verbosity")
    parser.add_argument('--binarization_step', type=int, default=20, help="After how many epochs to apply the more drstical bernoulli distribution")
    parser.add_argument('--KL_penalty', action='store_true', default=True, help="Use KL divergence as gate penalty term")
    parser.add_argument('--accumulation', type=int, default=0, help="Perform gradient accumulation. If >0 it specify the number of batches to accumulate")
    parser.add_argument('--num_workers', type=int, default=3, help="Specifies the number of workers used by the data loader.")
    parser.add_argument('--warm_up', type=int, default=0, help="The number of epochs in which the concept attention is trained only to reconstruct the input embedding")
    parser.add_argument('--bound', type=int, default=-1, help="The limit for the logits generated by the dot product between the concept embedding and concept prototype")        
    parser.add_argument('--multi', action='store_true', default=False, help="Use categorical distribution insted of multiple bernoulli distributions")        
    parser.add_argument('--concept_encoder', type=str, default='attention', help="Use categorical distribution insted of multiple bernoulli distributions")        
    parser.add_argument('--expand_recon_bottleneck', action='store_true', default=False, help="Use categorical distribution insted of multiple bernoulli distributions") 
    parser.add_argument('--results_folder_name', type=str, default='results', help="Name of the Foldr where to store the results")       
    parser.add_argument('--fine_tune', action='store_true', default=False, help='Whether to fine tuning or not the pre-trained backbone')
    parser.add_argument('--device', type=int, default=1, help='Device to use for the training')
    
    args = parser.parse_args()    

    main(args)
