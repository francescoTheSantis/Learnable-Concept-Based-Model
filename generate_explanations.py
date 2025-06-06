import argparse
import torch
import os
import numpy as np
import json
from utilities import *
from models import *
from training import *
from loaders import *


def main(args):

    device = args.device
    print(f'Device: {device}')
    
    path = f"results/concept_attention/{args.backbone}/{args.dataset}/{args.seed}"
    print('Path:', path)
    
    # load concept attention
    if os.path.exists(f"{path}/concept_attention.pth"):
        print("Loading pre-trained concept attention")
    else:
        raise ValueError('There is no pre-trained concept attention!')

    saved_model = torch.load(f'{path}/concept_attention.pth', weights_only=False, map_location=torch.device(f'cuda:{device}' if torch.cuda.is_available() else 'cpu'))
    model_args = {
        'n_concepts': saved_model.n_concepts,
        'emb_size': saved_model.emb_size,
        'n_labels': saved_model.n_labels,
        'size': saved_model.size,
        'channels': saved_model.channels,
        'embedding': False,
        'backbone': args.backbone,
        'device': device,
        'deep_parameterization': saved_model.deep_parameterization,
        'use_bias': saved_model.use_bias,
        'bound': saved_model.bound,
        'multi_dist': saved_model.multi_dist,
        'concept_encoder': saved_model.concept_encoder,
        'expand_recon_bottleneck': saved_model.expand_recon_bottleneck,
        'fine_tune': saved_model.fine_tune
    }
    concept_attention = Concept_Attention(**model_args).to(device)
    concept_attention.load_state_dict(saved_model.state_dict())
    concept_queries = concept_attention.get_prototypes()
 
    with open(f'{path}/concept_attention_config.json', 'r') as file:
        saved_args = json.load(file)   
    
    if args.dataset=='CIFAR10':
        train_loader, val_loader, test_loader, test_dataset, _, _ = CIFAR10_loader(saved_args['batch_size'], saved_args['val_size'], args.backbone, num_workers=saved_args['num_workers'])
        classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
    elif args.dataset=='MNIST_even_odd':
        train_loader, val_loader, test_loader, test_dataset, _, _ = MNIST_loader(saved_args['batch_size'], saved_args['val_size'], args.backbone, num_workers=saved_args['num_workers'])
        classes = ('odd', 'even')
    elif args.dataset=='CIFAR100':
        train_loader, val_loader, test_loader, test_dataset, _, _ = CIFAR100_loader(saved_args['batch_size'], saved_args['val_size'], args.backbone, num_workers=saved_args['num_workers'])
        classes = None 
    elif args.dataset=='imagenet':
        train_loader, val_loader, test_loader, test_dataset, _, _ = TinyImagenet_loader(saved_args['batch_size'], saved_args['val_size'], args.backbone, num_workers=saved_args['num_workers'])
        classes = None
    elif args.dataset=='MNIST_sum':
        train_loader, val_loader, test_loader, test_dataset, _, _ = MNIST_addition_loader(saved_args['batch_size'], saved_args['val_size'], args.backbone, num_workers=saved_args['num_workers'], incomplete=False)
        classes = None
    elif args.dataset=='MNIST_sum_incomplete':
        train_loader, val_loader, test_loader, test_dataset, _, _ = MNIST_addition_loader(saved_args['batch_size'], saved_args['val_size'], args.backbone, num_workers=saved_args['num_workers'], incomplete=True)
        classes = None
    elif args.dataset=='Skin':
        data_root = os.path.join(os.getcwd(), "datasets/skin_lesions")
        train_loader, val_loader, test_loader, test_dataset, _, _ = SkinDatasetLoader(saved_args['batch_size'], args.backbone, data_root, num_workers=saved_args['num_workers'])
        classes = None
    elif args.dataset=='CUB200':
        train_loader, val_loader, test_loader, test_dataset, _, _ = CUB200_loader(saved_args['batch_size'], saved_args['val_size'], num_workers=saved_args['num_workers'])
        classes = []
        with open(os.path.join(os.getcwd(), 'datasets/CUB_200_2011/classes.txt'), 'r', encoding='utf-8') as file:
            for line in file:
                # Split the line by the first space to separate the index and the label
                label = line.strip().split(' ')[1].split('.')[1].replace(' ̇', ' ')
                classes.append(label)
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
    c_preds = torch.where(c_preds>0.5, 1, 0)

    y_true = y_true.cpu().numpy()
    y_preds = y_preds.argmax(-1).detach().cpu().numpy()

    # generate table with images the lead to the highest concept activation
    learned_concepts(concept_attention, c_logits, test_dataset, 7, saved_args["n_concepts"], 
                     folder=path, dim=(8,8), device=device, alpha=0.6, classes=classes, concept_order=None)
    

    # generate table with images the lead to the highest concept activation
    learned_concepts(concept_attention, c_logits, test_dataset, 7, saved_args["n_concepts"], 
                     folder=path, dim=(8,8), device=device, alpha=0, classes=classes, concept_order=None)
    
    exp_path = f"{path}/explanations"
    if not os.path.exists(exp_path):
        os.makedirs(exp_path)   
    
    if args.dataset=='CUB200':
        image_idxs = list(np.random.choice(len(test_dataset), 10, replace=False))
    else:
        image_idxs = list(np.random.choice(len(test_dataset), 15, replace=False))

    for i, image_idx in enumerate(image_idxs):
        sample_explanation(concept_attention, c_logits, test_dataset, saved_args["n_concepts"], 
                           image_idx, exp_path, dim=(5,3), device=device, classes=classes)    
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="A script that generates concept prototypes and samples explanation.")
    parser.add_argument('--seed', type=int, default=0, help="Seed of the experiment")
    parser.add_argument('--dataset', type=str, default='CUB200', help="Name of the dataset used for the experiment")
    parser.add_argument('--backbone', type=str, default='resnet', help="Backbone used to process the image")
    parser.add_argument('--n_samples', type=int, default=10, help="Number of prototypes to generate for each concept")
    parser.add_argument('--device', type=int, default=1, help="Device")
    
    args = parser.parse_args()    

    main(args)