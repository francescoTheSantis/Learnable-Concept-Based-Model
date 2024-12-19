import argparse
import torch
import torch.nn as nn
import os
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
from sklearn_extra.cluster import KMedoids
from sklearn.metrics import homogeneity_score, f1_score
from tqdm import tqdm
from hungarian_algorithm import algorithm
from transformers import CLIPModel
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
import warnings

warnings.simplefilter("ignore", UserWarning)

# Initialize the CLIP model
clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to("cuda" if torch.cuda.is_available() else "cpu")

def reconstruct_prototype(path, device='cuda'):
    model = torch.load(f'{path}/concept_attention.pth').to(device)
    model.to(device)
    model.eval()
    prototypes = model.get_prototypes().unsqueeze(0)
    for noise in [0.1,0.2,0.3,0,0.4,0.5,0.6,0.7,0.8,0.9,1]:
        fig, axs = plt.subplots(1, model.n_concepts)
        fig.set_size_inches(15,6)
        with torch.no_grad():
            c_pred = torch.eye(model.n_concepts).to(device)
            emb_proj = prototypes * noise        
            c_recon_emb = emb_proj * c_pred[:,:,None]
            z = model.decoder(c_recon_emb.flatten(start_dim=1)).permute(0,2,3,1)
            if z.is_cuda:
                z = z.cpu()
            z = z.numpy()
            for i, ax in enumerate(axs):
                img = z[i,:,:,:]
                img = (img - img.min()) / (img.max() - img.min())
                ax.imshow(img)
                ax.set_title(f'Prototype {i}')
                ax.set_xticks([])
                ax.set_yticks([])
            plt.tight_layout()
            plt.show()

        plt.savefig(f'{path}/prototype_reconstruction_{noise}.pdf')
        
def evaluate_average_importance(concept_attention, loaded_set, proportion, label_idx=1, device='cuda', normalize=True):
    concept_attention.to(device)
    concept_attention.eval()
    prototypes = concept_attention.get_prototypes()
    # y_preds = torch.zeros(1).to(device)
    all_weights = torch.zeros(1, concept_attention.n_concepts).to(device)
    with torch.no_grad():
        for (data, labels, alternative_labels) in loaded_set:
            data = data.to(device)
            labels = labels.to(device)
            alternative_labels = alternative_labels.to(device)
            bsz = data.shape[0]
            _, _, c_emb, c_pred, _, _, _ = concept_attention(data)
            # y_pred = torch.zeros(bsz, concept_attention.n_labels).to(concept_attention.device)
            weights = concept_attention.weights_generator[label_idx](c_emb) # batch, n_concepts, 1
            all_weights = torch.cat([all_weights, weights[:,:,0] * c_pred], axis=0)
            #y_pred[:,i] = torch.bmm(c_pred.unsqueeze(1), weights).squeeze() 
            #y_preds = torch.cat([y_preds, y_pred.argmax(1)], axis=0)
        avg_imp = all_weights[1:,:].mean(0) #(all_weights[1:,:].sum(0)/torch.where(c_pred>0.5,1,0).sum(0))
        if avg_imp.is_cuda:
            avg_imp = avg_imp.cpu()
        avg_imp = avg_imp.numpy()
        # y_preds = y_preds[1:].numpy().astype(int)
        proto_imp = concept_attention.weights_generator[label_idx](prototypes.unsqueeze(0))
        if proto_imp.is_cuda:
            proto_imp = proto_imp.cpu()
        proto_imp = proto_imp[0,:,0].numpy()
        if normalize:
            avg_imp /= np.linalg.norm(avg_imp)
            proto_imp /= np.linalg.norm(proto_imp)
    return avg_imp, proto_imp

def compute_importance(path, loaded_set, device):
    model = torch.load(f'{path}/concept_attention.pth').to(device)
    n_concepts = model.n_concepts
    df = pd.DataFrame()
    avg_imp, proto_imp = evaluate_average_importance(model, loaded_set, device)
    for i in range(avg_imp.shape[0]):
        d = {'Concept':i, 'Average importance':avg_imp[i], 'Prototype importance':proto_imp[i]}
        df = pd.concat([df, pd.DataFrame([d])], ignore_index=True)
    return df

def apply_intervention(c_pred, c_emb, prototypes, proportion):
    ranomd_tensor = torch.rand(c_pred.shape)
    mask = ranomd_tensor < proportion  
    # toggle the values in the binary tensor using the mask
    c_pred_toggled = c_pred.clone()  
    c_pred_toggled[mask] = 1 - c_pred_toggled[mask] 
    # create the concept embedding mask
    mask = torch.abs(c_pred - c_pred_toggled)[:,:,None]
    c_emb = prototypes * mask + c_emb * (1-mask)
    return c_pred_toggled, c_emb

def evaluate_model_with_intervention(concept_attention, y_true, loaded_set, proportion, device='cuda'):
    concept_attention.to(device)
    concept_attention.eval()
    y_preds = torch.zeros(1).to(device)
    with torch.no_grad():
        for (data, labels, alternative_labels) in loaded_set:
            data = data.to(device)
            labels = labels.to(device)
            alternative_labels = alternative_labels.to(device)
            bsz = data.shape[0]
            _, _, c_emb, c_pred, _, _, _ = concept_attention(data)
            y_pred = torch.zeros(bsz, concept_attention.n_labels).to(concept_attention.device)
            prototypes = concept_attention.get_prototypes().unsqueeze(0).expand(bsz, -1, -1)
            c_pred, c_emb = apply_intervention(c_pred, c_emb, prototypes, proportion)
            for i in range(concept_attention.n_labels):
                weights = concept_attention.cls.weights_generator[i](c_emb) # batch, n_concepts, 1
                y_pred[:,i] = torch.bmm(c_pred.unsqueeze(1), weights).squeeze() 
            y_preds = torch.cat([y_preds, y_pred.argmax(1)], axis=0)
    if y_preds.is_cuda:
        y_preds = y_preds.cpu()
    y_preds = y_preds[1:].numpy().astype(int)
    y_true = y_true.astype(int)
    acc = np.sum(y_preds==y_true) / len(y_true)
    return acc 

def intervention(path, loaded_test, trues, prop_list, device):
    model = torch.load(f'{path}/concept_attention.pth').to(device)
    df = pd.DataFrame()
    for prop in tqdm(prop_list):
        acc = evaluate_model_with_intervention(model, trues, loaded_test, prop, device)
        d = {'proportion':prop, 'accuracy':acc}
        df = pd.concat([df, pd.DataFrame([d])], ignore_index=True)
    return df

def get_clip_embeddings(dataloader):
    all_embeddings = []
    clip_model.eval()  # Set the model to evaluation mode
    with torch.no_grad():  # Disable gradient calculation for faster computation
        for batch in dataloader:
            images = batch[0].to("cuda" if torch.cuda.is_available() else "cpu")  # Assuming batch[0] contains the images
            # Generate image embeddings directly without further preprocessing
            image_embeddings = clip_model.get_image_features(images)
            # Normalize the embeddings (optional, but common for CLIP embeddings)
            # image_embeddings = image_embeddings / image_embeddings.norm(dim=-1, keepdim=True)
            # Collect embeddings
            all_embeddings.append(image_embeddings.cpu())
    # Concatenate all embeddings into a single tensor
    return torch.cat(all_embeddings, dim=0)

def clip_clustering(dataloader, pred_concepts):
    n_concepts = pred_concepts.shape[1]
    img_embs = get_clip_embeddings(dataloader)
    silhouette_scores = []
    ch_scores = []
    db_scores = []
    for i in tqdm(range(n_concepts), total=n_concepts):
        pred_cluster = pred_concepts[:,i]
        silhouette_scores.append(silhouette_score(img_embs, pred_cluster, metric='cosine'))
        ch_scores.append(calinski_harabasz_score(img_embs, pred_cluster))
        db_scores.append(davies_bouldin_score(img_embs, pred_cluster))
    return np.mean(np.array(silhouette_scores)), np.mean(np.array(ch_scores)), np.mean(np.array(db_scores))


def concept_alignment_score(
    c_vec,
    c_test,
    y_test,
    step,
    force_alignment=False,
    alignment=None,
    progress_bar=True,
):
    """
    Computes the concept alignment score between learnt concepts and labels.

    :param c_vec: predicted concept representations (can be concept embeddings)
    :param c_test: concept ground truth labels
    :param y_test: task ground truth labels
    :param step: number of integration steps
    :return: concept alignment AUC, task alignment AUC
    """

    # First lets compute an alignment between concept
    # scores and ground truth concepts
    if force_alignment:
        if alignment is None:
            purity_mat = purity.concept_purity_matrix(
                c_soft=c_vec,
                c_true=c_test,
            )
            alignment = purity.find_max_alignment(purity_mat)
        # And use the new vector with its corresponding alignment
        if c_vec.shape[-1] < c_test.shape[-1]:
            # Then the alignment will need to be done backwards as
            # we will have to get rid of the dimensions in c_test
            # which have no aligment at all
            c_test = c_test[:, list(filter(lambda x: x is not None, alignment))]
        else:
            c_vec = c_vec[:, alignment]

    # compute the maximum value for the AUC
    n_clusters = np.linspace(
        2,
        c_vec.shape[0],
        step,
    ).astype(int)
    max_auc = np.trapz(np.ones(len(n_clusters)))

    # for each concept:
    #   1. find clusters
    #   2. compare cluster assignments with ground truth concept/task labels
    concept_auc, task_auc = [], []
    for concept_id in range(c_test.shape[1]):
        concept_homogeneity, task_homogeneity = [], []
        for nc in tqdm(n_clusters):
            kmedoids = KMedoids(n_clusters=nc, random_state=0)
            if c_vec.shape[1] != c_test.shape[1]:
                c_cluster_labels = kmedoids.fit_predict(
                    np.hstack([
                        c_vec[:, concept_id][:, np.newaxis],
                        c_vec[:, c_test.shape[1]:]
                    ])
                )
            elif c_vec.shape[1] == c_test.shape[1] and len(c_vec.shape) == 2:
                c_cluster_labels = kmedoids.fit_predict(
                    c_vec[:, concept_id].reshape(-1, 1)
                )
            else:
                c_cluster_labels = kmedoids.fit_predict(c_vec[:, concept_id, :])

            # compute alignment with ground truth labels
            concept_homogeneity.append(
                homogeneity_score(c_test[:, concept_id], c_cluster_labels)
            )
            task_homogeneity.append(
                homogeneity_score(y_test, c_cluster_labels)
            )
        # compute the area under the curve
        concept_auc.append(np.trapz(np.array(concept_homogeneity)) / max_auc)
        task_auc.append(np.trapz(np.array(task_homogeneity)) / max_auc)
    # return the average alignment across all concepts
    concept_auc = np.mean(concept_auc)
    task_auc = np.mean(task_auc)
    if force_alignment:
        return concept_auc, task_auc, alignment
    return concept_auc, task_auc

def accuracy(predicted, ground_truth):
    return np.mean(predicted == ground_truth)


def find_best_permutation(predictions, ground_truths, emb, metric='f1'):
    n_samples, n_concepts = predictions.shape
    G = {}
    for i in range(n_concepts):
        row = {}
        for j in range(n_concepts):
            if metric=='f1':
                row[str(j)] = int(float(f1_score(predictions[:,i], ground_truths[:,j]))*1000) # int(value * 1000) since the hungarian algorithm struggle to work with floats
            else:
                row[str(j)] = int(float(accuracy(predictions[:,i], ground_truths[:,j]))*1000)
        G[i] = row
 
    assignments = algorithm.find_matching(G, matching_type = 'max', return_type = 'list')
    if assignments:
        new_order = np.zeros(n_concepts, dtype=int)
        for i in range(n_concepts):
            k = int(assignments[i][0][1])
            v = int(assignments[i][0][0])
            new_order[k] = v
    else:
        new_order = np.arange(n_concepts, dtype=int)
    predictions = predictions[:,new_order]
    avg_acc = 0
    avg_f1 = 0
    for i in range(n_concepts):
        avg_acc += accuracy(predictions[:,i], ground_truths[:,i])
        avg_f1 += f1_score(predictions[:,i], ground_truths[:,i])
    avg_acc /= n_concepts
    avg_f1 /= n_concepts
    if isinstance(emb, np.ndarray):
        return predictions, avg_acc, avg_f1, emb[:,new_order,:]
    else:
        return predictions, avg_acc, avg_f1
        

def main():
    parser = argparse.ArgumentParser(description="A script that trains the Concept Attention Model")
    parser.add_argument('--seed', type=int, help="Seed of the experiment")
    parser.add_argument('--backbone', type=str, help="Backbone used for the visual feature extraction")
    parser.add_argument('--dataset', type=str, help="Name of the dataset used for the experiment")

    args = parser.parse_args()    

    # device check
    device = torch.cuda.current_device() if torch.cuda.is_available() else "CPU"
    print(f'Device: {device}')

    # define/create the path to store the results
    path = f"results/concept_attention/{args.backbone}/{args.dataset}/{args.seed}"

    with open(f'{path}/concept_attention_config.json', 'r') as file:
        saved_args = json.load(file)   
        
    # load dataset
    if args.dataset=='CIFAR10':
        train_loader, val_loader, test_loader, test_dataset, _, _ = CIFAR10_loader(saved_args['batch_size'], saved_args['val_size'], args.backbone, args.seed, num_workers=saved_args['num_workers'])
        classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
    elif args.dataset=='MNIST_even_odd':
        train_loader, val_loader, test_loader, test_dataset, _, _ = MNIST_loader(saved_args['batch_size'], saved_args['val_size'], args.backbone, args.seed, num_workers=saved_args['num_workers'])
        classes = ('odd', 'even')
    elif args.dataset=='CIFAR100':
        train_loader, val_loader, test_loader, test_dataset, _, _ = CIFAR100_loader(saved_args['batch_size'], saved_args['val_size'], args.backbone, args.seed, num_workers=saved_args['num_workers'])
        classes = None 
    elif args.dataset=='imagenet':
        train_loader, val_loader, test_loader, test_dataset, _, _ = TinyImagenet_loader(saved_args['batch_size'], saved_args['val_size'], args.backbone, args.seed, num_workers=saved_args['num_workers'])
        classes = None
    elif args.dataset=='MNIST_sum':
        train_loader, val_loader, test_loader, test_dataset, _, _ = MNIST_addition_loader(saved_args['batch_size'], saved_args['val_size'], args.backbone, args.seed, num_workers=saved_args['num_workers'])
        classes = None
    else:
        raise ValueError('Dataset not yet implemented!') 
    
    # load the arrays containing the truth values
    true_concept = np.load(f'{path}/concept_ground_truth.npy')
    true_task = np.load(f'{path}/task_ground_truth.npy')

    # load both concept and task predictions
    pred_concept = np.load(f'{path}/concept_predictions.npy')
    pred_concept = np.where(pred_concept>0.5,1,0)
    pred_task = np.load(f'{path}/task_predictions.npy')  
    pred_task = np.argmax(pred_task, axis=1)
    pred_embs = np.load(f'{path}/concept_embeddings.npy')

    if args.dataset in ['MNIST_sum', 'MNIST_even_odd', 'CIFAR100']:

        # get the correct permutation of concept labels that maximizes the concept accuracy    
        pred_concept, concept_accuracy, concept_f1, pred_embs = find_best_permutation(pred_concept, true_concept, pred_embs)
        print("Concept Accuracy:", concept_accuracy, "Concept macro F1:", concept_f1)

        # compute CLIP metric
        sil, ch, db = clip_clustering(test_loader, pred_concept)

        # compute CAS metric
        step = 5
        concept_auc, _ = concept_alignment_score(pred_embs, true_concept, true_task, step) # use concept embedding for concept attention and concept scores for the other methodologies

        df = pd.DataFrame([{'Concept accuracy':concept_accuracy, 'Concept macro F1-score':concept_f1, 'CAS':concept_auc, 'Silhouette':sil, 'Calinski-Harabasz':ch, 'Davies-Bouldin':db}])
        df.to_csv(f'{path}/concept_results.csv')

    # compute intervention
    intervention_df = intervention(path, test_loader, true_task, np.arange(0,1.1,0.1), device)
    intervention_df.to_csv(f'{path}/interventions.csv')

    '''
    reconstruct_prototype(path, device)
    
    # prototype analysis
    if args.dataset=='MNIST_even_odd':
        #print('here')
        imp_df = compute_importance(path, test_loader, device)
        #reconstruct_prototype(path, device)
        imp_df.to_csv(f'{path}/global_importance.csv')
    '''
        
if __name__ == '__main__':
    main()
