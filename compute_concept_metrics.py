import argparse
import torch
import os
import numpy as np
from utilities import *
from models import *
from training import *
from loaders import *
from sklearn_extra.cluster import KMedoids
from sklearn.metrics import homogeneity_score
from tqdm import tqdm
import warnings
from sklearn.cluster import KMeans
from scipy.optimize import linear_sum_assignment
from sklearn.metrics import f1_score

warnings.simplefilter("ignore", UserWarning)

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
    all_weights = torch.zeros(1, concept_attention.n_concepts).to(device)
    with torch.no_grad():
        for (data, labels, alternative_labels) in loaded_set:
            data = data.to(device)
            labels = labels.to(device)
            alternative_labels = alternative_labels.to(device)
            bsz = data.shape[0]
            _, _, c_emb, c_pred, _, _, _ = concept_attention(data)
            weights = concept_attention.weights_generator[label_idx](c_emb) # batch, n_concepts, 1
            all_weights = torch.cat([all_weights, weights[:,:,0] * c_pred], axis=0)
        avg_imp = all_weights[1:,:].mean(0) 
        if avg_imp.is_cuda:
            avg_imp = avg_imp.cpu()
        avg_imp = avg_imp.numpy()
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

def invert_tensor_values(tensor, min_val, max_val):
    return min_val + (1 - (tensor - min_val) / (max_val - min_val)) * (max_val - min_val)

def apply_intervention(c_pred, c_emb, prototypes, proportion, method='concept_attention'):
    ranomd_tensor = torch.rand(c_pred.shape)
    mask = ranomd_tensor < proportion  
    # toggle the values in the binary tensor using the mask
    c_pred_toggled = c_pred.clone()  

    if method=='concept_attention':
        c_pred_toggled[mask] = 1 - c_pred_toggled[mask] 
    else:
        max_val = c_pred.max()
        min_val = c_pred.min()
        c_pred = invert_tensor_values(c_pred, min_val, max_val)
        c_pred_toggled[mask] = c_pred[mask]

    # create the concept embedding mask
    mask = torch.abs(c_pred - c_pred_toggled)[:,:,None]
    c_emb = prototypes * mask + c_emb * (1-mask)
    return c_pred_toggled, c_emb

def evaluate_model_with_intervention(model, y_true, loaded_set, proportion, model_name='concept_attention', device='cpu'):
    #print('funciton device', device)
    #print('model device', concept_attention.device)
    model.eval()
    y_preds = torch.zeros(1).to(device)
    with torch.no_grad():
        for (data, labels, alternative_labels) in loaded_set:
            data = data.to(device)
            labels = labels.to(device)
            alternative_labels = alternative_labels.to(device)
            bsz = data.shape[0]
            if model_name=='concept_attention':
                _, _, c_emb, c_pred, _, _, _ = model(data)
                y_pred = torch.zeros(bsz, model.n_labels).to(model.device)
                prototypes = model.get_prototypes().unsqueeze(0).expand(bsz, -1, -1)
                c_pred, c_emb = apply_intervention(c_pred, c_emb, prototypes, proportion)
                for i in range(model.n_labels):
                    weights = model.cls.weights_generator[i](c_emb) # batch, n_concepts, 1
                    y_pred[:,i] = torch.bmm(c_pred.unsqueeze(1), weights).squeeze() 
            else:
                y_pred, c_pred = model(data)
                if proportion>0:
                    c_pred, _ = apply_intervention(c_pred, c_pred.unsqueeze(-1), c_pred.unsqueeze(-1), proportion, method='lf_cbm')
                    y_pred = model.classifier(c_pred)   
            y_preds = torch.cat([y_preds, y_pred.argmax(1)], axis=0)
    if y_preds.is_cuda:
        y_preds = y_preds.cpu()
    y_preds = y_preds[1:].numpy().astype(int)
    y_true = y_true.astype(int)
    acc = np.sum(y_preds==y_true) / len(y_true)
    return acc 

def intervention(path, loaded_test, trues, prop_list, device, args):
    #print('funciton device', device)
    #print('model device', model.device)
    df = pd.DataFrame()

    if args.method in ['concept_attention']:
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
        model = Concept_Attention(**model_args).to(device)
        model.load_state_dict(saved_model.state_dict())
    else:
        saved_model = torch.load(f'{path}/cbm.pth', weights_only=False, map_location=torch.device(f'cuda:{device}' if torch.cuda.is_available() else 'cpu'))
        
        model_args = {
            'n_concepts': saved_model.classifier[0].in_features,
            'n_labels': saved_model.classifier[0].out_features,
            'backbone': args.backbone,
            'device': device,
            'fine_tune': False,
            'label_free': saved_model.label_free,
            'task_interpretable': saved_model.task_interpretable,
        }

        model = cbm_model(**model_args).to(device)
        model.load_state_dict(saved_model.state_dict())

    for prop in tqdm(prop_list):
        acc = evaluate_model_with_intervention(model, trues, loaded_test, prop, args.method, device)
        d = {'proportion':prop, 'accuracy':acc}
        df = pd.concat([df, pd.DataFrame([d])], ignore_index=True)

    return df


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

def find_best_permutation(pred_concept, true_concept, emb, metric='f1'):
    num_classes = pred_concept.shape[1]
    
    # Compute pairwise F1 scores
    cost_matrix = np.zeros((num_classes, num_classes))
    for i in range(num_classes):
        for j in range(num_classes):
            cost_matrix[i, j] = f1_score(true_concept[:, i], pred_concept[:, j], average='macro')
    
    # Solve the assignment problem (maximize total F1 score)
    row_ind, col_ind = linear_sum_assignment(cost_matrix, maximize=True)
    
    # Reorder the columns of pred
    reordered_pred = pred_concept[:, col_ind]
    avg_f1 = np.mean([cost_matrix[i, j] for i, j in zip(row_ind, col_ind)])
    avg_acc=0

    if isinstance(emb, np.ndarray):
        return reordered_pred, avg_acc, avg_f1, emb[:,col_ind,:]
    else:
        return reordered_pred, avg_acc, avg_f1

# Apply clustering to get the thrshold to use for the concept predictions
def compute_average_threshold(tensor):
    thresholds = []
    for col in range(tensor.shape[1]):
        column_data = tensor[:, col].reshape(-1, 1)
        kmeans = KMeans(n_clusters=2, random_state=0).fit(column_data)
        centers = kmeans.cluster_centers_.flatten()
        threshold = np.mean(centers)
        thresholds.append(threshold)
    average_threshold = np.mean(thresholds)
    return average_threshold

def main(args):  
    # device check
    if torch.cuda.is_available():
        device = args.device
    else:
        device = "cpu"

    root = os.getcwd()
    path = os.path.join(root, f"results/{args.method}/{args.backbone}/{args.dataset}/{args.seed}")

    #with open(f'{path}/concept_attention_config.json', 'r') as file:
    #    saved_args = json.load(file)   
    saved_args = {}
    saved_args['batch_size']=128
    saved_args['val_size']=0.1
    saved_args['num_workers']=3
    
    # Load dataset
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
        classes = None
    else:
        raise ValueError('Dataset not yet implemented!')
    
    # load the arrays containing the truth values
    true_concept = np.load(f'{path}/concept_ground_truth.npy')
    true_task = np.load(f'{path}/task_ground_truth.npy')

    # load both concept and task predictions
    pred_concept = np.load(f'{path}/concept_predictions.npy')
    
    pred_task = np.load(f'{path}/task_predictions.npy')  
    pred_task = np.argmax(pred_task, axis=1)

    if args.method in ['cbm', 'protopnet', 'cbm_supervised', 'lf_cbm']:
        #th = compute_average_threshold(pred_concept)
        th = 0.5
        pred_concept = np.where(pred_concept>th,1,0)
        pred_embs = np.copy(pred_concept)
        pred_embs = pred_embs[..., np.newaxis]
    else:
        pred_concept = np.where(pred_concept>0.5,1,0)
        pred_embs = np.load(f'{path}/concept_embeddings.npy')

    if args.dataset in ['MNIST_sum', 'MNIST_even_odd', 'CIFAR100', 'CUB200', 'Skin']:

        # get the correct permutation of concept labels that maximizes the concept accuracy    
        pred_concept, concept_accuracy, concept_f1, pred_embs = find_best_permutation(pred_concept, true_concept, pred_embs)
        print("Concept Accuracy:", concept_accuracy, "Concept macro F1:", concept_f1)

        # compute CLIP metric
        #sil, ch, db = clip_clustering(test_loader, pred_concept)
        sil, ch, db = 0, 0, 0
        
        # compute CAS metric
        step = 5
        concept_auc, _ = concept_alignment_score(pred_embs, true_concept, true_task, step) # use concept embedding for concept attention and concept scores for the other methodologies
        print('CAS:', concept_auc)
        df = pd.DataFrame([{'Concept accuracy':concept_accuracy, 'Concept macro F1-score':concept_f1, 'CAS':concept_auc, 'Silhouette':sil, 'Calinski-Harabasz':ch, 'Davies-Bouldin':db}])
        df.to_csv(f'{path}/concept_results.csv')

        # compute intervention
    intervention_df = intervention(path, test_loader, true_task, np.arange(0,1.1,0.1), device, args)
    intervention_df.to_csv(f'{path}/interventions.csv')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="A script that trains the Concept Attention Model")
    parser.add_argument('--seed', type=int, default=0, help="Seed of the experiment")
    parser.add_argument('--backbone', type=str, default='resnet', help="Backbone used for the visual feature extraction")
    parser.add_argument('--dataset', type=str, default='MNIST_even_odd', help="Name of the dataset used for the experiment")
    parser.add_argument('--method', type=str, default='lf_cbm', help="Methodology used for the experiment")
    parser.add_argument('--device', type=int, default=1, help="Device used for the experiment")
    args = parser.parse_args()  
    main(args)