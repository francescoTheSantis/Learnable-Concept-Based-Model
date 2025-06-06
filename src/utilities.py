import torch
import torch.nn as nn
import os
import torch.nn.functional as F
import numpy as np
import cv2
from torchvision.datasets.folder import default_loader
from torchvision.datasets.utils import download_url
from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from torchvision import transforms
import random
from tqdm.auto import tqdm
from tqdm import tqdm
import scipy
import sklearn
import tensorflow as tf
import captum
from captum.attr import Occlusion, LayerGradCam, Saliency
from captum.attr import visualization as viz
import matplotlib.image as mpimg
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
from matplotlib.gridspec import GridSpec
import scienceplots
from scipy.ndimage import zoom

plt.style.use(['science', 'ieee', 'no-latex'])

def set_seed(seed):
    
    # Set seed for torch
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    # Ensure deterministic behavior in torch
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False 

        
def freeze_params(params):
    for param in params:
        param.requires_grad = False
    
    
def unfreeze_params(params):
    for param in params:
        param.requires_grad = True

        
def clamp(x):
    return torch.clamp(x, min = torch.finfo(torch.float32).eps)


# Similarity loss used by label-free CBM
def cos_similarity_cubed_single(clip_feats, target_feats):
    """
    Substract mean from each vector, then raises to third power and compares cos similarity
    Does not modify any tensors in place
    Only compares first neuron to first concept etc.
    """

    clip_feats = clip_feats.float()
    clip_feats = clip_feats - torch.mean(clip_feats, dim=0, keepdim=True)
    target_feats = target_feats - torch.mean(target_feats, dim=0, keepdim=True)

    clip_feats = clip_feats**3
    target_feats = target_feats**3

    clip_feats = clip_feats/torch.norm(clip_feats, p=2, dim=0, keepdim=True)
    target_feats = target_feats/torch.norm(target_feats, p=2, dim=0, keepdim=True)

    similarities = torch.sum(target_feats*clip_feats, dim=0)
    return similarities



def KL_divergence(logit, alpha):
    n_concepts = logit.shape[1]
    # get the probability form the logit and compute the average over the batch dimension
    pi = torch.sigmoid(logit).mean(dim=0)
    first_term = (1 - pi) * torch.log(clamp(1-pi)/(1-alpha))
    second_term = pi * torch.log(clamp(pi)/alpha) 
    kl_div = (first_term + second_term).sum()
    kl_div = kl_div/(n_concepts)**0.5
    return kl_div

def Gate_penalty(logit, alpha=None):
    pi = torch.sigmoid(logit)
    return pi.mean()

    
def plot_training_curves(task_losses, task_losses_val, gate_penalty_losses, gate_penalty_losses_val, 
                         reconstruction_losses, reconstruction_losses_val, dim=(22,5), folder=None, validation_step=1):
    
    if gate_penalty_losses==None and gate_penalty_losses_val==None and reconstruction_losses==None and reconstruction_losses_val==None:
        n=1
        flag=False
    else:
        n=3
        flag=True
        
    fig, ax = plt.subplots(1,n)
    fig.set_size_inches(dim)

    num_epochs = len(task_losses)

    if flag:
        ax[0].set_title('Task Loss')
        ax[0].plot(range(1,num_epochs+1), task_losses, label='Training')
        ax[0].plot(range(1, num_epochs + 1, validation_step), task_losses_val, label='Validation')
        ax[0].grid()
        ax[0].set_xlabel("Epochs")
        ax[0].legend()
        
        ax[1].set_title('Gate penalty Loss')
        ax[1].plot(range(1,num_epochs+1), gate_penalty_losses, label='Training')
        ax[1].plot(range(1, num_epochs + 1, validation_step), gate_penalty_losses_val, label='Validation')
        ax[1].grid()
        ax[1].set_xlabel("Epochs")
        ax[1].legend()

        ax[2].set_title('Reconstruction Loss')
        ax[2].plot(range(1,num_epochs+1), reconstruction_losses, label='Training')
        ax[2].plot(range(1, num_epochs + 1, validation_step), reconstruction_losses_val, label='Validation')
        ax[2].grid()
        ax[2].set_xlabel("Epochs")
        ax[2].legend()
    else:
        ax.set_title('Task Loss')
        ax.plot(range(1,num_epochs+1), task_losses, label='Training')
        ax.plot(range(1, num_epochs + 1, validation_step), task_losses_val, label='Validation')
        ax.grid()
        ax.set_xlabel("Epochs")
        ax.legend()
    
    plt.tight_layout()
    if folder != None:
        if not os.path.exists(folder):
            os.makedirs(folder)
        plt.savefig(folder+"/training_curves_concept_attention.pdf")
    plt.show()
    

def plot_training_curves_2(loss, val_loss, dim=(22,5), folder=None):
    fig, ax = plt.subplots()
    fig.set_size_inches(dim)
    num_epochs = len(loss)   
    #ax.set_title('Loss')
    ax.plot(range(1,num_epochs+1), loss, label='Training')
    ax.plot(range(1,num_epochs+1), val_loss, label='Validation')
    ax.grid()
    ax.set_xlabel("Epochs")
    ax.legend()
    plt.tight_layout()
    if folder != None:
        if not os.path.exists(folder):
            os.makedirs(folder)
        plt.savefig(folder+"/training_curves_e2e.pdf")
    plt.show()

    
def concept_distribution(c_preds, n_concepts, dim=(15,4)):
    data_np = c_preds.detach().cpu().numpy()
    fig, axs = plt.subplots(1,n_concepts)
    fig.set_size_inches(dim)
    for i in range(n_concepts):
        counts, bin_edges = np.histogram(data_np[:, i], bins=20)
        axs[i].bar(bin_edges[:-1], counts, width=np.diff(bin_edges), align='edge')
        axs[i].set_title(f'Distribution of Concept {i}')
        axs[i].set_xlabel('Value')
        axs[i].set_xlim([0,1])
        axs[i].grid()
    plt.tight_layout() 
    plt.show()
    
    
def concept_dist_per_label(n_labels, y_preds, c_preds, folder=None, dim = (25,4)):
    fig, ax = plt.subplots(1, n_labels)
    fig.set_size_inches(dim)

    if n_labels==1:
        ax = [ax, None]
    
    for label in range(n_labels):
        idxs = np.argwhere(y_preds==label).squeeze()
        selected = torch.where(c_preds[idxs,:]>0.5, 1, 0)
        probabilities = (selected.sum(0)/selected.shape[0]).cpu().numpy()
        variables = np.arange(selected.shape[1])
        ax[label].barh(variables, probabilities)
        #ax[label].set_yticklabels([f'Concept {i}' for i in variables])
        ax[label].set_title(f'Label: {label}')
        ax[label].set_xlim([0,1])
    fig.tight_layout()
    if folder != None:
        if not os.path.exists(folder):
            os.makedirs(folder)        
        plt.savefig(folder+'/concept_activations_x_label.pdf')
    plt.show()

    for label in range(n_labels):
        idxs = np.argwhere(y_preds==label).squeeze()
        selected = torch.where(c_preds[idxs,:]>0.5, 1, 0).cpu().detach()
        unique_rows, counts = np.unique(selected, axis=0, return_counts=True)
        probabilities = counts / selected.shape[0]
        d = {str(k):round(100*v,1) for k,v in zip(unique_rows, probabilities)}
        sorted_dict = dict(reversed(sorted(d.items(), key=lambda item: item[1])))
        print(f'Class {label}')
        cnt=0
        for row, prob in sorted_dict.items():
            print(f"Row: {row}, Probability: {prob}%")
            if cnt==3:
                break
            cnt+=1
        print()
    
def sample_explanation(model, test_logits, test_dataset, n_concepts, idx, folder, dim=(5,5), device='cuda', classes=None, alpha=0.3):    
    original_idx = int(idx)
    folder += f'/{original_idx}'
    if not os.path.exists(folder):
        os.makedirs(folder) 

    img = test_dataset[original_idx][0]
    if classes!=None:
        original_label = classes[int(test_dataset[original_idx][1])]
    else:
        original_label = int(test_dataset[original_idx][1])       
    img = img.to(device).unsqueeze(0)
    y_pred, logits, _, c_pred, c_logit, _, _ = model(img)

    #concept_idxs = torch.argwhere(c_logit.squeeze()>0)
    concept_idxs = torch.argsort(c_logit.squeeze(), descending=True).cpu().numpy()
    prediction = int(torch.argmax(y_pred).cpu().detach().numpy())
    importance_values = logits[0, :, prediction].cpu().detach().numpy()
    if concept_idxs.size==1:
        concept_idxs = np.expand_dims(concept_idxs,0)

    # per i concetti attivi voglio calcolare la gradcam, sovrapporla all'immagine e salvare il risultato in folder.
    model.eval()
    target_layer = model.backbone.resnet[-2][1].conv2
    grads_path = []
    for i, concept_idx in enumerate(concept_idxs):
        if i==3:
            break
        # create gradcam
        img = test_dataset[original_idx][0]
        img = img.to(device).unsqueeze(0)
        img.requires_grad_()
        layer_gradcam = LayerGradCam(model, target_layer)

        attributions_lgc = layer_gradcam.attribute(img, additional_forward_args=True, target=int(concept_idx))
        upsamp_attr_lgc = LayerGradCam.interpolate(attributions_lgc, img.shape[2:]).squeeze().cpu().detach().numpy()

        img = img[0,:,:,:].permute(1,2,0).cpu().detach().numpy()
        img = (img - img.min()) / (img.max() - img.min())
        fig, ax = plt.subplots()
        fig.set_size_inches(4,4)
        ax.imshow(img)
        if c_pred.squeeze()[concept_idx]>0.5:
            ax.imshow(np.where(upsamp_attr_lgc>0,upsamp_attr_lgc,0), cmap='bwr', alpha=alpha)
        ax.set_ylabel(f'C{concept_idx}', fontsize=90)     
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_xlabel('')        
        ax.set_title('')   
        plt.savefig(f'{folder}/concept_{concept_idx}.png')
        plt.close()
        grads_path.append(f'{folder}/concept_{concept_idx}.png')

    fig, axs = plt.subplots(1,2)
    fig.set_size_inches((11,5))
    ax=axs[0]
    
    model.eval()
    img = test_dataset[original_idx][0]
    img = img.to(device).unsqueeze(0)
    img = img[0,:,:,:].permute(1,2,0).cpu().detach().numpy()
    img = zoom(img, (5, 5, 1)) 
    img = (img - img.min()) / (img.max() - img.min())
    ax.imshow(img)
    ax.set_ylabel('')
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    ax.minorticks_off()
    ax.axis('off')
    #plt.savefig(f'{folder}/original_image.png')

    ax = axs[1]
    ax.set_title(f'Class: {original_label}', fontsize=30)

    importance_values = [importance_values[x] for i, x in enumerate(concept_idxs[:3])]
    colors = ['blue' if x>=0 else 'red' for x in importance_values] #np.where(importance_values >= 0, 'blue', 'red')
    ax.barh([f'Concept_{x}' for x in range(n_concepts) if x in concept_idxs[:3]], 
            importance_values, color=colors)
    #ax.axvline(x=sum(importance_values), color='tab:green', linestyle='--', label='Logit sum')
    ax.axvline(x=0, color='black', linestyle='-')
    ax.set_xlabel('Importance', fontsize=30)
    ax.tick_params(axis='x', labelsize=28)
    ax.set_ylabel('')
    ax.minorticks_off()

    y = list(range(len(concept_idxs)))
    ax.set_yticklabels([''] * len(y))
    grad_imgs = [mpimg.imread(img_path) for img_path in grads_path]

    # Add images to y-ticks
    for i in range(3):
        imagebox = OffsetImage(grad_imgs[i], zoom=0.045)  # Adjust zoom as needed
        ab = AnnotationBbox(imagebox, (0, i), frameon=False, box_alignment=(1.03, 0.5))
        ax.add_artist(ab)

    #plt.tight_layout()
    #plt.savefig(f'{folder}/explanatoin.pdf')
    plt.subplots_adjust(wspace=0.5)
    plt.savefig(f'{folder}/explanatoin.pdf', bbox_inches='tight', pad_inches=0.5)


    
def learned_concepts(model, c_logits, test_dataset, K, n_concepts, folder=None, dim=(10,4), 
                     device='cuda', alpha=0, classes=None, concept_order=None):
    
    model.eval()
    target_layer = model.backbone.resnet[-2][1].conv2

    heights = [224 for x in range(K)]
    widths = [224 for x in range(n_concepts)]
    
    fig_width = 9

    # calculate the figure width
    fig_height = fig_width * sum(heights) / sum(widths)
    
    fig, ax = plt.subplots(K, n_concepts, figsize=(fig_width,fig_height), gridspec_kw={'height_ratios':heights}) 
    #fig, ax = plt.subplots(n_concepts, K, figsize=(fig_width,fig_height), gridspec_kw={'height_ratios':heights}) 

#    fig.subplots_adjust(wspace=0, hspace=0)

    if concept_order==None:
        concept_order = list(range(n_concepts))
    
    for j, concept in enumerate(concept_order):
        _, top_k_indices = torch.topk(c_logits[:,concept], K)
        p = np.random.choice(top_k_indices.shape[0], K, replace=False)
        top_k_indices = top_k_indices[p]    
        for idx, image_idx in enumerate(top_k_indices.cpu().numpy()):
            img = test_dataset[int(image_idx)][0].unsqueeze(0).to(device)
            
            
            # create gradcam
            img.requires_grad_()
            layer_gradcam = LayerGradCam(model, target_layer)

            attributions_lgc = layer_gradcam.attribute(img, additional_forward_args=True, target=concept)
            upsamp_attr_lgc = LayerGradCam.interpolate(attributions_lgc, img.shape[2:]).squeeze().cpu().detach().numpy()
            
            
            #print(img.shape)
            img = img[0,:,:,:].permute(1,2,0).cpu().detach().numpy()
            img = (img - img.min()) / (img.max() - img.min())
            ax[idx, j].imshow(img)
            
            ax[idx, j].imshow(upsamp_attr_lgc, cmap='RdYlGn', alpha=alpha)
            
            if idx==0:
                ax[idx, j].set_title(f'C{j}', fontsize=12)
            else:
                ax[idx, j].set_title('')
            
            ax[idx, j].axis('off')
            
    plt.subplots_adjust(wspace=0, hspace=0, left=0, right=1, bottom=0, top=1)
    
    if folder != None:
        folder = os.path.join(folder, 'dictionary')
        if not os.path.exists(folder):
            os.makedirs(folder)        
        plt.savefig(folder+f'/concept_activations_{alpha}.pdf')
    plt.show()
    
    
    
    


################################################################################
## Metrics utilities
################################################################################



def concept_similarity_matrix(
    concept_representations,
    compute_ratios=False,
    eps=1e-5,
):
    """
    Computes a matrix such that its (i,j)-th entry represents the average
    normalized dot product between samples representative of concept i and
    samples representative of concept j.
    This metric is defined by Chen et al. in "Concept Whitening for
    Interpretable Image Recognition" (https://arxiv.org/abs/2002.01650)

    :param List[np.ndarray] concept_representations: A list of tensors
        containing representative samples for each concept. The i-th element
        of this list must be a tensor whose first dimension is the batch
        dimension and last dimension is the channel dimension.
    :param bool compute_ratios: If True, then each element in the output matrix
        is  the similarity ratio coefficient as defined by Chen et al.. This is
        the ratio between the inter-similarity of (i, j) and the square root
        of the product between the intra-similarity of concepts i and j.
    :param float eps: A small value for numerical stability when performing
        divisions.
    """
    num_concepts = len(concept_representations)
    result = np.zeros((num_concepts, num_concepts), dtype=np.float32)
    m_representations_normed = {}
    intra_dot_product_means_normed = {}
    for i in range(num_concepts):
        m_representations_normed[i] = (
            concept_representations[i] /
            np.linalg.norm(concept_representations[i], axis=-1, keepdims=True)

        )
        intra_dot_product_means_normed[i] = np.matmul(
            m_representations_normed[i],
            m_representations_normed[i].transpose()
        ).mean()

        if compute_ratios:
            result[i, i] = 1.0
        else:
            result = np.matmul(
                concept_representations[i],
                concept_representations[i].transpose()
            ).mean()

    for i in range(num_concepts):
        for j in range(i + 1, num_concepts):
            inter_dot = np.matmul(
                m_representations_normed[i],
                m_representations_normed[j].transpose()
            ).mean()
            if compute_ratios:
                result[i, j] = np.abs(inter_dot) / np.sqrt(np.abs(
                    intra_dot_product_means_normed[i] *
                    intra_dot_product_means_normed[j]
                ))
            else:
                result[i, j] = np.matmul(
                    concept_representations[i],
                    concept_representations[j].transpose(),
                ).mean()
            result[j, i] = result[i, j]

    return result


################################################################################
## Alignment Functions
################################################################################


def find_max_alignment(matrix):
    """
    Finds the maximum (greedy) alignment between columns in this matrix and
    its rows. It returns a list `l` with as many elements as columns in the input
    matrix such that l[i] is the column best aligned with row `i` given the
    scores in `matrix`.
    For this, we proceed in a greedy fashion where we bind columns with rows
    in descending order of their values in the matrix.

    :param np.ndarray matrix: A matrix with at least as many rows as columns.

    :return List[int]: the column-to-row maximum greedy alignment.
    """
    sorted_inds = np.dstack(
        np.unravel_index(np.argsort(-matrix.ravel()), matrix.shape)
    )[0]
    result_alignment = [None for _ in range(matrix.shape[1])]
    used_rows = set()
    used_cols = set()
    for (row, col) in sorted_inds:
        if (col in used_cols) or (row in used_rows):
            # Then this is not something we can use any more
            continue
        # Else, let's add this mapping into our alignment!
        result_alignment[col] = row
        used_rows.add(row)
        used_cols.add(col)
        if len(used_rows) == matrix.shape[1]:
            # Then we are done in here!
            break
    return result_alignment


def max_alignment_matrix(matrix):
    """
    Helper function that computes the (greedy) max alignment of the input
    matrix and it rearranges so that each column is aligned to its corresponding
    row. In this case, this means that the diagonal matrix of the resulting
    matrix will correspond to the entries in `matrix` that were aligned.

    :param np.ndarray matrix: A matrix with at least as many rows as columns.

    :return np.ndarray: A square matrix representing the column-aligned matrix
        of the given input tensor.
    """
    inds = find_max_alignment(matrix)
    return np.stack(
        [matrix[inds[i], :] for i in range(matrix.shape[1])],
        axis=0
    )


################################################################################
## Purity Matrix Computation
################################################################################


def concept_purity_matrix(
    c_soft,
    c_true,
    concept_label_cardinality=None,
    predictor_model_fn=None,
    predictor_train_kwags=None,
    test_size=0.2,
    ignore_diags=False,
    jointly_learnt=False,
):
    """
    Computes a concept purity matrix where the (i,j)-th entry represents the
    predictive accuracy of a classifier trained to use the i-th concept's soft
    labels (as given by c_soft_train) to predict the ground truth value of the
    j-th concept.

    This process is informally defined only for binary concepts by Mahinpei et
    al.'s in "Promises and Pitfalls of Black-Box Concept Learning Models".
    Nevertheless, this method supports both binary concepts (given as a 2D
    matrix in c_soft) or categorical concepts (given by a list of 2D matrices
    in argument c_soft).

    :param Or[np.ndarray, List[np.ndarray]] c_soft: Predicted set of "soft"
        concept representations by a concept encoder model applied to the
        testing data. This argument must be an np.ndarray with shape
        (n_samples, ..., n_concepts) where the concept representation may be
        of any rank as long as the last dimension is the dimension used to
        separate distinct concept representations. If concepts have distinct
        array shapes for their representations, then this argument is expected
        to be a list of `n_concepts` np.ndarrays where the i-th element in the
        list is an array with shape (n_samples, ...) containing the tensor
        representation of the i-th concept.
        Note that in either case we only require that the first dimension.
    :param np.ndarray c_true: Ground truth concept values in one-to-one
        correspondence with concepts in c_soft. Shape must be
        (n_samples, n_concepts).
    :param List[int] concept_label_cardinality: If given, then this is a list
        of integers such that its i-th index contains the number of classes
        that the it-th concept may take. If not given, then we will assume that
        all concepts have the same cardinality as the number of activations in
        their soft representations.
    :param Function[(int, int), sklearn-like Estimator] predictor_model_fn: A
        function generator that takes as an argument two values, the number of
        classes for the input concept and the number of classes for the output
        target concept, respectively, and produces an sklearn-like Estimator
        which one can train for predicting a concept given another concept's
        soft concept values. If not given then we will use a 3-layer ReLU MLP
        as our predictor.
    :param Dict[Any, Any] predictor_train_kwags: optional arguments to pass
        the estimator being when calling its `fit` method.
    :param float test_size: A value in [0, 1] indicating the fraction of the
        given data that will be used to evaluate the trained concept-based
        classifier. The rest of the data will be used for training said
        classifier.

    :return np.ndarray: a matrix with shape (n_concepts, n_concepts)
        where the (i,j)-th entry specifies the testing AUC of using the i-th
        concept soft representations to predict the j-th concept.
    """
    # Start by handling default arguments
    predictor_train_kwags = predictor_train_kwags or {}

    # Check that their rank is the expected one
    assert len(c_true.shape) == 2, (
        f'Expected testing concept predictions to be a matrix with shape '
        f'(n_samples, n_concepts) but instead got a matrix with shape '
        f'{c_true.shape}'
    )

    # Construct a list concept_label_cardinality that maps a concept to the
    # cardinality of its label set as specified by the testing data
    (n_samples, n_true_concepts) = c_true.shape
    if isinstance(c_soft, np.ndarray):
        n_soft_concepts = c_soft.shape[-1]
    else:
        assert isinstance(c_soft, list), (
            f'c_soft must be passed as either a list or a np.ndarray. '
            f'Instead we got an instance of "{type(c_soft).__name__}".'
        )
        n_soft_concepts = len(c_soft)

    assert n_soft_concepts >= n_true_concepts, (
        f'Expected at least as many soft concept representations as true '
        f'concepts labels. However we received {n_soft_concepts} soft concept '
        f'representations per sample while we have {n_true_concepts} true '
        f'concept labels per sample.'
    )

    if isinstance(c_soft, np.ndarray):
        # Then, all concepts must have the same representation size
        assert c_soft.shape[0] == c_true.shape[0], (
            f'Expected a many test soft-concepts as ground truth test '
            f'concepts. Instead got {c_soft.shape[0]} soft-concepts '
            f'and {c_true.shape[0]} ground truth test concepts.'
        )
        if concept_label_cardinality is None:
            concept_label_cardinality = [2 for _ in range(n_soft_concepts)]
        # And for simplicity and consistency, we will rewrite c_soft as a
        # list such that i-th entry contains an array with shape
        # (n_samples, repr_size) indicating the representation of the i-th
        # concept for all samples
        new_c_soft = [None for _ in range(n_soft_concepts)]
        for i in range(n_soft_concepts):
            if len(c_soft.shape) == 1:
                # If it is a scalar representation, then let's make it explicit
                new_c_soft[i] = np.expand_dims(c_soft[..., i], axis=-1)
            else:
                new_c_soft[i] = c_soft[..., i]
        c_soft = new_c_soft
    else:
        # Else, time to infer these values from the given list of soft
        # labels
        assert isinstance(c_soft, list), (
            f'c_soft must be passed as either a list or a np.ndarray. '
            f'Instead we got an instance of "{type(c_soft).__name__}".'
        )
        if concept_label_cardinality is None:
            concept_label_cardinality = [None for _ in range(n_soft_concepts)]
            for i, soft_labels in enumerate(c_soft):
                concept_label_cardinality[i] = max(soft_labels.shape[-1], 2)
                assert soft_labels.shape[0] == c_true.shape[0], (
                    f"For concept {i}'s soft labels, we expected "
                    f"{c_true.shape[0]} samples as we were given that many "
                    f"in the ground-truth array. Instead we found "
                    f"{soft_labels.shape[0]} samples."
                )

    # Handle the default parameters for both the generating function and
    # the concept label cardinality
    if predictor_model_fn is None:
        # Then by default we will use a simple MLP classifier with one hidden
        # ReLU layer with 32 units in it
        def predictor_model_fn(
            output_concept_classes=2,
        ):
            estimator = tf.keras.models.Sequential([
                tf.keras.layers.Dense(
                    32,
                    activation='relu',
                    name="predictor_fc_1",
                ),
                tf.keras.layers.Dense(
                    output_concept_classes if output_concept_classes > 2 else 1,
                    # We will merge the activation into the loss for numerical
                    # stability
                    activation=None,
                    name="predictor_fc_out",
                ),
            ])
            if jointly_learnt:
                loss = tf.nn.sigmoid_cross_entropy_with_logits
            else:
                loss = (
                    tf.keras.losses.SparseCategoricalCrossentropy(
                        from_logits=True
                    ) if output_concept_classes > 2 else
                    tf.keras.losses.BinaryCrossentropy(
                        from_logits=True,
                    )
                )
            estimator.compile(
                # Use ADAM optimizer by default
                optimizer='adam',
                # Note: we assume labels come without a one-hot-encoding in the
                #       case when the concepts are categorical.
                loss=loss,
            )
            return estimator

    predictor_train_kwags = predictor_train_kwags or {
        'epochs': 25,
        'batch_size': min(512, n_samples),
        'verbose': 0,
    }

    # Time to start formulating our resulting matrix
    result = np.zeros((n_soft_concepts, n_true_concepts), dtype=np.float32)

    # Split our test data into two subsets as we will need to train
    # a classifier and then use that trained classifier in the remainder of the
    # data for computing our scores
    train_indexes, test_indexes = train_test_split(
        list(range(n_samples)),
        test_size=test_size,
    )

    for src_soft_concept in tqdm(range(n_soft_concepts)):

        # Construct a test and training set of features for this concept
        concept_soft_train_x = c_soft[src_soft_concept][train_indexes, ...]
        concept_soft_test_x = c_soft[src_soft_concept][test_indexes, ...]
        if len(concept_soft_train_x.shape) == 1:
            concept_soft_train_x = tf.expand_dims(
                concept_soft_train_x,
                axis=-1,
            )
            concept_soft_test_x = tf.expand_dims(
                concept_soft_test_x,
                axis=-1,
            )
        if jointly_learnt:
            # Construct a new estimator for performing this prediction
            output_size = 0
            for tgt_true_concept in range(n_true_concepts):
                output_size += (
                    concept_label_cardinality[tgt_true_concept]
                    if concept_label_cardinality[tgt_true_concept] > 2
                    else 1
                )
            estimator = predictor_model_fn(output_size)
            # Train it
            estimator.fit(
                concept_soft_train_x,
                c_true[train_indexes, :],
                **predictor_train_kwags,
            )
            # Compute the AUC of this classifier on the test data
            preds = estimator.predict(concept_soft_test_x)
            for tgt_true_concept in range(n_true_concepts):
                true_concepts = c_true[test_indexes, tgt_true_concept]
                used_preds = preds[:, tgt_true_concept]
                if concept_label_cardinality[tgt_true_concept] > 2:
                    # Then lets apply a softmax activation over all the probability
                    # classes
                    used_preds = scipy.special.softmax(used_preds, axis=-1)

                    # And make sure we only compute the AUC of labels that are
                    # actually used
                    used_labels = np.sort(np.unique(true_concepts))

                    # And select just the labels that are in fact being used
                    true_concepts = tf.keras.utils.to_categorical(
                        true_concepts,
                        num_classes=concept_label_cardinality[tgt_true_concept],
                    )[:, used_labels]
                    used_preds = used_preds[:, used_labels]
                if len(np.unique(true_concepts)) > 1:
                    auc = sklearn.metrics.roc_auc_score(
                        true_concepts,
                        used_preds,
                        multi_class='ovo',
                    )
                else:
                    if concept_label_cardinality[tgt_true_concept] <= 2:
                        used_preds = (
                            scipy.special.expit(used_preds) >= 0.5
                        ).astype(np.int32)
                    else:
                        used_preds = np.argmax(used_preds, axis=-1)
                        true_concepts = np.argmax(true_concepts, axis=-1)
                    auc = sklearn.metrics.accuracy_score(
                        true_concepts,
                        used_preds,
                    )

                # Finally, time to populate the actual entry of our resulting
                # matrix
                result[src_soft_concept, tgt_true_concept] = auc
        else:
            for tgt_true_concept in range(n_true_concepts):
                # Let's populate the (i,j)-th entry of our matrix by first
                # training a classifier to predict the ground truth value of
                # concept j using the soft-concept labels for concept i.
                if ignore_diags and (src_soft_concept == tgt_true_concept):
                    # Then for simplicity sake we will simply set this to one
                    # as it is expected to be perfectly predictable
                    result[src_soft_concept, tgt_true_concept] = 1
                    continue

                # Construct a new estimator for performing this prediction
                estimator = predictor_model_fn(
                    concept_label_cardinality[tgt_true_concept]
                )
                # Train it
                estimator.fit(
                    concept_soft_train_x,
                    c_true[train_indexes, tgt_true_concept:(tgt_true_concept + 1)],
                    **predictor_train_kwags,
                )

                # Compute the AUC of this classifier on the test data
                preds = estimator.predict(concept_soft_test_x)
                true_concepts = c_true[test_indexes, tgt_true_concept]
                if concept_label_cardinality[tgt_true_concept] > 2:
                    # Then lets apply a softmax activation over all the
                    # probability classes
                    preds = scipy.special.softmax(preds, axis=-1)

                    # And make sure we only compute the AUC of labels that are
                    # actually used
                    used_labels = np.sort(np.unique(true_concepts))

                    # And select just the labels that are in fact being used
                    true_concepts = tf.keras.utils.to_categorical(
                        true_concepts,
                        num_classes=concept_label_cardinality[tgt_true_concept],
                    )[:, used_labels]
                    preds = preds[:, used_labels]

                auc = sklearn.metrics.roc_auc_score(
                    true_concepts,
                    preds,
                    multi_class='ovo',
                )

                # Finally, time to populate the actual entry of our resulting
                # matrix
                result[src_soft_concept, tgt_true_concept] = auc

    # And that's all folks
    return result


def encoder_concept_purity_matrix(
    encoder_model,
    features,
    concepts,
    predictor_model_fn=None,
    predictor_train_kwags=None,
    test_size=0.2,
    jointly_learnt=False,
):
    """
    Computes a concept purity matrix where the (i,j)-th entry represents the
    predictive accuracy of a classifier trained to use the i-th concept's soft
    representation (as given by the encoder model) to predict the ground truth
    value of the j-th concept.

    This process is informally defined only for binary concepts by Mahinpei et
    al.'s in "Promises and Pitfalls of Black-Box Concept Learning Models".
    Nevertheless, this method supports arbitrarily-shaped concept
    representations (given as a (n_samples, ..., n_concepts) tensor output when
    using the encoder's predict method) as well as concepts with different
    representation shapes (given as a list of n_concepts  tensors with shapes
    (n_samples, ...) when using the encoder's predict method).

    :param skelearn-like Estimator encoder_model: An encoder estimator capable
        of extracting concept representations from a set of features. For
        example, this estimator may produce a vector of binary concept
        probabilities for each sample (i.e., in the case of all concepts being
        binary) or a list of vectors representing probability distributions over
        the labels for each concept (i.e., in the case of one or more concepts
        being categorical).
    :param np.ndarray features: An array of testing samples with shape
        (n_samples, ...) used to compute the purity matrix.
    :param np.ndarray concepts: Ground truth concept values in one-to-one
        correspondence with samples in features. Shape must be
        (n_samples, n_concepts).
    :param Function[(int,), sklearn-like Estimator] predictor_model_fn: A
        function generator that takes as an argument the number of
        the output target concept and produces an sklearn-like Estimator
        which one can train for predicting a concept given another concept's
        soft concept values. If not given then we will use a 3-layer ReLU MLP
        as our predictor.
    :param Dict[Any, Any] predictor_train_kwags: optional arguments to pass
        the estimator when calling its `fit` method.
    :param float test_size: A value in [0, 1] indicating the fraction of the
        given data that will be used to evaluate the trained concept-based
        classifier. The rest of the data will be used for training said
        classifier.

    :return np.ndarray: a matrix with shape (n_concepts, n_concepts)
        where the (i,j)-th entry specifies the testing AUC of using the i-th
        concept soft labels to predict the j-th concept.
    """
    # Simply use the concept purity matrix computation defined above when given
    # soft concepts as computed by the encoder model
    return concept_purity_matrix(
        c_soft=encoder_model.predict(features),
        c_true=concepts,
        predictor_model_fn=predictor_model_fn,
        predictor_train_kwags=predictor_train_kwags,
        test_size=test_size,
        jointly_learnt=jointly_learnt,
    )


def oracle_purity_matrix(
    concepts,
    concept_label_cardinality=None,
    predictor_model_fn=None,
    predictor_train_kwags=None,
    test_size=0.2,
    jointly_learnt=False,
):
    """
    Computes an oracle's concept purity matrix where the (i,j)-th entry
    represents the predictive accuracy of a classifier trained to use the i-th
    concept (ground truth) to predict the ground truth value of the j-th
    concept.

    :param np.ndarray concepts: Ground truth concept values. Shape must be
        (n_samples, n_concepts).
    :param List[int] concept_label_cardinality: If given, then this is a list
        of integers such that its i-th index contains the number of classes
        that the it-th concept may take. If not given, then we will assume that
        all concepts are binary (i.e., concept_label_cardinality[i] = 2 for all
        i).
    :param Function[(int,), sklearn-like Estimator] predictor_model_fn: A
        function generator that takes as an argument the number of
        the output target concept and produces an sklearn-like Estimator
        which one can train for predicting a concept given another concept's
        soft concept values. If not given then we will use a 3-layer ReLU MLP
        as our predictor.
    :param Dict[Any, Any] predictor_train_kwags: optional arguments to pass
        the estimator when calling its `fit` method.
    :param float test_size: A value in [0, 1] indicating the fraction of the
        given data that will be used to evaluate the trained concept-based
        classifier. The rest of the data will be used for training said
        classifier.

    :return np.ndarray: a matrix with shape (n_concepts, n_concepts)
        where the (i,j)-th entry specifies the testing AUC of using the i-th
        concept label to predict the j-th concept.
    """

    return concept_purity_matrix(
        c_soft=concepts,
        c_true=concepts,
        concept_label_cardinality=concept_label_cardinality,
        predictor_model_fn=predictor_model_fn,
        predictor_train_kwags=predictor_train_kwags,
        test_size=test_size,
        ignore_diags=True,
        jointly_learnt=jointly_learnt,
    )


################################################################################
## Purity Metrics
################################################################################

def normalize_impurity(impurity, n_concepts):
    return impurity / (n_concepts / 2)


def oracle_impurity_score(
    c_soft,
    c_true,
    predictor_model_fn=None,
    predictor_train_kwags=None,
    test_size=0.2,
    norm_fn=lambda x: np.linalg.norm(x, ord='fro'),
    oracle_matrix=None,
    purity_matrix=None,
    output_matrices=False,
    alignment_function=None,
    concept_label_cardinality=None,
    jointly_learnt=False,
    include_diagonal=True,
):
    """
    Returns the oracle impurity score (OIS) of the given soft concept
    representations `c_soft` with respect to their corresponding ground truth
    concepts `c_true`. This value is higher if concepts encode unnecessary
    information from other concepts in their soft representation and lower
    otherwise. If zero, then all soft concept labels are considered to be
    "pure".

    We compute this metric by calculating the norm of the absolute difference
    between the purity matrix derived from the soft concepts and the purity
    matrix derived from an oracle model. This oracle model is trained using
    the ground truth labels instead of the soft labels and may capture trivial
    relationships between different concept labels.

    :param Or[np.ndarray, List[np.ndarray]] c_soft: Predicted set of "soft"
        concept representations by a concept encoder model applied to the
        testing data. This argument must be an np.ndarray with shape
        (n_samples, ..., n_concepts) where the concept representation may be
        of any rank as long as the last dimension is the dimension used to
        separate distinct concept representations. If concepts have distinct
        array shapes for their representations, then this argument is expected
        to be a list of `n_concepts` np.ndarrays where the i-th element in the
        list is an array with shape (n_samples, ...) containing the tensor
        representation of the i-th concept.
        Note that in either case we only require that the first dimension.
    :param np.ndarray c_true: Ground truth concept values in one-to-one
        correspondence with concepts in c_soft. Shape must be
        (n_samples, n_concepts).
    :param Function[(int,), sklearn-like Estimator] predictor_model_fn: A
        function generator that takes as an argument the number of
        the output target concept and produces an sklearn-like Estimator
        which one can train for predicting a concept given another concept's
        soft concept values. If not given then we will use a 3-layer ReLU MLP
        as our predictor.
    :param Dict[Any, Any] predictor_train_kwags: optional arguments to pass
        the estimator being when calling its `fit` method.
    :param float test_size: A value in [0, 1] indicating the fraction of the
        given data that will be used to evaluate the trained concept-based
        classifier. The rest of the data will be used for training said
        classifier.
    :param Function[(np.ndarray), float] norm_fn: A norm function applicable to
        a 2D numpy matrix representing the absolute difference between the
        oracle purity score matrix and the predicted purity score matrix. If not
        given then we will use the 2D Frobenius norm.
    :param np.ndarray oracle_matrix: If given, then this must be a 2D array with
        shape (n_concepts, n_concepts) such that the (i, j)-th entry represents
        the AUC of an oracle that predicts the value of concept j given the
        ground truth of concept i. If not given, then this matrix will be
        computed using the ground truth concept labels.
    :param np.ndarray purity_matrix: If given, then this must be a 2D array with
        shape (n_concepts, n_concepts) such that the (i, j)-th entry represents
        the AUC of predicting the value of concept j given the soft
        representation of concept i. If not given, then this matrix will be
        computed using the purity scores from the input soft representations.
    :param bool output_matrices: If True then this method will output a tuple
        (score, purity_matrix, oracle_matrix) containing the computed purity
        score, purity matrix, and oracle matrix given this function's
        arguments.
    :param Function[(np.ndarray), np.ndarray] alignment_function: an optional
        alignment function that takes as an input an (k, n_concepts) purity
        matrix, where k >= n_concepts and its (i, j) value is the AUC of
        predicting true concept j using soft representations i, and returns a
        (n_concepts, n_concepts) matrix where a subset of n_concepts soft
        concept representations has been aligned in a bijective fashion with
        the set of all ground truth concepts.


    :returns Or[Tuple[float, np.ndarray, np.ndarray], float]: If output_matrices
        is False (default behavior) then the output will be a non-negative float
        in [0, 1] representing the degree to which individual concepts
        representations encode unnecessary information for other concepts. Higher
        values mean more impurity and the concepts are considered to be pure if
        the returned value is 0. If output_matrices is True, then the output
        will be a tuple (score, purity_matrix, oracle_matrix) containing the
        computed purity score, purity matrix, and oracle matrix given this
        function's arguments. If alignment_function is given, then the purity
        matrix will be a tuple (purity_matrix, aligned_purity_matrix) containing
        the pre and post alignment purity matrices, respectively.
    """

    # Now the concept_label_cardinality vector from the given soft labels
    (n_samples, n_concepts) = c_true.shape
    if concept_label_cardinality is None:
        concept_label_cardinality = [
            len(set(c_true[:, i]))
            for i in range(n_concepts)
        ]
    # First compute the predictor soft-concept purity matrix
    if purity_matrix is not None:
        pred_matrix = purity_matrix
    else:
        pred_matrix = concept_purity_matrix(
            c_soft=c_soft,
            c_true=c_true,
            predictor_model_fn=predictor_model_fn,
            predictor_train_kwags=predictor_train_kwags,
            test_size=test_size,
            concept_label_cardinality=concept_label_cardinality,
            jointly_learnt=jointly_learnt,
        )

    # Compute the oracle's purity matrix
    if oracle_matrix is None:
        oracle_matrix = oracle_purity_matrix(
            concepts=c_true,
            concept_label_cardinality=concept_label_cardinality,
            predictor_model_fn=predictor_model_fn,
            predictor_train_kwags=predictor_train_kwags,
            test_size=test_size,
            jointly_learnt=jointly_learnt,
        )

    # Finally, compute the norm of the absolute difference between the two
    # matrices
    if alignment_function is not None:
        # Then lets make sure we align our prediction matrix correctly
        aligned_matrix = alignment_function(pred_matrix)
        if not include_diagonal:
            used_aligned_matrix = np.copy(aligned_matrix)
            np.fill_diagonal(used_aligned_matrix, 1)
            used_oracle_matrix = np.copy(oracle_matrix)
            np.fill_diagonal(used_oracle_matrix, 1)
        else:
            used_oracle_matrix = oracle_matrix
            used_aligned_matrix = aligned_matrix
        score = norm_fn(np.abs(used_oracle_matrix - used_aligned_matrix))
        if output_matrices:
            return score, (pred_matrix, aligned_matrix), oracle_matrix
        return score

    if not include_diagonal:
        used_pred_matrix = np.copy(pred_matrix)
        np.fill_diagonal(used_pred_matrix, 1)
        used_oracle_matrix = np.copy(oracle_matrix)
        np.fill_diagonal(used_oracle_matrix, 1)
    else:
        used_oracle_matrix = oracle_matrix
        used_pred_matrix = pred_matrix
    score = normalize_impurity(
        impurity=norm_fn(np.abs(used_oracle_matrix - used_pred_matrix)),
        n_concepts=n_concepts,
    )
    if output_matrices:
        return score, pred_matrix, oracle_matrix
    return score


def encoder_oracle_impurity_score(
    encoder_model,
    features,
    concepts,
    predictor_model_fn=None,
    predictor_train_kwags=None,
    test_size=0.2,
    norm_fn=lambda x: np.linalg.norm(x, ord='fro'),
    oracle_matrix=None,
    output_matrices=False,
    purity_matrix=None,
    alignment_function=None,
    include_diagonal=True,
):
    """
    Returns the OIS of the concept representations generated by
    `encoder_model` when given `features` with respect to their corresponding
    ground truth concepts `concepts`. This value is higher if concepts encode
    unnecessary information from other concepts in their soft representation and
    lower otherwise. If zero, then all soft concept labels are considered to be
    "pure".

    We compute this metric by calculating the norm of the absolute difference
    between the purity matrix derived from the soft concepts and the purity
    matrix derived from an oracle model. This oracle model is trained using
    the ground truth labels instead of the soft labels and may capture trivial
    relationships between different concept labels.

    :param skelearn-like Estimator encoder_model: An encoder estimator capable
        of extracting concepts from a set of features. This estimator may
        produce a vector of binary concept probabilities for each sample (i.e.,
        in the case of all concepts being binary) or a list of vectors
        representing probability distributions over the labels for each concept
        (i.e., in the case of one or more concepts being categorical).
    :param np.ndarray features: An array of testing samples with shape
        (n_samples, ...) used to compute the purity matrix.
    :param np.ndarray concepts: Ground truth concept values in one-to-one
        correspondence with samples in features. Shape must be
        (n_samples, n_concepts).
    :param Function[(int,), sklearn-like Estimator] predictor_model_fn: A
        function generator that takes as an argument the number of
        the output target concept and produces an sklearn-like Estimator
        which one can train for predicting a concept given another concept's
        soft concept values. If not given then we will use a 3-layer ReLU MLP
        as our predictor.
    :param Dict[Any, Any] predictor_train_kwags: optional arguments to pass
        the estimator when calling its `fit` method.
    :param float test_size: A value in [0, 1] indicating the fraction of the
        given data that will be used to evaluate the trained concept-based
        classifier. The rest of the data will be used for training said
        classifier.
    :param Function[(np.ndarray), float] norm_fn: A norm function applicable to
        a 2D numpy matrix representing the absolute difference between the
        oracle purity score matrix and the predicted purity score matrix. If not
        given then we will use the 2D Frobenius norm.
    :param np.ndarray oracle_matrix: If given, then this must be a 2D array with
        shape (n_concepts, n_concepts) such that the (i, j)-th entry represents
        the AUC of an oracle that predicts the value of concept j given the
        ground truth of concept i. If not given, then this matrix will be
        computed using the ground truth concept labels.
    :param np.ndarray purity_matrix: If given, then this must be a 2D array with
        shape (n_concepts, n_concepts) such that the (i, j)-th entry represents
        the AUC of predicting the value of concept j given the soft
        representation generated by the encoder for concept i. If not given,
        then this matrix will be computed using the purity scores from the input
        encoder's soft representations.
    :param bool output_matrices: If True then this method will output a tuple
        (score, purity_matrix, oracle_matrix) containing the computed purity
        score, purity matrix, and oracle matrix given this function's
        arguments.
    :param Function[(np.ndarray,), np.ndarray] alignment_function: an optional
        alignment function that takes as an input an (k, n_concepts) purity
        matrix, where k >= n_concepts and its (i, j) value is the AUC of
        predicting true concept j using soft representations i, and returns a
        (n_concepts, n_concepts) matrix where a subset of n_concepts soft
        concept representations has been aligned in a bijective fashion with
        the set of all ground truth concepts.

    :returns Or[Tuple[float, np.ndarray, np.ndarray], float]: If output_matrices
        is False (default behavior) then the output will be a non-negative float
        representing the degree to which individual concepts in the given
        bottleneck encode unnecessary information for other concepts. Higher
        values mean more impurity and the concepts are considered to be pure if
        the returned value is 0. If output_matrices is True, then the output
        will be a tuple (score, purity_matrix, oracle_matrix) containing the
        computed purity score, purity matrix, and oracle matrix given this
        function's arguments. If alignment_function is given, then the purity
        matrix will be a tuple (purity_matrix, aligned_purity_matrix) containing
        the pre and post alignment purity matrices, respectively.
    """
    # Simply use the concept purity metric defined above when given
    # soft concepts as computed by the encoder model
    return oracle_impurity_score(
        c_soft=encoder_model.predict(features),
        c_true=concepts,
        predictor_model_fn=predictor_model_fn,
        predictor_train_kwags=predictor_train_kwags,
        test_size=test_size,
        norm_fn=norm_fn,
        oracle_matrix=oracle_matrix,
        purity_matrix=purity_matrix,
        output_matrices=output_matrices,
        alignment_function=alignment_function,
        include_diagonal=include_diagonal,
    )