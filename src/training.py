import torch
import torch.nn as nn
import torch.optim as optim
import os
from tqdm.auto import tqdm
from torch.optim import lr_scheduler
import torch.nn.functional as F
import copy
from utilities import *
from models import * 

#####################################
############### LCBM ################
#####################################

def evaluate_concept_attention(concept_attention, loaded_set, n_concepts, n_labels, alpha, task_loss_form, recon_loss, gate_penalty_form, reconstruct_embedding, device='cuda'):
    concept_attention.eval()
    emb_size = concept_attention.emb_size
    run_task_loss = 0
    run_gate_penalty_loss = 0
    run_reconstruction_loss = 0
    c_preds = torch.zeros(1, n_concepts).to(device)
    c_logits = torch.zeros(1, n_concepts).to(device)
    y_preds = torch.zeros(1, n_labels).to(device)
    y_true = torch.zeros(1).to(device)
    c_true = torch.zeros(1, n_concepts).to(device)
    c_embs = torch.zeros(1, n_concepts, emb_size).to(device)
    with torch.no_grad():
        for (data, labels, alternative_labels) in loaded_set:
            data = data.to(device)
            labels = labels.to(device)
            alternative_labels = alternative_labels.to(device)
            bsz = data.shape[0]
            y_pred, _, c_emb, c_pred, c_logit, img_embedding, z = concept_attention(data)
            if reconstruct_embedding:
                reconstruction_loss = recon_loss(img_embedding, z).sum(-1).mean()
            else:
                reconstruction_loss = recon_loss(data, z).mean()
            task_loss = task_loss_form(y_pred, labels)
            gate_penalty = gate_penalty_form(c_logit, alpha)    
            run_task_loss += task_loss.item()
            run_gate_penalty_loss += gate_penalty.item() 
            run_reconstruction_loss += reconstruction_loss.item() 
            c_preds = torch.cat([c_preds, c_pred], axis=0)
            y_preds = torch.cat([y_preds, y_pred], axis=0)
            y_true = torch.cat([y_true, labels], axis=0)
            c_logits = torch.cat([c_logits, c_logit], axis=0)
            c_true = torch.cat([c_true, alternative_labels], axis=0)
            c_embs = torch.cat([c_embs, c_emb], axis=0)
    return run_task_loss/len(loaded_set), run_gate_penalty_loss/len(loaded_set), run_reconstruction_loss/len(loaded_set), c_preds[1:,:], y_preds[1:,:], y_true[1:], c_logits[1:,:], c_true[1:,:], c_embs[1:,:,:]


def train_concept_attention(concept_attention, train_loader, val_loader, test_loader, n_concepts, n_labels, lr, num_epochs, step_size, gamma, alpha=0.1, lambda_gate=1, lambda_recon=1e-3, lambda_task=1, verbose=0, device='cuda', train=True, binarization_step=5, KL_penalty=True, accumulation=0, reconstruct_embedding=True, folder=None, warm_up=0):
    optimizer = optim.Adam(concept_attention.parameters(), lr=lr)
    task_losses = []
    gate_penalty_losses = []
    reconstruction_losses = []
    task_losses_val = []
    gate_penalty_losses_val = []
    reconstruction_losses_val = []
    ckpt_path = f"{folder}/ckpt"
    if not os.path.exists(ckpt_path):
        os.makedirs(ckpt_path)       
    if n_labels==1:
        task_loss_form = nn.MSELoss(reduction='mean')
    else:
        task_loss_form = nn.CrossEntropyLoss(reduction='mean')
        
    recon_loss = nn.MSELoss(reduction='none')

    if KL_penalty:
        gate_penalty_form = KL_divergence
    else:
        gate_penalty_form = Gate_penalty
    scheduler = lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)
    best_val = np.finfo(np.float64).max
    if train==False:
        task_loss, gate_penalty, reconstruction, c_preds, y_preds, y_true, c_logits, c_true, c_embs = evaluate_concept_attention(concept_attention, test_loader, n_concepts, n_labels, alpha, task_loss_form, recon_loss, gate_penalty_form, reconstruct_embedding, device)
        print(f"Task: {lambda_task * task_loss};\tGate penalty: {lambda_gate * gate_penalty};\tRecon : {lambda_recon * reconstruction}")        
        return c_preds, y_preds, y_true, c_logits, c_true, reconstruction, c_embs
    for epoch in range(1, num_epochs+1):
        concept_attention.train()
        run_task_loss = 0
        run_gate_penalty_loss = 0
        run_reconstruction_loss = 0
        print('Bound applied for numerical stability:', concept_attention.bound)
        if epoch==10 and concept_attention.bound != -1:
            concept_attention.bound = 50
        elif epoch==20 and concept_attention.bound == 50:
            concept_attention.bound = -1
        if epoch==binarization_step:
            print('Binarization applied!')
            concept_attention.b = torch.Tensor([1/3]).to(device)
        if epoch==warm_up:
            print('Last warm-up epoch')
        for (step, (data, labels, alternative_labels)) in tqdm(enumerate(train_loader), total=len(train_loader)):
            data = data.to(device)
            bsz = data.shape[0]
            labels = labels.to(device)
            y_pred, _, c_emb, c_pred, c_logit, img_embedding, z = concept_attention(data)
            if reconstruct_embedding:
                reconstruction_loss = recon_loss(img_embedding, z).sum(-1).mean()
            else:
                reconstruction_loss = recon_loss(data, z).mean()
                
            task_loss = task_loss_form(y_pred, labels)
            gate_penalty = gate_penalty_form(c_logit, alpha)
            if epoch<=warm_up:
                loss = lambda_recon * reconstruction_loss
            else:
                loss = (lambda_task * task_loss) + (lambda_recon * reconstruction_loss) + (lambda_gate * gate_penalty)  
            if accumulation>0:
                loss = loss / accumulation
            loss.backward()
            if accumulation>0:
                if (step + 1) % accumulation == 0 or (step + 1 == len(train_loader)):
                    optimizer.step()  
                    optimizer.zero_grad() 
            else:
                optimizer.step()  
                optimizer.zero_grad() 
            run_task_loss += task_loss.item()
            run_gate_penalty_loss += gate_penalty.item()
            run_reconstruction_loss += reconstruction_loss.item()
        scheduler.step()
        task_losses.append(run_task_loss/len(train_loader))
        gate_penalty_losses.append(run_gate_penalty_loss/len(train_loader))
        reconstruction_losses.append(run_reconstruction_loss/len(train_loader))
        # evaluate on the validation
        task_loss_val, gate_penalty_val, reconstruction_val, c_preds, y_preds, y_true, c_logits, c_true, _ = evaluate_concept_attention(concept_attention, val_loader, n_concepts, n_labels, alpha, task_loss_form, recon_loss, gate_penalty_form, reconstruct_embedding, device)
        task_losses_val.append(task_loss_val)
        gate_penalty_losses_val.append(gate_penalty_val)
        reconstruction_losses_val.append(reconstruction_val)
        if verbose>=1:
            print(f'Epoch {epoch}')
            print(f"Validation: \nTask: {lambda_task * task_loss_val};\tGate penalty: {lambda_gate * gate_penalty_val};\tRecon: {lambda_recon * reconstruction_val}")
            print('Expected activations:', (list(map(lambda x: round(x,3), (torch.where(c_pred>0.5,1,0).sum(dim=0)/c_pred.shape[0]).detach().cpu().numpy()))))
            print('Expected number of active concepts per sample:', (torch.where(c_pred>0.5,1,0).sum(dim=0)/c_pred.shape[0]).detach().cpu().numpy().mean()*n_concepts)
            if verbose>1:
                print('logits:\n'), 
                for l in range(verbose):
                    print(c_logit[l,:].detach().cpu().numpy().round(2))
            print()
        selection_loss = task_loss_val
        if folder != None and selection_loss<best_val:
            torch.save(concept_attention, f"{folder}/concept_attention.pth")  
            best_val = selection_loss  
        # evaluate on test: this is done in order to compute the mutual information metric (computed for each epoch)
        if folder != None:
            _, _, reconstruction_val, c_preds, y_preds, y_true, c_logits, c_true, c_embs = evaluate_concept_attention(concept_attention, test_loader, n_concepts, n_labels, alpha, task_loss_form, recon_loss, gate_penalty_form, reconstruct_embedding, device)

            # store: c_preds, y_preds, y_true, c_true, c_emb
            tensors = [c_preds, y_preds, c_embs]
            names = ['concept_predictions', 'task_predictions','concept_embeddings']
            for name, tensor in zip(names, tensors):
                if tensor.is_cuda:
                    tensor = tensor.cpu()
                array = tensor.numpy()
                np.save(f'{ckpt_path}/{name}_{epoch}.npy', array)
            if verbose>=1:
                y_true = y_true.cpu().numpy()
                y_preds = y_preds.argmax(-1).detach().cpu().numpy()
                print("Task Accuracy:", np.mean(y_true==y_preds))
                
    return concept_attention, task_losses, gate_penalty_losses, reconstruction_losses, task_losses_val, gate_penalty_losses_val, reconstruction_losses_val


#####################################
################ E2E ################
#####################################


def evaluate_e2e(model, loaded_set, n_labels, task_loss_form, device='cuda'):
    model.eval()
    run_task_loss = 0
    y_preds = torch.zeros(1, n_labels).to(device)
    y_true = torch.zeros(1).to(device)
    with torch.no_grad():
        for (data, labels, alternative_labels) in loaded_set:
            data = data.to(device)
            bsz = data.shape[0]
            labels = labels.to(device)
            y_pred = model(data)
            task_loss = task_loss_form(y_pred, labels)   
            run_task_loss += task_loss.item()
            y_preds = torch.cat([y_preds, y_pred], axis=0)
            y_true = torch.cat([y_true, labels], axis=0)
    return run_task_loss/len(loaded_set), y_preds, y_true
        
    
def train_e2e(model, train_loader, val_loader, test_loader, n_labels, lr, num_epochs, step_size, gamma, verbose=0, device='cuda', train=True, accumulation=0, folder=None):
    optimizer = optim.AdamW(model.parameters(), lr=lr)
    task_losses = []
    task_losses_val = []
    if n_labels==1:
        task_loss_form = nn.MSELoss(reduction='mean')
    else:
        task_loss_form = nn.CrossEntropyLoss(reduction='mean')
    best_val = np.finfo(np.float64).max
    scheduler = lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)
    if train==False:
        task_loss, y_preds, y_true = evaluate_e2e(model, test_loader, n_labels, task_loss_form, device)
        print(f"Task: {task_loss}")
        return y_preds, y_true
    for epoch in range(1, num_epochs+1):
        model.train()
        run_task_loss = 0
        for (step, (data, labels, alternative_labels)) in tqdm(enumerate(train_loader), total=len(train_loader)):
            data = data.to(device)
            bsz = data.shape[0]
            labels = labels.to(device)
            y_pred = model(data)
            task_loss = task_loss_form(y_pred, labels)
            if accumulation>0:
                task_loss = task_loss / accumulation
            task_loss.backward()
            if accumulation>0:
                if (step + 1) % accumulation == 0 or (step + 1 == len(train_loader)):
                    optimizer.step()  
                    optimizer.zero_grad() 
            else:
                optimizer.step()  
                optimizer.zero_grad() 
            run_task_loss += task_loss.item()
        scheduler.step()
        task_losses.append(run_task_loss/len(train_loader))
        # evaluate on validation
        task_loss_val, _, _, = evaluate_e2e(model, test_loader, n_labels, task_loss_form, device)
        task_losses_val.append(task_loss_val)
        if verbose>=1:
            print(f'Epoch {epoch}')
            print(f"Validation Task loss: {task_loss_val}")
            print()
        if folder != None and task_loss_val<best_val:
            torch.save(model, f"{folder}/e2e.pth")  
            best_val = task_loss_val           
    return model, task_losses, task_losses_val


################################################
################ CBM label free ################
################################################


def evaluate_cbm(cbm, loaded_set, n_concepts, n_labels, task_loss_form, regularization_loss_form, device='cuda', supervised=False):
    cbm.eval()
    run_task_loss = 0
    run_reg_loss = 0
    c_preds = torch.zeros(1, n_concepts).to(device)
    y_preds = torch.zeros(1, n_labels).to(device)
    y_true = torch.zeros(1).to(device)
    c_true = torch.zeros(1, n_concepts).to(device)
    with torch.no_grad():
        for (data, labels, alternative_labels, clip_scores) in loaded_set:
            data = data.to(device)
            bsz = data.shape[0]
            alternative_labels = alternative_labels.to(device)
            labels = labels.to(device)
            y_pred, c_pred = cbm(data)
            task_loss = task_loss_form(y_pred, labels)
            
            if supervised:
                regularization_loss = 0
                for c_idx in range(n_concepts):
                    regularization_loss += regularization_loss_form(c_pred[:,c_idx], alternative_labels[:,c_idx].float())
                regularization_loss = regularization_loss.sum(-1)                
                regularization_loss = regularization_loss/n_concepts
                regularization_loss = regularization_loss.mean()
            else:
                regularization_loss = -regularization_loss_form(clip_scores.to(device), c_pred).mean()

            run_task_loss += task_loss.item()
            run_reg_loss += regularization_loss.item()
  
            c_preds = torch.cat([c_preds, c_pred], axis=0)
            y_preds = torch.cat([y_preds, y_pred], axis=0)
            y_true = torch.cat([y_true, labels], axis=0)
            c_true = torch.cat([c_true, alternative_labels], axis=0)
    return run_task_loss/len(loaded_set), regularization_loss/len(loaded_set), c_preds[1:,:], y_preds[1:,:], y_true[1:], c_true[1:,:]


def train_cbm(cbm, train_loader, val_loader, test_loader, n_concepts, n_labels, lr, num_epochs, step_size, gamma, verbose=0, device='cuda', train=True, folder=None, supervised=False):
    optimizer = optim.Adam(cbm.parameters(), lr=lr)
    task_losses = []
    regularization_losses = []
    task_losses_val = []
    regularization_losses_val = []
    ckpt_path = f"{folder}/ckpt"
    if not os.path.exists(ckpt_path):
        os.makedirs(ckpt_path)       
    if n_labels==1:
        task_loss_form = nn.MSELoss(reduction='mean')
    else:
        task_loss_form = nn.CrossEntropyLoss(reduction='mean')

    if supervised:
        regularization_loss_form = nn.BCELoss(reduction='none')
    else:
        regularization_loss_form = cos_similarity_cubed_single

    scheduler = lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)
    best_val = np.finfo(np.float64).max
    if train==False:
        task_loss, reg_loss, c_preds, y_preds, y_true, c_true = evaluate_cbm(cbm, test_loader, n_concepts, n_labels, task_loss_form, regularization_loss_form, device, supervised)
        print(f"Task: {task_loss};\tRegularization penalty: {reg_loss}")        
        return c_preds, y_preds, y_true, c_true
    for epoch in range(1, num_epochs+1):
        print('epoch:', epoch)
        cbm.train()
        run_task_loss = 0
        run_reg_loss = 0
        for (step, (data, labels, alternative_labels, clip_scores)) in tqdm(enumerate(train_loader), total=len(train_loader)):
            data = data.to(device)
            bsz = data.shape[0]
            labels = labels.to(device)
            y_pred, c_pred = cbm(data)
            
            task_loss = task_loss_form(y_pred, labels)
            regularization_loss = 0
            
            if supervised:
                for c_idx in range(n_concepts):
                    regularization_loss += regularization_loss_form(c_pred[:,c_idx], alternative_labels[:,c_idx].float().to(device))
                regularization_loss = regularization_loss.sum(-1)                
                regularization_loss = regularization_loss/n_concepts
                regularization_loss = regularization_loss.mean()
            else:
                regularization_loss = -regularization_loss_form(clip_scores.to(device), c_pred).mean()

            loss = task_loss + regularization_loss
            loss.backward()
            optimizer.step()  
            optimizer.zero_grad() 

            run_task_loss += task_loss.item()
            run_reg_loss += regularization_loss.item()
        scheduler.step()
        task_losses.append(run_task_loss/len(train_loader))
        regularization_losses.append(run_reg_loss/len(train_loader))
        # evaluate on the validation
        task_loss_val, regularization_loss_val, c_preds, y_preds, y_true, c_true = evaluate_cbm(cbm, val_loader, n_concepts, n_labels, task_loss_form, regularization_loss_form, device, supervised)
        task_losses_val.append(task_loss_val)
        regularization_losses_val.append(regularization_loss_val)

        if verbose>=1:
            print('Task loss:', task_loss_val, 'Reg. Loss:', regularization_loss_val)
        
        if folder != None and task_loss_val<best_val:
            torch.save(cbm, f"{folder}/cbm.pth")  
            best_val = task_loss_val 

        # evaluate on test: this is done in order to compute the mutual information metric (computed for each epoch)
        if folder != None:
            _, _, c_preds, y_preds, y_true, c_true = evaluate_cbm(cbm, test_loader, n_concepts, n_labels, task_loss_form, regularization_loss_form, device, supervised)

            # store: c_preds, y_preds, y_true, c_true, c_emb
            tensors = [c_preds, y_preds, y_true, c_true]
            names = ['concept_predictions', 'task_predictions', 'task_ground_truth', 'concept_ground_truth']
            for name, tensor in zip(names, tensors):
                if tensor.is_cuda:
                    tensor = tensor.cpu()
                array = tensor.numpy()
                np.save(f'{ckpt_path}/{name}_{epoch}.npy', array)
            if verbose>=1:
                y_true = y_true.cpu().numpy()
                y_preds = y_preds.argmax(-1).detach().cpu().numpy()
                c_true = c_true.cpu().numpy()
                c_preds = c_preds.cpu().numpy()
                acc_conc = 0
                for c in range(n_concepts):
                    acc_conc = np.mean(y_true==y_preds)
                print("Task Accuracy:", np.mean(y_true==y_preds))
                
    return cbm, task_losses, regularization_loss, task_losses_val, regularization_loss_val
