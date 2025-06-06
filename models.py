import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import vgg19, densenet201, resnet18
from transformers import ViTModel
from utilities import * 


class DenseNet(nn.Module):
    def __init__(self):
        super(DenseNet, self).__init__()
        # get the pretrained DenseNet201 network
        self.densenet = densenet201(pretrained=True)
        # disect the network to access its last convolutional layer
        self.features_conv = self.densenet.features
        # add the average global pool
        self.global_avg_pool = nn.AvgPool2d(kernel_size=7, stride=1)
    
    def forward(self, x):
        x = self.features_conv(x)
        x = self.global_avg_pool(x)
        x = x.flatten(start_dim=1)
        return x
    

class VGG(nn.Module):
    def __init__(self):
        super(VGG, self).__init__()
        # get the pretrained VGG19 network
        self.vgg = vgg19(pretrained=True)
        # disect the network to access its last convolutional layer
        self.features_conv = self.vgg.features[:36]
        # get the max pool of the features stem
        self.max_pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
        self.linear = self.vgg.classifier[0]
    
    def forward(self, x):
        x = self.features_conv(x)
        # apply the remaining pooling
        x = self.max_pool(x)
        x = x.flatten(start_dim=1)
        x = self.linear(x)
        return x
    
    
class ViT(nn.Module):
    def __init__(self, fine_tune):
        super(ViT, self).__init__()
        # Load the pre-trained ViT-B/16 model from HuggingFace
        self.vit = ViTModel.from_pretrained('google/vit-base-patch16-224-in21k')
        
        for param in self.vit.parameters():
            param.requires_grad = fine_tune
            
    
    def forward(self, x):
        outputs = self.vit(pixel_values=x).last_hidden_state[:, 0]  # Use the [CLS] token representation
        return outputs
    
    
class ResNet(nn.Module):
    def __init__(self, fine_tune):
        super(ResNet, self).__init__()
        # Load the pre-trained ResNet-18 model
        self.resnet = resnet18(pretrained=True)
        # Remove the final fully connected layer (classifier)
        self.resnet = nn.Sequential(*list(self.resnet.children())[:-1])  # Remove the FC layer

        for param in self.resnet.parameters():
            param.requires_grad = fine_tune
            
    def forward(self, x):
        # Extract features, ResNet outputs [batch_size, 512, 1, 1]
        embeddings = self.resnet(x)  # This will output embeddings
        embeddings = embeddings.view(embeddings.size(0), -1)  # Flatten to [batch_size, 512]
        return embeddings
    

# Custom Reshape class
class Reshape(nn.Module):
    def __init__(self, shape1, shape2, shape3):
        super(Reshape, self).__init__()
        self.shape = [shape1, shape2, shape3]
        
    def forward(self, x):
        return x.view(x.size(0), *self.shape)

    
# Linear concept embedding model
class LinearModel(nn.Module):
    def __init__(self, emb_size, n_labels, use_bias=False, deep_parameterization=True, device='cuda'):
        super(LinearModel, self).__init__()
        
        self.emb_size = emb_size
        self.n_labels = n_labels
        self.use_bias = use_bias
        self.device = device
        self.deep_parameterization = deep_parameterization
        
        self.weights_generator = torch.nn.ModuleList()
        if self.use_bias:
            self.bias = nn.Parameter(torch.zeros(n_labels))
        for i in range(n_labels):
            if self.deep_parameterization:
                self.weights_generator.append(
                    nn.Sequential(
                        nn.Linear(emb_size, emb_size), 
                        nn.ReLU(), 
                        nn.Linear(emb_size, 1)
                    )
                )                    
            else:
                self.weights_generator.append(nn.Sequential(nn.Linear(emb_size, 1, bias=False)))
                
    def forward(self, c_emb, c_pred):
        bsz = c_emb.shape[0]
        n_concepts = c_emb.shape[1]
        y = torch.zeros(bsz, self.n_labels).to(self.device)
        logits = torch.zeros(bsz, n_concepts, self.n_labels).to(self.device)
        for i in range(self.n_labels):
            weights = self.weights_generator[i](c_emb) # batch, n_concepts, 1
            y[:,i] = torch.bmm(c_pred.unsqueeze(1), weights).squeeze() 
            if self.use_bias:
                y[:,i] += self.bias[i]
            logits[:, :, i] = weights.squeeze() * c_pred      
        return y, logits
    
    
class MaxNormActivation(nn.Module):
    def __init__(self, max_norm):
        super(MaxNormActivation, self).__init__()
        self.max_norm = max_norm

    def forward(self, x):
        # Compute the norm of each vector along dimension 1
        norms = torch.norm(x, dim=1, keepdim=True)  # Shape: [N, 1]
        # Calculate scaling factors for vectors exceeding max_norm
        scaling_factors = torch.clamp(self.max_norm / (norms + 1e-8), max=1.0)
        # Scale down vectors with norm greater than max_norm
        x_scaled = x * scaling_factors
        return x_scaled

    
class Concept_Attention(torch.nn.Module):
    def __init__(
            self,
            n_concepts,
            emb_size,
            n_labels,
            size,
            channels,
            embedding,
            backbone = 'resnet',
            device='cuda',
            deep_parameterization=False,
            use_bias = False,
            bound = None,
            multi_dist = False,
            concept_encoder = 'attention',
            expand_recon_bottleneck = False,
            fine_tune = False
    ):
        super().__init__()
        self.emb_size = emb_size
        self.n_concepts = n_concepts
        self.device = device
        self.use_bias = use_bias
        self.n_labels = n_labels
        self.eps = torch.finfo(torch.float32).eps
        self.b = torch.Tensor([2/3]).to(device)
        self.tau = 1
        self.size = size
        self.channels = channels
        self.bound = bound
        self.multi_dist = multi_dist
        self.concept_encoder = concept_encoder
        self.expand_recon_bottleneck = expand_recon_bottleneck
        self.deep_parameterization = deep_parameterization
        self.fine_tune = fine_tune
        
        if self.bound > 0:
            self.softsign = MaxNormActivation(bound**0.5) #nn.Tanh() #nn.Softsign()
        
        if backbone=='vgg':
            self.backbone = VGG().to(device)
            self.in_features = 4096   
        elif backbone=='densenet':
            self.backbone = DenseNet().to(device)
            self.in_features = 1920  
        elif backbone=='vit':
            self.backbone = ViT(self.fine_tune).to(device)
            self.in_features = 768 
        elif backbone=='resnet':
            self.backbone = ResNet(self.fine_tune).to(device)
            self.in_features = 512
        else:
            raise ValueError('Backbone not implemented!')

        #if not self.fine_tune:
        #    freeze_params(self.backbone.parameters())
        
        if self.concept_encoder=='attention':
            self.concept_query = nn.Parameter(torch.randn(n_concepts, emb_size))
            self.concept_embedding_generator = torch.nn.Sequential(
                    nn.Linear(self.in_features, n_concepts * emb_size),
                    nn.ReLU(),
                    nn.Linear(n_concepts * emb_size, n_concepts * emb_size),
                    #nn.LeakyReLU(0.1)
            )
            
            #self.projector = nn.Linear(self.emb_size, 10, bias=False)

        elif self.concept_encoder=='cem':
            self.concept_prob_predictors = torch.nn.ModuleList()
            for _ in range(self.n_concepts):
                self.concept_prob_predictors.append(
                    nn.Linear(self.emb_size, 1, bias=False)
                )
            self.concept_embedding_generator = torch.nn.Sequential(
                    nn.Linear(self.in_features, self.n_concepts * self.emb_size),
                    nn.ReLU(),
                    nn.Linear(self.n_concepts * self.emb_size, self.n_concepts * self.emb_size),
                    nn.LeakyReLU(0.1)
            )
        else:
            raise ValueError('Concept encoder not implemented!')

        self.cls = LinearModel(self.emb_size, self.n_labels, self.use_bias, self.deep_parameterization, self.device)
        
        if self.expand_recon_bottleneck:
            decoder_in = self.n_concepts * self.emb_size
        else:
            decoder_in = self.n_concepts
            
        if embedding:
            self.decoder = nn.Sequential(
                nn.Linear(decoder_in, self.in_features),
                nn.ReLU(),
                nn.Linear(self.in_features, self.in_features)
            )
        else:
            self.decoder = nn.Sequential(
                    nn.Linear(decoder_in, 256 * 7 * 7),
                    Reshape(256, 7, 7),                                     
                    nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),  
                    nn.BatchNorm2d(128),
                    nn.ReLU(True),
                    nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),  
                    nn.BatchNorm2d(64),
                    nn.ReLU(True),
                    nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),    
                    nn.BatchNorm2d(32),
                    nn.ReLU(True),
                    nn.ConvTranspose2d(32, 16, kernel_size=4, stride=2, padding=1),    
                    nn.BatchNorm2d(16),
                    nn.ReLU(True),
                    nn.ConvTranspose2d(16, 8, kernel_size=4, stride=2, padding=1),    
                    nn.BatchNorm2d(8),
                    nn.ReLU(True),
                    nn.Conv2d(8, 3, kernel_size=3, stride=1, padding=1),
                )                
                
                
    def clamp_min_max(self, x):
        return torch.clamp(x, min = torch.finfo(torch.float32).eps, max = 1-torch.finfo(torch.float32).eps)

    def reparameterize_bernoulli(self, logit_pi, training_flag):
        if training_flag:
            u = self.clamp_min_max(torch.rand_like(logit_pi).to(self.device))
            z = torch.sigmoid((logit_pi + torch.log(u/(1-u)))/self.b)
        else:
            z = torch.where(logit_pi>0, 1, 0).float()
        return z

    def apply_constraint(self, x):
        max_entry = (self.bound/self.emb_size)**0.5
        bounded_tensor =  self.softsign(x) * max_entry
        return bounded_tensor
    
    def get_prototypes(self):
        if self.concept_encoder=='cem':
            prototypes = []
            for i, concept_predictor in enumerate(self.concept_prob_predictors):
                prototypes.append(concept_prediction.weight.detach())
            return torch.cat(prototypes, axis=0)
        else:
            if self.bound > 0:
                return self.apply_constraint(self.concept_query)
            else:
                return self.concept_query
    
    def encode(self, x):
        img_embedding = self.backbone(x)
        bsz = x.shape[0] 
        if self.concept_encoder=='attention':
            c_emb = self.concept_embedding_generator(img_embedding).view(-1, self.n_concepts, self.emb_size)
            queries = self.concept_query
            if self.bound > 0:
                c_emb = self.apply_constraint(c_emb)
                queries = self.apply_constraint(queries)
            resized_queries = queries.unsqueeze(0).expand(bsz,-1,-1)
            concept_dot_products = torch.bmm(resized_queries, c_emb.permute(0,2,1))
            c_logit = torch.diagonal(concept_dot_products, dim1=-1, dim2=-2)
            if self.multi_dist:
                c_pred = F.gumbel_softmax(c_logit, tau=self.tau, hard=(not self.training)) 
            else:
                c_pred = self.reparameterize_bernoulli(c_logit, self.training) 
        elif self.concept_encoder=='cem':
            c_emb = self.concept_embedding_generator(img_embedding).view(-1, self.n_concepts, self.emb_size)
            c_pred_list, c_logit_list = [], []
            for i, concept_predictor in enumerate(self.concept_prob_predictors):
                c_logits = concept_predictor(c_emb[:,i,:])
                c_logit_list.append(c_logits.unsqueeze(1))
                c_pred = self.reparameterize_bernoulli(c_logits, self.training)
                c_pred_list.append(c_pred.unsqueeze(1))
            c_pred = torch.cat(c_pred_list, axis=1)[:,:,0]
            c_logit = torch.cat(c_logit_list, axis=1)[:,:,0]           
        return c_emb, c_pred, c_logit, img_embedding
    
    
    def forward(self, x, saliency=False):  
        c_emb, c_pred, c_logit, img_embedding = self.encode(x)
        if saliency:
            return c_logit
        else:
            y, logits = self.cls(c_emb, c_pred)
            if self.expand_recon_bottleneck:
                flattened_embs = (c_emb * c_pred[:,:,None]).flatten(start_dim=1) 
                z = self.decoder(flattened_embs) 
            else:
                z = self.decoder(c_pred) 
            return y, logits, c_emb, c_pred, c_logit, img_embedding, z

    

class e2e_model(torch.nn.Module):
    def __init__(
            self,
            n_labels,
            backbone = 'densenet',
            device='cuda',
            fine_tune=False
    ):
        super().__init__()
        self.device = device
        self.fine_tune = fine_tune
        
        if backbone=='vgg':
            self.backbone = VGG().to(device)
            self.in_features = 4096   
        elif backbone=='densenet':
            self.backbone = DenseNet().to(device)
            self.in_features = 1920  
        elif backbone=='vit':
            self.backbone = ViT(fine_tune).to(device)
            self.in_features = 768 
        elif backbone=='resnet':
            self.backbone = ResNet(fine_tune).to(device)
            self.in_features = 512
        else:
            raise ValueError('Backbone not implemented!')


        self.classifier = torch.nn.Sequential(
                nn.Linear(self.in_features, self.in_features),
                nn.ReLU(),
                nn.Linear(self.in_features, n_labels),
            )

    def forward(self, x):
        img_embeddings = self.backbone(x)
        labels = self.classifier(img_embeddings)
        return labels
    

class cbm_model(torch.nn.Module):
    def __init__(
            self,
            n_labels,
            n_concepts,
            backbone = 'densenet',
            device='cuda',
            label_free = True,
            task_interpretable = True,
            fine_tune = False
    ):
        super().__init__()
        self.device = device
        self.label_free = label_free
        self.task_interpretable = task_interpretable

        if backbone=='vgg':
            self.backbone = VGG().to(device)
            self.in_features = 4096   
        elif backbone=='densenet':
            self.backbone = DenseNet().to(device)
            self.in_features = 1920  
        elif backbone=='vit':
            self.backbone = ViT(fine_tune).to(device)
            self.in_features = 768 
        elif backbone=='resnet':
            self.backbone = ResNet(fine_tune).to(device)
            self.in_features = 512
        else:
            raise ValueError('Backbone not implemented!')

        if label_free:
             self.concept_encoder = torch.nn.Sequential(
                    nn.Linear(self.in_features, self.in_features),
                    nn.ReLU(),
                    nn.Linear(self.in_features, n_concepts),
                )           
        else:
            self.concept_encoder = torch.nn.Sequential(
                    nn.Linear(self.in_features, self.in_features),
                    nn.ReLU(),
                    nn.Linear(self.in_features, n_concepts),
                    nn.Sigmoid()
                )
        
        if task_interpretable:
            self.classifier = torch.nn.Sequential(
                nn.Linear(n_concepts, n_labels)
            )
        else:
            self.classifier = torch.nn.Sequential(
                    nn.Linear(n_concepts, n_concepts),
                    nn.ReLU(),
                    nn.Linear(n_concepts, n_labels),
                )

    def forward(self, x):
        img_embeddings = self.backbone(x)
        concepts = self.concept_encoder(img_embeddings)
        labels = self.classifier(concepts)
        return labels, concepts