import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from math import floor
from torchvision.models import vgg19, densenet201, resnet18
from typing import Dict, Tuple
from torchvision import transforms
from torchvision.utils import save_image, make_grid
from transformers import ViTModel
from tqdm.auto import tqdm
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
    def __init__(self):
        super(ViT, self).__init__()
        # Load the pre-trained ViT-B/16 model from HuggingFace
        self.vit = ViTModel.from_pretrained('google/vit-base-patch16-224-in21k')

    def forward(self, x):
        outputs = self.vit(pixel_values=x).last_hidden_state[:, 0]  # Use the [CLS] token representation
        return outputs
    
    
class ResNet(nn.Module):
    def __init__(self):
        super(ResNet, self).__init__()
        # Load the pre-trained ResNet-18 model
        self.resnet = resnet18(pretrained=True)
        # Remove the final fully connected layer (classifier)
        self.resnet = nn.Sequential(*list(self.resnet.children())[:-1])  # Remove the FC layer

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
            expand_recon_bottleneck = False
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
        
        if self.bound > 0:
            self.softsign = MaxNormActivation(bound**0.5) #nn.Tanh() #nn.Softsign()
        
        if backbone=='vgg':
            self.backbone = VGG().to(device)
            self.in_features = 4096   
        elif backbone=='densenet':
            self.backbone = DenseNet().to(device)
            self.in_features = 1920  
        elif backbone=='vit':
            self.backbone = ViT().to(device)
            self.in_features = 768 
        elif backbone=='resnet':
            self.backbone = ResNet().to(device)
            self.in_features = 512
        else:
            raise ValueError('Backbone not implemented!')

        freeze_params(self.backbone.parameters())
        
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
    ):
        super().__init__()
        self.device = device
        
        if backbone=='vgg':
            self.backbone = VGG().to(device)
            self.in_features = 4096   
        elif backbone=='densenet':
            self.backbone = DenseNet().to(device)
            self.in_features = 1920  
        elif backbone=='vit':
            self.backbone = ViT().to(device)
            self.in_features = 768 
        elif backbone=='resnet':
            self.backbone = ResNet().to(device)
            self.in_features = 512
        else:
            raise ValueError('Backbone not implemented!')

        freeze_params(self.backbone.parameters())

        self.classifier = torch.nn.Sequential(
                nn.Linear(self.in_features, self.in_features),
                nn.ReLU(),
                nn.Linear(self.in_features, n_labels),
            )

    def forward(self, x):
        img_embeddings = self.backbone(x)
        labels = self.classifier(img_embeddings)
        return labels


'''
### DDPM implementation ###


class SelfAttention(nn.Module):
    def __init__(self, channels, size):
        super(SelfAttention, self).__init__()
        self.channels = channels
        self.size = size
        self.mha = nn.MultiheadAttention(channels, 4, batch_first=True)
        self.ln = nn.LayerNorm([channels])
        self.ff_self = nn.Sequential(
            nn.LayerNorm([channels]),
            nn.Linear(channels, channels),
            nn.GELU(),
            nn.Linear(channels, channels),
        )

    def forward(self, x):
        x = x.view(-1, self.channels, self.size * self.size).swapaxes(1, 2)
        x_ln = self.ln(x)
        attention_value, _ = self.mha(x_ln, x_ln, x_ln)
        attention_value = attention_value + x
        attention_value = self.ff_self(attention_value) + attention_value
        return attention_value.swapaxes(2, 1).view(-1, self.channels, self.size, self.size)


class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels, mid_channels=None, residual=False):
        super().__init__()
        self.residual = residual
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(1, mid_channels),
            nn.GELU(),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(1, out_channels),
        )

    def forward(self, x):
        if self.residual:
            return F.gelu(x + self.double_conv(x))
        else:
            return self.double_conv(x)


class Down(nn.Module):
    def __init__(self, in_channels, out_channels, emb_dim=256):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, in_channels, residual=True),
            DoubleConv(in_channels, out_channels),
        )

        self.emb_layer = nn.Sequential(
            nn.SiLU(),
            nn.Linear(
                emb_dim,
                out_channels
            ),
        )

    def forward(self, x, t):
        x = self.maxpool_conv(x)
        emb = self.emb_layer(t)[:, :, None, None].repeat(1, 1, x.shape[-2], x.shape[-1])
        return x + emb


class Up(nn.Module):
    def __init__(self, in_channels, out_channels, emb_dim=256):
        super().__init__()

        self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
        self.conv = nn.Sequential(
            DoubleConv(in_channels, in_channels, residual=True),
            DoubleConv(in_channels, out_channels, in_channels // 2),
        )

        self.emb_layer = nn.Sequential(
            nn.SiLU(),
            nn.Linear(
                emb_dim,
                out_channels
            ),
        )

    def forward(self, x, skip_x, t):
        x = self.up(x)
        x = torch.cat([skip_x, x], dim=1)
        x = self.conv(x)
        emb = self.emb_layer(t)[:, :, None, None].repeat(1, 1, x.shape[-2], x.shape[-1])
        return x + emb


class UNet_conditional(nn.Module):
    def __init__(self, c_in=3, c_out=3, time_dim=256, num_classes=None, device="cuda"):
        super().__init__()
        self.device = device
        self.time_dim = time_dim
        self.inc = DoubleConv(c_in, 64)
        self.down1 = Down(64, 128)
        self.sa1 = SelfAttention(128, 32)
        self.down2 = Down(128, 256)
        self.sa2 = SelfAttention(256, 16)
        self.down3 = Down(256, 256)
        self.sa3 = SelfAttention(256, 8)

        self.bot1 = DoubleConv(256, 512)
        self.bot2 = DoubleConv(512, 512)
        self.bot3 = DoubleConv(512, 256)

        self.up1 = Up(512, 128)
        self.sa4 = SelfAttention(128, 16)
        self.up2 = Up(256, 64)
        self.sa5 = SelfAttention(64, 32)
        self.up3 = Up(128, 64)
        self.sa6 = SelfAttention(64, 64)
        self.outc = nn.Conv2d(64, c_out, kernel_size=1)
        
        #if num_classes is not None:
        #    self.label_emb = nn.Embedding(num_classes, time_dim)
        if num_classes is not None:
            self.label_emb = nn.Linear(num_classes, time_dim, bias=False)

    def pos_encoding(self, t, channels):
        inv_freq = 1.0 / (
            10000
            ** (torch.arange(0, channels, 2, device=self.device).float() / channels)
        )
        pos_enc_a = torch.sin(t.repeat(1, channels // 2) * inv_freq)
        pos_enc_b = torch.cos(t.repeat(1, channels // 2) * inv_freq)
        pos_enc = torch.cat([pos_enc_a, pos_enc_b], dim=-1)
        return pos_enc

    def forward(self, x, t, y):
        t = t.unsqueeze(-1).type(torch.float)
        t = self.pos_encoding(t, self.time_dim)

        if y is not None:
            t += self.label_emb(y)

        x1 = self.inc(x)
        x2 = self.down1(x1, t)
        x2 = self.sa1(x2)
        x3 = self.down2(x2, t)
        x3 = self.sa2(x3)
        x4 = self.down3(x3, t)
        x4 = self.sa3(x4)

        x4 = self.bot1(x4)
        x4 = self.bot2(x4)
        x4 = self.bot3(x4)

        x = self.up1(x4, x3, t)
        x = self.sa4(x)
        x = self.up2(x, x2, t)
        x = self.sa5(x)
        x = self.up3(x, x1, t)
        x = self.sa6(x)
        output = self.outc(x)
        return output


class Diffusion:
    def __init__(self, noise_steps=1000, beta_start=1e-4, beta_end=0.02, img_size=64, device="cuda"):
        self.noise_steps = noise_steps
        self.beta_start = beta_start
        self.beta_end = beta_end

        self.beta = self.prepare_noise_schedule().to(device)
        self.alpha = 1. - self.beta
        self.alpha_hat = torch.cumprod(self.alpha, dim=0)

        self.img_size = img_size
        self.device = device

    def prepare_noise_schedule(self):
        return torch.linspace(self.beta_start, self.beta_end, self.noise_steps)

    def noise_images(self, x, t):
        sqrt_alpha_hat = torch.sqrt(self.alpha_hat[t])[:, None, None, None]
        sqrt_one_minus_alpha_hat = torch.sqrt(1 - self.alpha_hat[t])[:, None, None, None]
        noise = torch.randn_like(x)
        return sqrt_alpha_hat * x + sqrt_one_minus_alpha_hat * noise, noise 

    def sample_timesteps(self, n):
        return torch.randint(low=1, high=self.noise_steps, size=(n,))

    def sample(self, model, n, labels, c_emb, emb_conditioning, cfg_scale=3):
        model.eval()
        with torch.no_grad():
            x = torch.randn((n, 3, self.img_size, self.img_size)).to(self.device)
            for i in tqdm(reversed(range(1, self.noise_steps)), position=0, total=(self.noise_steps-1)):
                t = (torch.ones(n) * i).long().to(self.device)
                predicted_noise = model(x, t, labels)
                if cfg_scale > 0:
                    uncond_predicted_noise = model(x, t, torch.zeros_like(labels))
                    predicted_noise = torch.lerp(uncond_predicted_noise, predicted_noise, cfg_scale)
                alpha = self.alpha[t][:, None, None, None]
                alpha_hat = self.alpha_hat[t][:, None, None, None]
                beta = self.beta[t][:, None, None, None]
                if i > 1:
                    noise = torch.randn_like(x)
                else:
                    noise = torch.zeros_like(x)
                x = 1 / torch.sqrt(alpha) * (x - ((1 - alpha) / (torch.sqrt(1 - alpha_hat))) * predicted_noise) + torch.sqrt(beta) * noise
        return x
    
    
### DDPM implementation (start) ###        
    
def ddpm_schedules(beta1, beta2, T):

    assert beta1 < beta2 < 1.0, "beta1 and beta2 must be in (0, 1)"

    beta_t = (beta2 - beta1) * torch.arange(0, T + 1, dtype=torch.float32) / T + beta1
    sqrt_beta_t = torch.sqrt(beta_t)
    alpha_t = 1 - beta_t
    log_alpha_t = torch.log(alpha_t)
    alphabar_t = torch.cumsum(log_alpha_t, dim=0).exp()

    sqrtab = torch.sqrt(alphabar_t)
    oneover_sqrta = 1 / torch.sqrt(alpha_t)

    sqrtmab = torch.sqrt(1 - alphabar_t)
    mab_over_sqrtmab_inv = (1 - alpha_t) / sqrtmab

    return {
        "alpha_t": alpha_t,  # \alpha_t
        "oneover_sqrta": oneover_sqrta,  # 1/\sqrt{\alpha_t}
        "sqrt_beta_t": sqrt_beta_t,  # \sqrt{\beta_t}
        "alphabar_t": alphabar_t,  # \bar{\alpha_t}
        "sqrtab": sqrtab,  # \sqrt{\bar{\alpha_t}}
        "sqrtmab": sqrtmab,  # \sqrt{1-\bar{\alpha_t}}
        "mab_over_sqrtmab": mab_over_sqrtmab_inv,  # (1-\alpha_t)/\sqrt{1-\bar{\alpha_t}}
    }


class DDPM(nn.Module):
    def __init__(self, nn_model, betas, n_T, device, size, channels, compose, n_concepts, drop_prob=0.1):
        super(DDPM, self).__init__()
        self.nn_model = nn_model.to(device)

        # register_buffer allows accessing dictionary produced by ddpm_schedules
        # e.g. can access self.sqrtab later
        for k, v in ddpm_schedules(betas[0], betas[1], n_T).items():
            self.register_buffer(k, v)
            
        self.n_T = n_T
        self.device = device
        self.drop_prob = drop_prob
        self.size = size
        self.channels = channels
        self.compose = compose
        self.n_concepts = n_concepts
    
    def forward(self, x, conditioning=None, emb_conditioning=False):
        _ts = torch.randint(1, self.n_T+1, (x.shape[0],)).to(self.device)  # t ~ Uniform(0, n_T)
        noise = torch.randn_like(x)  # eps ~ N(0, 1)
        bsz = x.shape[0]
        x_t = self.sqrtab[_ts, None, None, None] * x + self.sqrtmab[_ts, None, None, None] * noise  # noisy image
        if emb_conditioning:
            predicted_noise = self.nn_model(x_t, _ts, conditioning).sample  #  / self.n_T
        else:
            predicted_noise = self.nn_model(x_t, _ts, conditioning).sample 
        return predicted_noise, noise

    
    def sample(self, n_sample, concept, c_emb, size, emb_conditioning, device, guide_w = 0.0):
        n_concepts = c_emb.shape[0]
        
        x_i = torch.randn(n_sample, *size).to(device)  # x_T ~ N(0, 1), sample initial noise
        c_i = torch.zeros(n_sample, n_concepts).to(device) # context for us just cycles throught the mnist labels
        c_i[:,concept] = torch.ones(c_i.shape[0]).to(device) # specify which is the active concept    

        context_mask = torch.zeros(c_i.shape[0]).to(device)
        # double the batch
        c_i = c_i.repeat(2, 1)
        context_mask = context_mask.repeat(2)
        context_mask[n_sample:] = 1. # makes second half of batch context free
        x_i_store = [] # keep track of generated steps in case want to plot something 
                
        for i in tqdm(range(self.n_T, 0, -1), total=self.n_T):
            #print(f'sampling timestep {i}',end='\r')
            t_is = torch.tensor([i]).to(device)  #  / self.n_T
            t_is = t_is.repeat(n_sample,1,1,1)
            # double batch
            x_i = x_i.repeat(2,1,1,1)
            t_is = t_is.repeat(2,1,1,1)
            z = torch.randn(n_sample, *size).to(device) if i > 1 else 0
            # split predictions and compute weighting

            t_is = t_is.squeeze()
            mask = drop_context(c_i, context_mask)

            if emb_conditioning:
                masked_c_emb = c_emb * mask[:,:,None]
                masked_c_emb = masked_c_emb.flatten(start_dim=1).unsqueeze(1)
                eps = self.nn_model(x_i, t_is, masked_c_emb).sample
            else:  
                eps = self.nn_model(x_i, t_is, mask.unsqueeze(1)).sample  
                
            eps1 = eps[:n_sample]
            eps2 = eps[n_sample:]
            eps = (1+guide_w)*eps1 - guide_w*eps2     
            x_i = x_i[:n_sample]
            x_i = (
                self.oneover_sqrta[i] * (x_i - eps * self.mab_over_sqrtmab[i])
                + self.sqrt_beta_t[i] * z
            )
            if i%20==0 or i==self.n_T or i<8:
                x_i_store.append(x_i.detach().cpu().numpy())
        x_i_store = np.array(x_i_store)
        return x_i, x_i_store

    
### DDPM implementation (end) ###        
'''