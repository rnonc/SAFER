"""
Rodolphe Nonclercq
rnonclercq@gmail.com
"""

import torch
import torch.nn as nn
import numpy as np
from torchvision.models import resnet50, ResNet50_Weights,resnet18, ResNet18_Weights

# Inspired by https://github.com/evelinehong/slot-attention-pytorch

def build_grid(resolution):
    ranges = [np.linspace(0., 1., num=res) for res in resolution]
    grid = np.meshgrid(*ranges, sparse=False, indexing="ij")
    grid = np.stack(grid, axis=-1)
    grid = np.reshape(grid, [resolution[0], resolution[1], -1])
    grid = np.expand_dims(grid, axis=0)
    grid = grid.astype(np.float32)
    return torch.from_numpy(np.concatenate([grid, 1.0 - grid], axis=-1))

"""Adds soft positional embedding with learnable projection."""
class SoftPositionEmbed(nn.Module):
    def __init__(self, hidden_size, resolution):
        """Builds the soft position embedding layer.
        Args:
        hidden_size: Size of input feature dimension.
        resolution: Tuple of integers specifying width and height of grid.
        """
        super().__init__()
        self.embedding = nn.Linear(4, hidden_size, bias=True)
        self.grid = build_grid(resolution)

    def forward(self, inputs):
        grid = self.embedding(self.grid.to(inputs.device))
        return inputs + grid

class Encoder(nn.Module):
    def __init__(self, resolution, hid_dim):
        super().__init__()
        self.conv1 = nn.Conv2d(3, hid_dim, 5, padding = 2)
        self.conv2 = nn.Conv2d(hid_dim, hid_dim, 5, padding = 2)
        self.conv3 = nn.Conv2d(hid_dim, hid_dim, 5, padding = 2)
        self.conv4 = nn.Conv2d(hid_dim, hid_dim, 5, padding = 2)
        self.relu = nn.ReLU()
        self.positional_encoder = SoftPositionEmbed(hid_dim, resolution)

        self.fc1 = nn.Linear(hid_dim, hid_dim)
        self.fc2 = nn.Linear(hid_dim, hid_dim)
        self.norm_input  = nn.LayerNorm(hid_dim)

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.relu(x)
        x = self.conv3(x)
        x = self.relu(x)
        x = self.conv4(x)
        x = self.relu(x)
        x = x.permute(0,2,3,1)
        x = self.positional_encoder(x)
        x = torch.flatten(x, 1, 2)
        x = nn.LayerNorm(x.shape[1:],device=x.device)(x)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)  # Feedforward network on set.
        return x

class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride = 1, downsample = None):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Sequential(
                        nn.Conv2d(in_channels, out_channels, kernel_size = 3, stride = stride, padding = 1),
                        nn.BatchNorm2d(out_channels),
                        nn.ReLU())
        self.conv2 = nn.Sequential(
                        nn.Conv2d(out_channels, out_channels, kernel_size = 3, stride = 1, padding = 1),
                        nn.BatchNorm2d(out_channels))
        self.downsample = downsample
        self.relu = nn.ReLU()
        self.out_channels = out_channels
        
    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.conv2(out)
        if self.downsample:
            residual = self.downsample(x)
        out += residual
        out = self.relu(out)
        return out

class Residual_model(nn.Module):
    def __init__(self,resolution, hid_dim,nb_block=8,num_layer_downsampling=None):
        super().__init__()
        if num_layer_downsampling is None:
            self.conv1 = nn.Conv2d(3, hid_dim, kernel_size = 3, stride = 1, padding = 1)
            Blocks = [ResidualBlock(hid_dim,hid_dim) for i in range(nb_block)]
        else:
            self.conv1 = nn.Conv2d(3, hid_dim//2, kernel_size = 3, stride = 1, padding = 1)
            Blocks = [ResidualBlock(hid_dim//2,hid_dim//2) for i in range(num_layer_downsampling)] +[nn.Conv2d(hid_dim//2,hid_dim,1,stride=2)] + [ResidualBlock(hid_dim,hid_dim) for i in range(num_layer_downsampling,nb_block)]

        self.blocks = nn.Sequential(*Blocks)

        if num_layer_downsampling is None:
            self.positional_encoder = SoftPositionEmbed(hid_dim, resolution)
        else:
            self.positional_encoder = SoftPositionEmbed(hid_dim, (resolution[0]//2,resolution[1]//2))
        self.relu = nn.ReLU()
        self.fc1 = nn.Linear(hid_dim, hid_dim)
        self.fc2 = nn.Linear(hid_dim, hid_dim)
        self.norm_input  = nn.LayerNorm(hid_dim)

    def forward(self,x):
        x = self.conv1(x)
        x = self.blocks(x)
        x = self.relu(x)
        x = x.permute(0,2,3,1)
        x = self.positional_encoder(x)
        x = torch.flatten(x, 1, 2)
        #x = nn.LayerNorm(x.shape[1:],device=x.device)(x)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)

        return x

class ResNet_encoder(nn.Module):
    def __init__(self,resolution, hid_dim,depth=[True,True],final_R=False):
        super().__init__()
        
        resnet = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)
        layer_dims = [256,512,1024,2048]
        
        resnet = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
        layer_dims = [64,128,256,512]
        self.first_conc = nn.Sequential(*list(resnet.children())[:4])
        self.layer_blocks = nn.ModuleList()
        self.change_dim = nn.ModuleList()
        self.positional_encoder = nn.ModuleList()
        self.fc_output = nn.ModuleList()
        self.depth = depth
        for i in range(len(depth)):
            self.layer_blocks.append(list(resnet.children())[4+i])
            self.change_dim.append(nn.Conv2d(layer_dims[i],hid_dim,1))
            self.positional_encoder.append(SoftPositionEmbed(hid_dim, (resolution[0]//pow(2,i+2),resolution[1]//pow(2,i+2))))
            self.fc_output.append(nn.Sequential(*[nn.Linear(hid_dim, hid_dim),nn.ReLU(),nn.Linear(hid_dim, hid_dim)]))
        if final_R:
            self.final_layers = nn.Sequential(*list(resnet.children())[4+len(depth):-1])
            self.fc_out = list(resnet.children())[-1]
        else:
            self.final_layers = None

    def forward(self,x,depth=None):
        if depth is None:
            depth = self.depth
        x = self.first_conc(x)
        y= []
        for i in range(len(depth)):
            x = self.layer_blocks[i](x)
            if depth[i]:
                rel = self.change_dim[i](x)
                rel = self.positional_encoder[i](rel.permute(0,2,3,1))
                y.append(self.fc_output[i](torch.flatten(rel,1,2)))
        
        if not self.final_layers is None:
            R = self.final_layers(x).squeeze(-1).squeeze(-1)
            R = self.fc_out(R)
            return y,R
        else:
            return y

class Decoder(nn.Module):
    def __init__(self, hid_dim, resolution):
        super().__init__()
        self.conv1 = nn.ConvTranspose2d(hid_dim, hid_dim, 5, stride=(2, 2), padding=2, output_padding=1)
        self.conv2 = nn.ConvTranspose2d(hid_dim, hid_dim, 5, stride=(2, 2), padding=2, output_padding=1)
        self.conv3 = nn.ConvTranspose2d(hid_dim, 64, 5, stride=(2, 2), padding=2, output_padding=1)
        self.conv4 = nn.ConvTranspose2d(64, 64, 5, stride=(2, 2), padding=2, output_padding=1)
        if resolution[0] > 128:
            self.conv4prime = nn.ConvTranspose2d(64, 64, 5, stride=(2, 2), padding=2, output_padding=1)
        else:
            self.conv4prime =  None
        self.conv5 = nn.ConvTranspose2d(64, 32, 5, stride=(1, 1), padding=2)
        self.conv6 = nn.ConvTranspose2d(32, 4, 3, stride=(1, 1), padding=1)
        self.relu = nn.ReLU()
        self.decoder_initial_size = (7, 7)
        self.final_offset = (0,0)

        """
        if resolution[0] < 128:
            self.final_offset = ((128%resolution[0])//2,(128%resolution[1])//2)
        else:
            self.final_offset = ((256%resolution[0])//2,(256%resolution[1])//2)
        """
        self.decoder_pos = SoftPositionEmbed(hid_dim, self.decoder_initial_size)
        self.resolution = resolution

    def forward(self, slots):

        slots_ex = slots.reshape((-1, slots.shape[-1])).unsqueeze(1).unsqueeze(2)
        slots_ex = slots_ex.repeat((1, self.decoder_initial_size[0], self.decoder_initial_size[1], 1))

        x = self.decoder_pos(slots_ex)
        x = x.permute(0,3,1,2)
        x = self.conv1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.relu(x)
        x = self.conv3(x)
        x = self.relu(x)
        x = self.conv4(x)
        x = self.relu(x)
        if not self.conv4prime is None:
            x = self.conv4prime(x)
            x = x.relu()
        x = self.conv5(x)
        x = self.relu(x)
        x = self.conv6(x)
        x = x[:,:,self.final_offset[0]:self.final_offset[0]+self.resolution[0], self.final_offset[1]:self.final_offset[1]+self.resolution[1]]

        recons, masks = x.reshape(slots.shape[0], -1, x.shape[1], x.shape[2], x.shape[3]).split([3,1], dim=-3)
        return recons,masks

class Initializer(nn.Module):
    def __init__(self,nb_slot,dim_slot,mono=False):
        super().__init__()
        self.nb_slot = nb_slot
        self.dim_slot = dim_slot
        if mono:
            self.slots_mu = nn.Parameter(torch.randn(1, 1, dim_slot))
            self.slots_sigma = nn.Parameter(torch.rand(1, 1, dim_slot))
        else:
            self.slots_mu = nn.Parameter(torch.randn(1, nb_slot, dim_slot))
            self.slots_sigma = nn.Parameter(torch.rand(1, nb_slot, dim_slot))
        
    def forward(self,batch,nb_slots=None,return_slots=True):
        if nb_slots is None:
            nb_slots = self.nb_slot
        mu = self.slots_mu.expand(batch, nb_slots, self.dim_slot)
        log_sigma = self.slots_sigma.expand(batch, nb_slots, self.dim_slot)

        if return_slots:
            eps  = torch.randn_like(log_sigma).to(mu.device)
            std = torch.exp(0.5*log_sigma)
            slots_init = mu + std*eps

            return slots_init,mu,log_sigma
        else:
            return mu,log_sigma

class Classifier(nn.Module):
    def __init__(self,dim_slot):
        super().__init__()
        self.norm =  nn.LayerNorm(dim_slot)
        self.to_k = nn.Linear(dim_slot,dim_slot)
        self.to_v = nn.Linear(dim_slot,dim_slot)
        self.to_q = nn.Linear(dim_slot,dim_slot)
        self.fc_1 = nn.Linear(dim_slot,1)
        self.scale = dim_slot ** -0.5

    def forward(self,x):
        x = self.norm(x)
        q,k,v = self.to_q(x),self.to_k(x),self.to_v(x)
        dots = torch.einsum('bid,bjd->bij', q, k) * self.scale
        attn = dots.softmax(dim=2)
        x = torch.einsum('bij,bjd->bid', attn, v)
        result = self.fc_1(x.sum(dim=1)).sigmoid()
        return result.squeeze(-1)
    
class SlotAttention(nn.Module):
    def __init__(self, num_iterations, hid_dim):
        """Builds the Slot Attention-based auto-encoder.
        Args:
        resolution: Tuple of integers specifying width and height of input image.
        num_slots: Number of slots in Slot Attention.
        num_iterations: Number of iterations in Slot Attention.
        """
        super().__init__()
        self.hid_dim = hid_dim
        self.num_iterations = num_iterations

        dim = hid_dim
        hidden_dim = hid_dim

        self.eps = 1e-8
        self.scale = dim ** -0.5


        self.to_q = nn.Linear(dim, dim)
        self.to_k = nn.Linear(dim, dim)
        self.to_v = nn.Linear(dim, dim)
        
        self.gru = nn.GRUCell(dim, dim)

        hidden_dim = max(dim, hidden_dim)

        self.fc1_slot = nn.Linear(dim, hidden_dim)
        self.fc2_slot = nn.Linear(hidden_dim, dim)

        self.norm_slots  = nn.LayerNorm(dim)
        self.norm_pre_ff = nn.LayerNorm(dim)
        self.norm_input = nn.LayerNorm(dim)

        self.relu = nn.ReLU()
    
    def img_k_v(self,x):    
        inputs = self.norm_input(x)  
        k, v = self.to_k(inputs), self.to_v(inputs)
        return k,v
    
    def slots_attention(self,slots,k,v,act_fc='softmax',iter=None,pond=None):

        b,n_s,d = slots.shape
        if iter is None:
            iter = self.num_iterations
        
        for i in range(iter):

            slots_prev = slots

            slots = self.norm_slots(slots)
            q = self.to_q(slots)

            # Q,K,V attention
            dots = torch.einsum('bid,bjd->bij', q, k) * self.scale
            if act_fc == 'softmax':
                attn = dots.softmax(dim=1) + self.eps
                
            elif act_fc == 'sigmoid':
                attn = dots.sigmoid() + self.eps
            if not pond is None:
                attn = attn*pond
            
            attn = attn/ attn.sum(dim=-1, keepdim=True)
            updates = torch.einsum('bjd,bij->bid', v, attn)
            
            # new slots
            slots = self.gru(
                updates.reshape(-1, d),
                slots_prev.reshape(-1, d)
            )

            slots = slots.reshape(b, -1, d)
            slots = slots + self.fc2_slot(self.relu(self.fc1_slot(self.norm_pre_ff(slots))))

        return slots
    
    def attention_mask(self,slots,k,reshape=True):
        slots = self.norm_slots(slots)
        q = self.to_q(slots)
        dots = torch.einsum('bid,bjd->bij', q, k) * self.scale
        if reshape:
            masks = dots.reshape(dots.shape[0],dots.shape[1],1,112,112)
        else:
            masks = dots
        return masks
    
    def forward(self, inputs,init_slots,act_fc = 'softmax'):
        # `image` has shape: [batch_size, num_channels, width, height].
        # Convolutional encoder with position embedding.
        k,v = self.img_k_v(inputs)
        
        # Slot Attention module.
        slots = self.slots_attention(init_slots,k,v,act_fc)
        # `slots` has shape: [batch_size, num_slots, slot_size].

        return slots

class MultiDepthSlotAttention(nn.Module):
    def __init__(self, num_iterations, hid_dim,input_stack_dim = None):
        """Builds the Slot Attention-based auto-encoder.
        Args:
        resolution: Tuple of integers specifying width and height of input image.
        num_slots: Number of slots in Slot Attention.
        num_iterations: Number of iterations in Slot Attention.
        """
        super().__init__()
        if input_stack_dim is None:
            self.input_stack_dim = [hid_dim]
            self.depth = 1
        else:
            self.input_stack_dim = input_stack_dim
            self.depth = len(input_stack_dim)
        
        self.dim = hid_dim
        self.num_iterations = num_iterations

        self.eps = 1e-8
        self.scale = self.dim ** -0.5

        self.to_q = nn.ModuleList()
        self.to_k = nn.ModuleList()
        self.to_v = nn.ModuleList()
        self.norm_input = nn.ModuleList()
        self.norm_slots = nn.ModuleList()

        for i in range(self.depth):
            self.to_q.append(nn.Linear(self.dim, self.dim))
            self.to_k.append(nn.Linear(self.input_stack_dim[i], self.dim))
            self.to_v.append(nn.Linear(self.input_stack_dim[i], self.dim))
            self.norm_input.append(nn.LayerNorm(self.input_stack_dim[i]))
            self.norm_slots.append(nn.LayerNorm(self.dim))
        
        self.gru = nn.GRUCell(self.dim, self.dim)

        self.fc1_slot = nn.Linear(self.dim, self.dim)
        self.fc2_slot = nn.Linear(self.dim, self.dim)

        self.norm_pre_ff = nn.LayerNorm(self.dim)

        self.relu = nn.ReLU()
    
    def img_k_v(self,x):
        if type(x) == torch.Tensor:
            x = [x]
        k = []
        v = []
        for i in range(len(x)):
            input = self.norm_input[i](x[i])  
            k.append(self.to_k[i](input))
            v.append(self.to_v[i](input))
        return k,v
    
    def slots_attention(self,slots,k,v,iter=None):

        b,n_s,d = slots.shape
        if iter is None:
            iter = self.num_iterations
        
        for i in range(iter):

            slots_prev = slots
            
            inputs = slots
            update_result = []
            for depth in range(len(k)):
                inputs = self.norm_slots[-1-depth](inputs)
                q = self.to_q[-1-depth](inputs)
                # Q,K,V attention
                dots = torch.einsum('bid,bjd->bij', q, k[-1-depth]) * self.scale
                attn = dots.softmax(dim=1) + self.eps 
                attn = attn/ attn.sum(dim=-1, keepdim=True)
                updates = torch.einsum('bjd,bij->bid', v[-1-depth], attn)
                update_result.append(updates)
                inputs = sum(update_result)
                
            final_update = sum(update_result)
            # new slots
            slots = self.gru(
                final_update.reshape(-1, d),
                slots_prev.reshape(-1, d)
            )

            slots = slots.reshape(b, -1, d)
            slots = slots + self.fc2_slot(self.relu(self.fc1_slot(self.norm_pre_ff(slots))))

        return slots
    
    def attention_mask(self,slots,k,reshape=True):
        # need to modify it
        slots = self.norm_slots(slots)
        q = self.to_q(slots)
        dots = torch.einsum('bid,bjd->bij', q, k) * self.scale
        if reshape:
            masks = dots.reshape(dots.shape[0],dots.shape[1],1,112,112)
        else:
            masks = dots
        return masks
    
    def forward(self, inputs,init_slots):
        # `image` has shape: [batch_size, num_channels, width, height].
        # Convolutional encoder with position embedding.
        k,v = self.img_k_v(inputs)
        
        # Slot Attention module.
        slots = self.slots_attention(init_slots,k,v)
        # `slots` has shape: [batch_size, num_slots, slot_size].

        return slots

class Attention(nn.Module):
    def __init__(self,dim,act=None):
        super().__init__()
        self.to_q = nn.Linear(dim, dim)
        self.to_k = nn.Linear(dim, dim)
        self.to_v = nn.Linear(dim, dim)
        self.norm_target  = nn.LayerNorm(dim)
        self.norm_source_k = nn.LayerNorm(dim)
        self.norm_source_v = nn.LayerNorm(dim)
        self.scale = dim ** -0.5
        if act is None:
            self.act = nn.Softmax(dim=2)
    def forward(self,target,source_k,source_v=None):
        if source_v is None:
            source_v = source_k
        target = self.norm_target(target)
        source_k,source_v = self.norm_source_k(source_k),self.norm_source_v(source_v)
        q,k,v = self.to_q(target),self.to_k(source_k),self.to_v(source_v)
        dots = torch.einsum('bid,bjd->bij', q, k) * self.scale
        attn =self.act(dots)
        x = torch.einsum('bij,bjd->bid', attn, v)
        return x

class SlotAttentionAutoEncoder(nn.Module):
    def __init__(self,resolution, num_iterations, hid_dim,device_ids=None,depth=[True,True],final_R=False):
        super().__init__()
        nb_depth = sum(depth)
        self.final_R= final_R
        #self.encoder = Residual_model(resolution,hid_dim,nb_block=nb_block,num_layer_downsampling=downsampling_layer)
        self.encoder = ResNet_encoder(resolution,hid_dim,depth=depth,final_R=final_R)
        self.decoder = Decoder(hid_dim,resolution)
        self.SlotAttention = MultiDepthSlotAttention(num_iterations,hid_dim,input_stack_dim=[hid_dim]*nb_depth)
        if not device_ids is None:
            self.decoder = nn.DataParallel(self.decoder,device_ids)

    def forward(self,image,slots_init):
        if self.final_R:
            inputs,R = self.encoder(image)
        else:
            inputs = self.encoder(image)
        slots = self.SlotAttention(inputs,slots_init)
        recons,masks = self.decoder(slots)

        if self.final_R:
            return recons,masks,slots,R
        else:
            return recons,masks,slots

class SlotAttentionVideoAutoEncoder(SlotAttentionAutoEncoder):
    def __init__(self, resolution, num_iterations, hid_dim,device_ids=None,depth=[True,True],final_R=False):
        super().__init__(resolution, num_iterations, hid_dim,device_ids,depth,final_R)
        self.final_R = final_R
        self.predictor = nn.Transformer(hid_dim,nhead=4,dim_feedforward=hid_dim*2,batch_first=True)

    def forward(self,image,init_slots):
        if len(image.shape) == 4 :
            if self.final_R:
                x,R = self.encoder(image)
            else:
                x = self.encoder(image)

            # Slot Attention module.
            slots = self.SlotAttention(x,init_slots)

            recons,masks = self.decoder(slots.detach())


            if self.final_R:
                return recons,masks,slots,R
            else:
                return recons,masks,slots
        """
        elif len(image.shape) == 5 :
            b,f,c,h,w = image.shape

            x = self.encoder(image.reshape(-1,c,h,w))

            k,v = self.SlotAttention.img_k_v(x)
            k,v = k.reshape(b,f,k.shape[1],k.shape[2]),v.reshape(b,f,v.shape[1],v.shape[2])

            list_slots = []
            for i in range(f):
                list_slots.append(self.SlotAttention.slots_attention(init_slots,k[:,i],v[:,i]))
                init_slots = list_slots[-1] + self.predictor(list_slots[-1],list_slots[-1])
            
            slots = torch.stack(list_slots,dim=1)

            recons,masks = self.decoder(slots.reshape(-1,slots.shape[-2],slots.shape[-1]))
            recons = recons.reshape(b,f,recons.shape[-4],recons.shape[-3],recons.shape[-2],recons.shape[-1])
            masks = masks.reshape(b,f,masks.shape[-4],masks.shape[-3],masks.shape[-2],masks.shape[-1])

            return recons,masks,slots
        """

