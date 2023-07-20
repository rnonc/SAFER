import torch
import torch.nn as nn
import numpy as np

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
        return x

class Decoder(nn.Module):
    def __init__(self, hid_dim, resolution):
        super().__init__()
        self.conv1 = nn.ConvTranspose2d(hid_dim, hid_dim, 5, stride=(2, 2), padding=2, output_padding=1)
        self.conv2 = nn.ConvTranspose2d(hid_dim, hid_dim, 5, stride=(2, 2), padding=2, output_padding=1)
        self.conv3 = nn.ConvTranspose2d(hid_dim, hid_dim, 5, stride=(2, 2), padding=2, output_padding=1)
        self.conv4 = nn.ConvTranspose2d(hid_dim, hid_dim, 5, stride=(2, 2), padding=2, output_padding=1)
        self.conv5 = nn.ConvTranspose2d(hid_dim, hid_dim, 5, stride=(1, 1), padding=2)
        self.conv6 = nn.ConvTranspose2d(hid_dim, 4, 3, stride=(1, 1), padding=1)
        self.relu = nn.ReLU()
        self.decoder_initial_size = (8, 8)
        self.decoder_pos = SoftPositionEmbed(hid_dim, self.decoder_initial_size)
        self.resolution = resolution

    def forward(self, x):
        x = self.decoder_pos(x)
        x = x.permute(0,3,1,2)
        x = self.conv1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.relu(x)
#         x = F.pad(x, (4,4,4,4)) # no longer needed
        x = self.conv3(x)
        x = self.relu(x)
        x = self.conv4(x)
        x = self.relu(x)
        x = self.conv5(x)
        x = self.relu(x)
        x = self.conv6(x)
        x = x[:,:,:self.resolution[0], :self.resolution[1]]
        x = x
        return x

class Initializer(nn.Module):
    def __init__(self,nb_slot,dim_slot):
        super().__init__()
        self.nb_slot = nb_slot
        self.dim_slot = dim_slot
        self.slots_mu = nn.Parameter(torch.randn(1, 1, dim_slot))
        self.slots_sigma = nn.Parameter(torch.rand(1, 1, dim_slot))

    def forward(self,x):
        mu = self.slots_mu.expand(x, self.nb_slot, self.dim_slot)
        log_sigma = self.slots_sigma.expand(x, self.nb_slot, self.dim_slot)
        
        return mu,log_sigma

class SlotAttentionAutoEncoder(nn.Module):
    def __init__(self, resolution, num_slots, num_iterations, hid_dim):
        """Builds the Slot Attention-based auto-encoder.
        Args:
        resolution: Tuple of integers specifying width and height of input image.
        num_slots: Number of slots in Slot Attention.
        num_iterations: Number of iterations in Slot Attention.
        """
        super().__init__()
        self.hid_dim = hid_dim
        self.resolution = resolution
        self.num_slots = num_slots
        self.num_iterations = num_iterations

        self.encoder_cnn = Encoder(self.resolution, self.hid_dim)
        self.decoder_cnn = Decoder(self.hid_dim, self.resolution)

        self.fc1 = nn.Linear(hid_dim, hid_dim)
        self.fc2 = nn.Linear(hid_dim, hid_dim)
        self.relu = nn.ReLU()

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

        self.norm_input  = nn.LayerNorm(dim)
        self.norm_slots  = nn.LayerNorm(dim)
        self.norm_pre_ff = nn.LayerNorm(dim)

        self.relu = nn.ReLU()
    
    def encoding(self,image):
        x = self.encoder_cnn(image)  # CNN Backbone.
        x = nn.LayerNorm(x.shape[1:],device=x.device)(x)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)  # Feedforward network on set.
        inputs = self.norm_input(x)        
        k, v = self.to_k(inputs), self.to_v(inputs)
        return k,v
    
    def decoding(self,slots):
        slots_ex = slots.reshape((-1, slots.shape[-1])).unsqueeze(1).unsqueeze(2)
        slots_ex = slots_ex.repeat((1, 8, 8, 1))
        x = self.decoder_cnn(slots_ex)

        recons, masks = x.reshape(slots.shape[0], -1, x.shape[1], x.shape[2], x.shape[3]).split([3,1], dim=-3)

        return recons, masks
    
    def slots_attention(self,slots,k,v):

        b,n_s,d = slots.shape

        for i in range(self.num_iterations):

            slots_prev = slots

            slots = self.norm_slots(slots)
            q = self.to_q(slots)

            # Q,K,V attention
            dots = torch.einsum('bid,bjd->bij', q, k) * self.scale

            attn = dots.softmax(dim=1) + self.eps

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
    
    def attention_mask(self,slots,k):
        slots = self.norm_slots(slots)
        q = self.to_q(slots)
        dots = torch.einsum('bid,bjd->bij', q, k) * self.scale

        masks = dots.reshape(dots.shape[0],dots.shape[1],1,112,112)
        return masks
    
    def forward(self, image,init_slots):
        # `image` has shape: [batch_size, num_channels, width, height].
        # Convolutional encoder with position embedding.
        k,v = self.encoding(image)
        
        # Slot Attention module.
        slots = self.slots_attention(init_slots,k,v)
        # `slots` has shape: [batch_size, num_slots, slot_size].

        # """Broadcast slot features to a 2D grid and collapse slot dimension.""".
        recons,masks = self.decoding(slots)
        # `recons` has shape: [batch_size, num_slots, num_channels , width, height].
        # `masks` has shape: [batch_size, num_slots, 1, width, height].

        return recons,masks,slots

class SlotVariationalAttentionAutoEncoder(SlotAttentionAutoEncoder):
    def __init__(self, resolution, num_slots, num_iterations, hid_dim):
        """Builds the Slot Attention-based auto-encoder.
        Args:
        resolution: Tuple of integers specifying width and height of input image.
        num_slots: Number of slots in Slot Attention.
        num_iterations: Number of iterations in Slot Attention.
        """
        super().__init__(resolution, num_slots, num_iterations, hid_dim)

        self.fc1_slot = nn.Linear(hid_dim, 2*hid_dim)
        self.fc2_slot = nn.Linear(2*hid_dim, 2*hid_dim)
        self.tanh = nn.Tanh()
    
    
    def slots_attention(self,slots,k,v,attn_fc='sigmoid'):

        b,n_s,d = slots.shape

        for i in range(self.num_iterations):
            slots_prev = slots

            slots = self.norm_slots(slots)
            q = self.to_q(slots)

            # Q,K,V attention
            dots = torch.einsum('bid,bjd->bij', q, k) * self.scale

            if attn_fc == 'exploitation':
                attn = dots.sigmoid() + self.eps
            elif  attn_fc == 'exploration':
                attn = dots.softmax(dim=1) + self.eps
            
            attn = attn/ attn.sum(dim=-1, keepdim=True)
            updates = torch.einsum('bjd,bij->bid', v, attn)
            
            # new slots
            slots = self.gru(
                updates.reshape(-1, d),
                slots_prev.reshape(-1, d)
            )

            slots = slots.reshape(b, -1, d)
            mu,log_sigma = self.fc2_slot(self.tanh(self.fc1_slot(self.norm_pre_ff(slots)))).split([slots.shape[-1],slots.shape[-1]],dim=-1)
            slots = self.reparameterize(mu,log_sigma)

        return mu,log_sigma

    
    def reparameterize(self,mu,log_sigma):
        eps  = torch.randn_like(log_sigma).to(mu.device)
        std = torch.exp(0.5*log_sigma)
        return  mu + std*eps
    
    def forward(self, image, init_slots,attn_fc='exploitation'):
        # Convolutional encoder with position embedding.
        k,v = self.encoding(image)
        
        # Slot Attention module.
        mu,log_sigma = self.slots_attention(init_slots,k,v,attn_fc=attn_fc)
        

        slots = self.reparameterize(mu,log_sigma)
        # `slots` has shape: [batch_size, num_slots, slot_size].

        # """Broadcast slot features to a 2D grid and collapse slot dimension.""".
        recons,masks = self.decoding(slots)
        # `recons` has shape: [batch_size, num_slots, num_channels , width, height].
        # `masks` has shape: [batch_size, num_slots, 1, width, height].

        return recons,masks,slots,mu,log_sigma

