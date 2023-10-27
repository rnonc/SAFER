#%%
import torch
import torch.nn as nn
import matplotlib.pyplot as plt


def optic_grid(image_size,AOV=80):
    plan_x = torch.zeros(image_size,image_size)
    plan_y = -(2*torch.arange(image_size)/(image_size-1)-1).unsqueeze(0).repeat(image_size,1)
    plan_z = (2*torch.arange(image_size)/(image_size-1)-1).unsqueeze(-1).repeat(1,image_size)
    plan = torch.stack([plan_x,plan_y,plan_z],dim=-1)
    alpha = 2*1.42/AOV
    lens = torch.Tensor([alpha,0,0])

    optic_grid = lens - plan
    optic_grid = optic_grid/torch.norm(optic_grid,dim=-1,keepdim=True)

    return optic_grid

def sixDOF(M):
    M1 = M[:,:,0]
    M1 = M1/torch.norm(M1,dim=-1,keepdim=True)
    M2 = M[:,:,1]-torch.sum(M[:,:,1]*M1,dim=-1,keepdim=True)*M1
    M2 = M2/torch.norm(M2,dim=-1,keepdim=True)
    M3 = torch.cross(M1,M2,dim=-1)
    R = torch.stack([M1,M2,M3],dim=2)
    return R

def plucker_relative_grid(spacial_grid,slots_space):
    slots_R,slots_p = slots_space.split([6,3],dim=-1)
    b,s,_ = slots_R.shape
    slots_R = slots_R.reshape(b,s,2,3)
    Rotation = sixDOF(slots_R) # rotation matrix Batch,Slot,new_coor,old_coor bsnd
    rel_grid = spacial_grid# Batch,1,Image,old_corr b1ld
    w,h,_ = rel_grid.shape
    l = torch.einsum('whd,bsnd->bswhn',rel_grid,Rotation)
    p = torch.einsum('bsd,bsnd->bsn',-slots_p,Rotation)
    m = torch.cross(p.unsqueeze(-2).unsqueeze(-2).repeat(1,1,w,h,1),l)
    plucker_rel_grid = torch.concat([l,m],dim=-1)
    return plucker_rel_grid,Rotation

class embedding_plucker_grid(nn.Module):
    def __init__(self,dim,resolution):
        super().__init__()
        self.fc1_grid = nn.Linear(6,dim)
        self.fc2_grid = nn.Linear(dim,dim)
        self.fc3_grid = nn.Linear(dim,dim)
        self.optic_grid = optic_grid(resolution)
    def forward(self,x,slots_space):
        plucker_rel_grid,_ = plucker_relative_grid(self.optic_grid.to(slots_space.device),slots_space)
        embed = self.fc1_grid(plucker_rel_grid)
        if len(x.shape) == 4:
            x = x.unsqueeze(1).repeat(1,slots_space.shape[1],1,1,1)
        x = x + embed.reshape(x.shape)
        x = self.fc2_grid(x).relu()
        x = self.fc3_grid(x)
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

class ResNet_encoder(nn.Module):
    def __init__(self, hid_dim,nb_block=8):
        super().__init__()
        self.conv1 = nn.Conv2d(3, hid_dim, kernel_size = 3, stride = 1, padding = 1)

        Blocks = [ResidualBlock(hid_dim,hid_dim) for i in range(nb_block)]
        self.blocks = nn.Sequential(*Blocks)
        
        self.fc = nn.Linear(hid_dim, hid_dim)

    def forward(self,x):
        x = self.conv1(x)
        x = self.blocks(x).relu()
        x = x.permute(0,2,3,1)
        x = self.fc(x)

        return x

class Decoder(nn.Module):
    def __init__(self, dim_slot, hid_dim, resolution,size_grid=8):
        super().__init__()
        self.conv1 = nn.ConvTranspose2d(hid_dim, hid_dim, 5, stride=(2, 2), padding=2, output_padding=1)
        self.conv2 = nn.ConvTranspose2d(hid_dim, hid_dim, 5, stride=(2, 2), padding=2, output_padding=1)
        self.conv3 = nn.ConvTranspose2d(hid_dim, hid_dim, 5, stride=(2, 2), padding=2, output_padding=1)
        self.conv4 = nn.ConvTranspose2d(hid_dim, hid_dim, 5, stride=(2, 2), padding=2, output_padding=1)
        self.conv5 = nn.ConvTranspose2d(hid_dim, hid_dim, 5, stride=(1, 1), padding=2)
        self.conv6 = nn.ConvTranspose2d(hid_dim, 4, 3, stride=(1, 1), padding=1)
        self.relu = nn.ReLU()
        self.emb_plucker = embedding_plucker_grid(dim_slot,resolution)
        self.init_grid_res = size_grid
        self.resolution = resolution

    def forward(self, slots,slots_space):

        slots_ex = slots.unsqueeze(-2).unsqueeze(-2)
        slots_ex = slots_ex.repeat((1,1, self.init_grid_res, self.init_grid_res, 1))

        x = self.emb_plucker (slots_ex,slots_space)
        x = x.reshape(-1,self.init_grid_res,self.init_grid_res,slots.shape[-1])
        x = x.permute(0,3,1,2)
        x = self.conv1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.relu(x)
        x = self.conv3(x)
        x = self.relu(x)
        x = self.conv4(x)
        x = self.relu(x)
        x = self.conv5(x)
        x = self.relu(x)
        x = self.conv6(x)
        x = x[:,:,:self.resolution[0], :self.resolution[1]]

        recons, masks = x.reshape(slots.shape[0], -1, x.shape[1], x.shape[2], x.shape[3]).split([3,1], dim=-3)
        return recons,masks

class PluckerSlotAttention(nn.Module):
    def __init__(self, num_iterations, hid_dim,resolution):
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

        self.emb_plucker = embedding_plucker_grid(dim,resolution)

        self.to_q = nn.Linear(dim, dim)
        self.to_k = nn.Linear(dim, dim)
        self.to_v = nn.Linear(dim, dim+9)

        self.gru = nn.GRUCell(dim, dim)
        self.gru_R = nn.GRUCell(3, 3)
        self.gru_P = nn.GRUCell(3, 3)


        self.fc1_slot = nn.Linear(dim, dim)
        self.fc2_slot = nn.Linear(dim, dim)

        self.norm_slots  = nn.LayerNorm(dim)
        self.norm_pre_ff = nn.LayerNorm(dim)
        self.norm_input = nn.LayerNorm(dim)

        self.relu = nn.ReLU()
    
    
    def slots_attention(self,x,slots,slots_space,iter=None):

        b,n_s,d = slots.shape
        if iter is None:
            iter = self.num_iterations
        
        for i in range(iter):
            inputs = self.emb_plucker(x,slots_space)
            inputs = inputs.reshape(b,-1,d)
            inputs = self.norm_input(inputs)
            k,v = self.to_k(inputs),self.to_v(inputs)

            slots_prev = slots
            slots_space_prev = slots_space

            slots = self.norm_slots(slots)
            q = self.to_q(slots)
            print(k.shape,q.shape)
            # Q,K,V attention
            dots = torch.einsum('bid,bijd->bij', q, k) * self.scale
            attn = dots.softmax(dim=1) + self.eps
            attn = attn/ attn.sum(dim=-1, keepdim=True)
            updates = torch.einsum('bijd,bij->bid', v, attn)
            updates_slots,updates_R,updates_P = updates.split([d,6,3],dim=-1)

            slots_R_prev,slots_P_prev = slots_space_prev.split([6,3],dim=-1)
            # new slots
            slots = self.gru(
                updates_slots.reshape(-1, d),
                slots_prev.reshape(-1, d)
            )
            slots_R = self.gru_R(
                updates_R.reshape(-1, 6),
                slots_R_prev.reshape(-1, 6)
            )
            slots_P = self.gru_P(
                updates_P.reshape(-1, 3),
                slots_P_prev.reshape(-1, 3)
            )
            slots_space = torch.concat([slots_R,slots_P],dim=-1).reshape(b,-1,9)

            slots = slots.reshape(b, -1, d)
            slots = slots + self.fc2_slot(self.relu(self.fc1_slot(self.norm_pre_ff(slots))))

        return slots,slots_space
    
    
    def forward(self, inputs,init_slots,init_slots_space):
        # `image` has shape: [batch_size, num_channels, width, height].
        # Convolutional encoder with position embedding.
        
        # Slot Attention module.
        slots = self.slots_attention(inputs,init_slots,init_slots_space)
        # `slots` has shape: [batch_size, num_slots, slot_size].

        return slots

class PluckerSAautoencoder(nn.Module):
    def __init__(self, num_iterations, hid_dim,resolution):
        super().__init__()
        self.encoder = ResNet_encoder(hid_dim,8)
        self.SA = PluckerSlotAttention(num_iterations, hid_dim,resolution)
        self.decoder = Decoder(hid_dim, hid_dim, resolution,8)
    def forward(self,x,slots,slots_space):
        x = self.encoder(x)
        slots,slots_space = self.SA(x,slots,slots_space)
        recons,masks = self.decoder(slots,slots_space)
        return recons, masks, slots, slots_space

if __name__ == '__main__':
    size = 112
    optic_grid = optic_grid(size)

    S_r = torch.Tensor([[1,0,0,    0,2,0],[1,0,0,    0,1,0]]).unsqueeze(0)
    S_p = torch.Tensor([[0,0.,0],[0,100,0]]).unsqueeze(0)
    slots_space = torch.concat([S_r,S_p],dim=-1)
    spacial_grid = optic_grid.reshape(1,-1,3)
    rel_grid,Rotation = plucker_relative_grid(spacial_grid,slots_space)
    rel_grid = rel_grid.reshape(1,2,size,size,6)
    print(Rotation)
    plt.imshow(rel_grid[0,0,:,:,:3].numpy(),vmin=0)


# %%
