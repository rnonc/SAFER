
import torch
import torch.nn as nn
from ..module_SA import Encoder, Decoder

class GraphConv(nn.Module):
    def __init__(self,input_node_dim, output_node_dim, input_edge_dim, output_edge_dim, edge_act = 'ReLu'):
        super().__init__()
        self.node_to_node = nn.Linear(input_node_dim,output_node_dim)
        self.node_hidden = nn.Linear(input_node_dim,output_node_dim)
        self.node_to_edge = nn.Linear(2*output_node_dim,output_edge_dim)
        self.edge_to_node = nn.Linear(output_edge_dim,output_node_dim)
        if edge_act == 'ReLu':
            self.act = nn.ReLU()
        if input_edge_dim > 0:
            self.edge_to_edge = nn.Linear(input_edge_dim,output_edge_dim)
        else:
            self.edge_to_edge = None
    def forward(self,N,E=0):
        b,n,d = N.shape
        # Update edge
        if not self.edge_to_edge is None:
            E_to_E = E.reshape(-1,E.shape[-1])
            E_to_E = self.edge_to_edge(E_to_E)
            E = E_to_E.reshape(E.shape[:-1]+E_to_E.shape[-1:])
        
        N_hidden = self.node_hidden(N.reshape(-1,d))
        N_hidden= N_hidden.reshape((b,n)+N_hidden[-1:])
        N_to_E = torch.concat([N_hidden.unsqueeze(2).expand((b,n,n,N_hidden.shape[-1])),N_hidden.unsqueeze(1).expand((b,n,n,N_hidden.shape[-1]))],dim=-1)
        N_to_E = N_to_E.reshape(-1,2*N_hidden.shape[-1])
        N_to_E = self.node_to_edge(N_to_E)
        N_to_E = N_to_E.reshape((b,n,n,)+N_to_E.shape[-1:])

        E = E + N_to_E

        #update node
        N = self.node_to_node(N.reshape(-1,d))
        N = N.reshape((b,n)+N.shape[-1:])

        E_to_N = E.reshape(-1,E.shape[-1])
        E_to_N = self.edge_to_node(E_to_N)
        E_to_N = E_to_N.reshape(E.shape[:-1] + E_to_N.shape[-1:])
        E_to_N = self.act(E_to_N)
        N =N + torch.sum(E_to_N*N_hidden.unsqueeze(2),dim=2)/n

        return N, E

class GraphAT(nn.Module):
    def __init__(self,input_node_dim, output_node_dim, input_edge_dim, output_edge_dim):
        super().__init__()
        self.node_to_node = nn.Linear(input_node_dim,output_node_dim)
        self.node_hidden = nn.Linear(input_node_dim,output_node_dim)
        self.node_to_edge = nn.Linear(2*output_node_dim,output_edge_dim)
        self.edge_to_edge = nn.Linear(input_edge_dim,output_edge_dim)
        self.edge_to_pond = nn.Linear(output_edge_dim,1)

        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim=2)
    def forward(self,N,E):
        b,n,d = N.shape
        
        # compute ponderation
        E_to_E = E.reshape(-1,E.shape[-1])
        E_to_E = self.edge_to_edge(E_to_E)
        E_to_E = E_to_E.reshape(E.shape[:-1]+E_to_E.shape[-1:])
        
        N_hidden = self.node_hidden(N.reshape(-1,d))
        N_hidden= N_hidden.reshape((b,n)+N_hidden.shape[-1:])
        N_to_E = torch.concat([N_hidden.unsqueeze(2).expand((b,n,n,N_hidden.shape[-1])),N_hidden.unsqueeze(1).expand((b,n,n,N_hidden.shape[-1]))],dim=-1)
        N_to_E = N_to_E.reshape(-1,2*N_hidden.shape[-1])
        N_to_E = self.node_to_edge(N_to_E)
        N_to_E = N_to_E.reshape((b,n,n,)+N_to_E.shape[-1:])

        E_pond = self.edge_to_pond((E_to_E + N_to_E).reshape(-1,N_to_E.shape[-1])).reshape(b,n,n,1)
        E_pond = self.softmax(E_pond)

        #update node
        N = self.relu(self.node_to_node(N.reshape(-1,d)))
        N = N.reshape((b,n)+N.shape[-1:])

        
        N =torch.sum(E_pond*N.unsqueeze(2),dim=2)

        return N, E_pond

class GraphPooling(nn.Module):
    def __init__(self,input_node_dim, output_node_dim):
        super().__init__()
        self.node_to_node = nn.Linear(input_node_dim,output_node_dim)
        self.node_to_pond = nn.Linear(input_node_dim,1)

        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim=1)
    def forward(self,N):
        b,n,d = N.shape
        
        N_pond = self.node_to_pond(N.reshape(-1,d))
        N_pond = N_pond.reshape(b,n,1)
        N_pond = self.softmax(N_pond)
        #update node
        N = self.relu(self.node_to_node(N.reshape(-1,d)))
        N = N.reshape((b,n)+N.shape[-1:])

        
        N =torch.sum(N_pond*N,dim=1)

        return N,N_pond

class GRU(nn.Module):
    def __init__(self,input_node_dim, memo_node_dim):
        super().__init__()
        self.node_reset = nn.Linear(input_node_dim+memo_node_dim, input_node_dim)
        self.node_update = nn.Linear(input_node_dim+memo_node_dim, input_node_dim)
        self.node_hidden = nn.Linear(input_node_dim+memo_node_dim, input_node_dim)

        self.sigmoid = nn.Sigmoid()
        self.tanh = nn.Tanh()

    def forward(self,N,E,N_new):
        b,n,d = N.shape
        N_conc = torch.concat([N,N_new],dim=-1)
        N_conc = N_conc.reshape(-1,N_conc.shape[-1])

        N_reset = self.node_reset(N_conc).reshape(b,n,-1)
        N_update = self.node_update(N_conc).reshape(b,n,-1)

        N_reset = self.sigmoid(N_reset)
        N_update = self.sigmoid(N_update)

        N_hidden = torch.concat([N_reset*N,N_new],dim=-1)
        N_hidden = N_hidden.reshape(-1,N_hidden.shape[-1])
        N_hidden = self.tanh(self.node_hidden(N_hidden).reshape(b,n,-1))
        N_ouput = (1-N_update)*N+N_update*N_hidden

        return N_ouput

class GraphPredictor(nn.Module):
    def __init__(self,input_node_dim, edge_node_dim):
        super().__init__()
        self.node_hidden = nn.Linear(input_node_dim, input_node_dim)
        self.edge_to_node = nn.Linear(edge_node_dim, input_node_dim)

        self.act = nn.Sigmoid()

    def forward(self,N,E):
        b,n,d = N.shape
        N_neigh = self.node_hidden(N.reshape(-1,d)).reshape(b,n,d)

        E_to_N = E.reshape(-1,E.shape[-1])
        E_to_N = self.edge_to_node(E_to_N)
        E_to_N = E_to_N.reshape(E.shape[:-1] + E_to_N.shape[-1:])
        E_to_N = self.act(E_to_N)

        N_pred = N + torch.sum(E_to_N*N_neigh.unsqueeze(2),dim=2)/n

        return N_pred
    
class GraphFocusor(nn.Module):
    def __init__(self,nb_slot,dim_slot):
        super().__init__()
        self.linear = nn.Linear(20,2)
        self.Pooling = GraphPooling(dim_slot,20)
        self.log_sigma = nn.Parameter(torch.rand(1,1,dim_slot))
        self.nb_slot = nb_slot
        self.tanh = nn.Tanh()

    def forward(self,N):
        b,n,d = N.shape
        x,N_pond = self.Pooling(N)
        x = self.linear(x)
        x = self.tanh(x)
        mu = torch.sum(N_pond*N,dim=1).unsqueeze(1).expand(b,self.nb_slot,d)
        log_sigma = self.log_sigma.expand(b,self.nb_slot,d)

        return x,mu,log_sigma
    
class GraphInitializer(nn.Module):
    def __init__(self,nb_slot,dim_slot):
        super().__init__()
        self.nb_slot = nb_slot
        self.dim_slot = dim_slot
        self.slots_mu = nn.Parameter(torch.randn(1, nb_slot, dim_slot))
        self.slots_sigma = nn.Parameter(torch.rand(1, nb_slot, dim_slot))

    def forward(self,x):
        mu = self.slots_mu.expand(x, self.nb_slot, self.dim_slot)
        log_sigma = self.slots_sigma.expand(x, self.nb_slot, self.dim_slot)
        
        return mu,log_sigma

class TransformerClassifier(nn.Module):
    def __init__(self,dim_slot,nhead=4):
        super().__init__()
        self.transformer = nn.Transformer(dim_slot,nhead=nhead,dim_feedforward=dim_slot*nhead//2,batch_first=True)
        self.fc1 = nn.Linear(dim_slot,1)
        self.fc2 = nn.Linear(dim_slot,1)
        self.tanh = nn.Tanh()
        #self.relu = nn.ReLU()
        self.target = nn.Parameter(torch.randn(1,2,dim_slot))
    def forward(self,x):
        target = self.target.expand(x.shape[0],self.target.shape[1],self.target.shape[2])
        x = self.transformer(x,target)
        x1 = self.fc1(x[:,0])
        x2 = self.fc1(x[:,1])
        result = torch.concat([x1,x2],dim=-1)
        result = self.tanh(result)
        return result

class GraphClassifier(nn.Module):
    def __init__(self,dim_slot,edge_dim):
        super().__init__()
        #self.ATconv = GraphAT(dim_slot,20,edge_dim,5)
        self.Pooling = GraphPooling(dim_slot,20)
        self.linear = nn.Linear(20,2)
        self.tanh = nn.Tanh()

    def forward(self,N):
        reshape = False
        if len(N.shape) == 4:
            b,f,n,d = N.shape
            N = N.reshape(-1,N.shape[-2],N.shape[-1])
            reshape = True
        #N,_ = self.ATconv(N,E)
        x,_ = self.Pooling(N)
        x = self.linear(x)
        x = self.tanh(x)
        if reshape:
            x = x.reshape(b,f,2)
        return x

class GraphClassifierVideo(nn.Module):
    def __init__(self,dim_slot,edge_dim):
        super().__init__()
        #self.ATconv = GraphAT(dim_slot,20,edge_dim,5)
        self.Pooling = GraphPooling(dim_slot,20)
        self.mu = nn.Parameter(torch.randn(1,20))
        self.log_sigma = nn.Parameter(torch.randn(1,20))
        self.gru = nn.GRUCell(20,20)
        self.linear = nn.Linear(20,2)
        self.tanh = nn.Tanh()

    def forward(self,N,E):
        if len(N.shape) == 4:
            b,f,n,d = N.shape
            N = N.reshape(-1,N.shape[-2],N.shape[-1])
            x,_ = self.Pooling(N)
            x = x.reshape(b,f,-1)
            mu = self.mu.expand(b,20)
            log_sigma = self.log_sigma.expand(b,20)
            last_elem = torch.normal(mu,torch.exp(log_sigma))
            elem = []
            for i in range(f):
                elem.append(self.gru(x[:,i],last_elem))
                last_elem = elem[-1]
            x = torch.stack(elem).permute(1,0,2).reshape(-1,20)
            x = self.linear(x)
            x = self.tanh(x)
            x = x.reshape(b,f,2)
        else:
            #N,_ = self.ATconv(N,E)
            x = self.Pooling(N)
            x = self.linear(x)
            x = self.tanh(x)
        return x



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
        # `x` has shape: [batch_size, width*height, input_size].
        
        # Slot Attention module.
        slots = self.slots_attention(init_slots,k,v,add_rest=False)
        # `slots` has shape: [batch_size, num_slots, slot_size].
        # """Broadcast slot features to a 2D grid and collapse slot dimension.""".
        recons,masks = self.decoding(slots)
        # `recons` has shape: [batch_size, num_slots, width, height, num_channels].
        # `masks` has shape: [batch_size, num_slots, width, height, 1].

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
        # `image` has shape: [batch_size, num_channels, width, height].
        if len(image.shape) == 4:
            # Convolutional encoder with position embedding.
            k,v = self.encoding(image)
            # `x` has shape: [batch_size, width*height, input_size].
            
            # Slot Attention module.
            mu,log_sigma = self.slots_attention(init_slots,k,v,attn_fc=attn_fc)
            # `slots` has shape: [batch_size, num_slots, slot_size].

            slots = self.reparameterize(mu,log_sigma)

            # """Broadcast slot features to a 2D grid and collapse slot dimension.""".
            recons,masks = self.decoding(slots)

            #masks_attention =  self.attention_mask(slots,k)

            return recons,masks,slots,mu,log_sigma

