import torch
import torch.nn as nn
import torch.nn.functional as F
import random
from abc import abstractmethod

class BaseModel(nn.Module):
    def __init__(self, node_num, input_dim, output_dim, seq_len=12, horizon=12):
        super(BaseModel, self).__init__()
        self.node_num = node_num
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.seq_len = seq_len
        self.horizon = horizon

    @abstractmethod
    def forward(self):
        raise NotImplementedError

    def param_num(self):
        return sum([param.nelement() for param in self.parameters()])

class S6(nn.Module):
    def __init__(self, seq_len, model_dim, state_size, device):
        super(S6,self).__init__()

        self.fc1 = nn.Linear(model_dim, model_dim, device=device)
        self.fc2 = nn.Linear(model_dim, state_size, device=device)
        self.fc3 = nn.Linear(model_dim, state_size, device=device)

        self.seq_len = seq_len
        self.model_dim = model_dim
        self.state_size = state_size

        self.A = nn.Parameter(F.normalize(torch.ones(model_dim, state_size, device=device), p=2, dim=-1)) #这里为什么要进行归一化呢？
        nn.init.xavier_uniform_(self.A)
        self.B = torch.zeros(batch_size, self.seq_len, self.state_size, device=device)
        self.C = torch.zeros(batch_size, self.seq_len, self.state_size, device=device)
        self.delta = torch.zeros(batch_size, self.seq_len, self.model_dim, device=device)
        self.dA = torch.zeros(batch_size, self.seq_len, self.model_dim, self.state_size, device=device)
        self.dB = torch.zeros(batch_size, self.seq_len, self.model_dim, self.state_size, device=device)

        # h [B:batch_size, L:seq_len, D:model_dim, S:state_size]
        self.h = torch.zeros(batch_size, self.seq_len, self.model_dim, self.state_size, device=device)
        self.y = torch.zeros(batch_size, self.seq_len, self.model_dim, device=device)

    def discretization(self):
        self.dB = torch.einsum("bld,bln->bldn", self.delta, self.B)
        self.dA = torch.exp(torch.einsum("bld,dn->bldn", self.delta, self.A))
        return self.dA, self.dB

    def forward(self, x):
        # Algorithm 2 in MAMBA paper
        self.B = self.fc2(x)
        self.C = self.fc3(x)
        self.delta = F.softplus(self.fc1(x))
        self.discretization()
        if DIFFERENT_H_STATES_RECURRENT_UPDATE_MECHANISM:
            global current_batch_size
            current_batch_size = x.shape[0]

            if self.h.shape[0] != current_batch_size:
                different_batch_size = True
                h_new = torch.einsum('bldn,bldn->bldn', self.dA, self.h[:current_batch_size, ...]) + rearrange(x,'b l d -> b l d 1') * self.dB
            else:
                different_batch_size = False
                h_new = torch.einsum('bldn,bldn->bldn', self.dA, self.h) + rearrange(x,'b l d -> b l d 1') * self.dB

            # y [B, L, D]
            self.y = torch.einsum('bln,bldn->bld', self.C, h_new)

            global temp_buffer
            temp_buffer = h_new.detach().clone() if not self.h.requires_grad else h_new.clone()

            return self.y
        
        else:
            # h [B, L, D, S]
            h = torch.zeros(x.size(0), self.seq_len, self.model_dim, self.state_size, device=device)
            y = torch.zeros_like(x)

            h = torch.einsum('bldn,bldn->bldn', self.dA, h) + rearrange(x, 'b l d -> b l d 1') * self.dB

            # y [B, L, D]
            y = torch.einsum('bln,bldn->bld', self.C, h)  

            return y

class RMSNorm(nn.Module):
    def __init__(self, model_dim: int, eps: float = 1e-5, device: str = 'cpu'):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(model_dim, device=device))

    def forward(self, x):
        output = x * torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps) * self.weight
        return output

class MambaBlock(nn.Module):
    def __init__(self, seq_len, model_dim, state_size, device):
        super(MambaBlock, self).__init__()
        self.in_proj = nn.Linear(model_dim, 2*model_dim, device=device)
        self.out_proj = nn.Linear(2*model_dim, model_dim, device=device)
        self.D = nn.Linear(model_dim, 2*model_dim, device=device) # For residual skip connection
        self.out_proj.bias._no_weight_decay = True # Set _no_weight_decay attribute on bias
        nn.init.constant_(self.out_proj.bias, 1.0) # Initialize bias to a small constant value
        self.S6 = S6(seq_len, 2*model_dim, state_size, device)
        self.conv = nn.Conv1d(seq_len, seq_len, kernel_size=3, padding=1, device=device) # Add 1D convolution with kernel size 3
        self.conv_linear = nn.Linear(2*model_dim, 2*model_dim, device=device)         # Add linear layer for conv output
        self.norm = RMSNorm(model_dim, device=device)  # rmsnorm

    def forward(self, x):
        '''
        x_proj.shape = torch.Size([B, L, 2*D])
        x_conv.shape = torch.Size([B, L, 2*D])
        x_conv_act.shape = torch.Size([B, L, 2*D])
        '''
        x = self.norm(x)
        x_proj = self.in_proj(x)
        x_conv = self.conv(x_proj) # Add 1 D convolution with kernel size 3
        x_conv_act = F.silu(x_conv)
        x_conv_out = self.conv_linear(x_conv_act) # Add linear layer for conv output
        x_ssm = self.S6(x_conv_out)
        x_act = F.silu(x_ssm) # Switch activation can be implemented as x*sigmoid(x)
        x_residual = F.silu(self.D(x))  # residual skip connection with nonlinearity introduced by multiplication
        x_combined = x_act * x_residual
        x_out = self.out_proj(x_combined)
        return x_out

class TFMamba(nn.Module):
    def __init__(self, num_nodes, device, horizon, in_steps=12, out_steps=12, steps_per_day=288, input_dim=3, output_dim=1, input_embedding_dim=24, tod_embedding_dim=24, dow_embedding_dim=24, spatial_embedding_dim=0, adaptive_embedding_dim=80, feed_forward_dim=256, num_heads=4, num_layers=3, dropout=0.1, use_mixed_proj=True, **args):
        super().__init__(**args)
        self.device = device
        self.horizon = horizon
        self.num_nodes = num_nodes
        self.in_steps = in_steps
        self.out_steps = out_steps
        self.steps_per_day = steps_per_day
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.input_embedding_dim = input_embedding_dim
        self.tod_embedding_dim = tod_embedding_dim
        self.dow_embedding_dim = dow_embedding_dim
        self.spatial_embedding_dim = spatial_embedding_dim
        self.adaptive_embedding_dim = adaptive_embedding_dim
        self.model_dim = (
            input_embedding_dim
            + tod_embedding_dim
            + dow_embedding_dim
            + spatial_embedding_dim
            + adaptive_embedding_dim
        )
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.use_mixed_proj = use_mixed_proj

        self.input_proj = nn.Linear(input_dim, input_embedding_dim)
        if tod_embedding_dim > 0:
            self.tod_embedding = nn.Embedding(steps_per_day, tod_embedding_dim)
        if dow_embedding_dim > 0:
            self.dow_embedding = nn.Embedding(7, dow_embedding_dim)
        if spatial_embedding_dim > 0:
            self.node_emb = nn.Parameter(
                torch.empty(self.num_nodes, self.spatial_embedding_dim)
            )
            nn.init.xavier_uniform_(self.node_emb)
        if adaptive_embedding_dim > 0:
            self.adaptive_embedding = nn.init.xavier_uniform_(
                nn.Parameter(torch.empty(in_steps, num_nodes, adaptive_embedding_dim))
            )

        if use_mixed_proj:
            self.output_proj = nn.Linear(in_steps * self.model_dim, out_steps * output_dim)
        else:
            self.temporal_proj = nn.Linear(in_steps, out_steps)
            self.output_proj = nn.Linear(self.model_dim, self.output_dim)

        self.mamba_layers = nn.ModuleList(
            [
                MambaBlock(in_steps, self.model_dim, state_size=128, device=device)
                for _ in range(num_layers)
            ]
        )

    def forward(self, x, label=None):
        # x: (batch_size, in_steps, num_nodes, input_dim3)
        B,T,N,D = x.shape
        # if self.tod_embedding_dim > 0:
        #     tod = x[..., 1]
        # if self.dow_embedding_dim > 0:
        #     dow = x[..., 2]
        # x = x[..., : self.input_dim]
        x = self.input_proj(x)  # =>(B, T, N, E)
        features = [x]
        if self.tod_embedding_dim > 0:
            tod_emb = self.tod_embedding((tod * self.steps_per_day).long())  # (batch_size, in_steps, num_nodes, tod_embedding_dim)
            features.append(tod_emb)
        if self.dow_embedding_dim > 0:
            dow_emb = self.dow_embedding(
                dow.long()
            )  # (batch_size, in_steps, num_nodes, dow_embedding_dim)
            features.append(dow_emb)
        if self.spatial_embedding_dim > 0:
            spatial_emb = self.node_emb.expand(
                batch_size, self.in_steps, *self.node_emb.shape
            )
            features.append(spatial_emb)
        if self.adaptive_embedding_dim > 0:
            adp_emb = self.adaptive_embedding.expand(
                size=(batch_size, *self.adaptive_embedding.shape)
            )
            features.append(adp_emb)
        x = torch.cat(features, dim=-1)  # (batch_size, in_steps, num_nodes, model_dim)

        for man in self.mamba_layers:
            x = mamba(x)
        # for attn in self.attn_layers_t:
        #     x = attn(x, dim=1)
        # for attn in self.attn_layers_s:
        #     x = attn(x, dim=2)
        # # (batch_size, in_steps, num_nodes, model_dim)

        if self.use_mixed_proj:
            out = x.transpose(1, 2)  # (batch_size, num_nodes, in_steps, model_dim)
            out = out.reshape(batch_size, self.num_nodes, self.in_steps * self.model_dim)
            out = self.output_proj(out).view(
                batch_size, self.num_nodes, self.out_steps, self.output_dim
            )
            out = out.transpose(1, 2)  # (batch_size, out_steps, num_nodes, output_dim)
        else:
            out = x.transpose(1, 3)  # (batch_size, model_dim, num_nodes, in_steps)
            out = self.temporal_proj(out)# (batch_size, model_dim, num_nodes, out_steps)
            out = self.output_proj(out.transpose(1, 3))  #(batch_size, out_steps, num_nodes, output_dim)
        return out
