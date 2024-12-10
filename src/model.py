import math
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass
from einops import rearrange, repeat, einsum
import random
from abc import abstractmethod

class BaseModel(nn.Module):
    def __init__(self, num_nodes, input_dim, output_dim, seq_length=12, horizon=12):
        super(BaseModel, self).__init__()
        self.num_nodes = num_nodes
        self.input_dim = input_dim
        self.output_dim = output_dim
        seq_length = seq_length
        self.horizon = horizon

    @abstractmethod
    def forward(self):
        raise NotImplementedError

    def param_num(self):
        return sum([param.nelement() for param in self.parameters()])

class S6(nn.Module):
    def __init__(self, args):
        super(S6, self).__init__()

        self.d_model = args.model_dim
        self.device = args.device
        self.num_nodes = args.num_nodes
        self.batch_size = args.batch_size
        self.seq_len = args.seq_length
        self.state_size = args.state_size
        self.d_inner = args.inner_dim

        self.fc1 = nn.Linear(self.d_inner, self.d_inner, device=self.device)
        self.fc2 = nn.Linear(self.d_inner, self.state_size, device=self.device)
        self.fc3 = nn.Linear(self.d_inner, self.state_size, device=self.device)

        self.A = nn.Parameter(F.normalize(torch.ones(self.num_nodes, self.d_inner, self.state_size, device=self.device), p=3, dim=-1))
        nn.init.xavier_uniform_(self.A)

        self.B = torch.zeros(self.batch_size, self.seq_len, self.num_nodes, self.state_size, device=self.device)
        self.C = torch.zeros(self.batch_size, self.seq_len, self.num_nodes, self.state_size, device=self.device)

        self.delta = torch.zeros(self.batch_size, self.seq_len, self.num_nodes, self.d_inner, device=self.device)
        self.dA = torch.zeros(self.batch_size, self.seq_len, self.num_nodes, self.d_inner, self.state_size, device=self.device)
        self.dB = torch.zeros(self.batch_size, self.seq_len, self.num_nodes, self.d_inner, self.state_size, device=self.device)

        # h  [self.d_model, seq_len, self.d_model, state_size]
        self.h = torch.zeros(self.batch_size, self.seq_len, self.num_nodes, self.d_inner, self.state_size, device=self.device)
        self.y = torch.zeros(self.batch_size, self.seq_len, self.num_nodes, self.d_inner, device=self.device)

    def discretization(self):
        self.dB = torch.einsum("blnd,blnr->blndr", self.delta, self.B)
        self.dA = torch.exp(torch.einsum("blnd,ndr->blndr", self.delta, self.A))
        return self.dA, self.dB

    def forward(self, x):
        # Algorithm 2  MAMBA paper=
        self.B = self.fc2(x)
        self.C = self.fc3(x)
        self.delta = F.softplus(self.fc1(x))

        self.discretization()
        h = torch.zeros(x.size(0), self.seq_len, self.num_nodes, self.d_inner, self.state_size, device=x.device) # h [self.d_model, seq_len, self.d_model, state_size]
        y = torch.zeros_like(x)
        
        h =  torch.einsum('blndr,blndr->blndr', self.dA, h) + rearrange(x, "b l n d -> b l n d 1") * self.dB
        y = torch.einsum('blnr,blndr->blnd', self.C, h) # y  [self.d_model, seq_len, d_model]

        return y

class MambaBlock(nn.Module):
    def __init__(self,args):
        super(MambaBlock, self).__init__()
        self.args = args
        self.ssm = args.ssm

        d_model = args.model_dim
        device = args.device
        seq_len = args.seq_length
        d_inner = args.inner_dim

        self.inp_proj = nn.Linear(d_model, d_inner, device=device)
        self.out_proj = nn.Linear(d_inner, d_model, device=device)
 
        self.D = nn.Linear(d_model, d_inner, device=device) # For residual skip connection

        self.out_proj.bias._no_weight_decay = True # Set _no_weight_decay attribute on bias
 
        nn.init.constant_(self.out_proj.bias, 1.0)  # Initialize bias to a small constant value
 
        self.S6 = S6(args)

        self.conv = nn.Conv1d(seq_len, seq_len, kernel_size=3, padding=1, device=device) # Add 1D convolution with kernel size 3
 
        self.conv_linear = nn.Linear(d_inner, d_inner, device=device) # Add linear layer for conv output

        self.norm = RMSNorm(d_model, device=device)  # rmsnorm
 
    def forward(self, x):
        x = self.norm(x)
        x_proj = self.inp_proj(x)
        # x_conv = self.conv(x_proj) # Add 1D convolution with kernel size 3
        x_conv_act = F.silu(x_proj)
        x_conv_out = self.conv_linear(x_conv_act) # Add linear layer for conv output
        
        if self.ssm:
            x_ssm = self.S6(x_conv_out)
        else:
            x_ssm = x_conv_out
        x_act = F.silu(x_ssm)  # Swish activation can be implemented as x * sigmoid(x)
        # residual skip connection with nonlinearity introduced by multiplication
        x_residual = F.silu(self.D(x))
        x_combined = x_act * x_residual
        x_out = self.out_proj(x_combined)

        return x_out
     
class ResidualBlock(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.mixer = MambaBlock(self.args)
        self.norm = RMSNorm(self.args.model_dim, device=args.device)

    def forward(self, x):
        B, T, N, D = x.shape
        y = self.norm(x)
        output = self.mixer(y).reshape(B, T, N, D) + x
        return output

class RMSNorm(nn.Module):
    def __init__(self,
                 d_model: int,
                 device: str,
                 eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(d_model).to(device))

    def forward(self, x):
        output = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps) * self.weight
        return output
    
#-------------------------------------------

class TFMamba(BaseModel):
    def __init__(self, args):
        #super(TFMamba, self).__init__(args.num_nodes, args.input_dim, args.output_dim, args.seq_length, args.horizon)
        super(TFMamba, self).__init__(args.num_nodes, args.input_dim, args.output_dim, args.seq_length, args.horizon)
        self.args = args
        self.args.tod_embedding_dim = args.tod_embedding_dim
        self.args.model_dim = (
            args.input_embedding_dim
            + args.tod_embedding_dim
            + args.dow_embedding_dim
            + args.spatial_embedding_dim
            + args.adaptive_embedding_dim
        )
        self.num_heads = args.num_heads
        self.num_layers = args.num_layers
        self.use_mixed_proj = args.use_mixed_proj

        self.input_proj = nn.Linear(self.args.input_dim, self.args.input_embedding_dim)
        if self.args.tod_embedding_dim > 0:
            self.tod_embedding = nn.Embedding(self.args.steps_per_day, self.args.tod_embedding_dim)
        if self.args.dow_embedding_dim > 0:
            self.dow_embedding = nn.Embedding(7, self.args.dow_embedding_dim)
        if self.args.spatial_embedding_dim > 0:
            self.node_emb = nn.Parameter(torch.empty(self.num_nodes, self.args.spatial_embedding_dim))
            nn.init.xavier_uniform_(self.node_emb)
        if self.args.adaptive_embedding_dim > 0:
            self.adaptive_embedding = nn.init.xavier_uniform_(nn.Parameter(torch.empty(self.args.seq_length, self.args.num_nodes, self.args.adaptive_embedding_dim)))

        if self.use_mixed_proj:
            self.output_proj = nn.Linear(self.args.seq_length * self.args.model_dim, self.args.horizon * self.args.output_dim)
        else:
            self.temporal_proj = nn.Linear(self.args.seq_length, self.args.horizon)
            self.output_proj = nn.Linear(self.args.model_dim, self.output_dim)

        self.args.state_size = args.state_size
        self.args.inner_dim = 2*self.args.model_dim
        self.mamba_layers = nn.ModuleList([ResidualBlock(self.args) for _ in range(self.args.num_layers)])

    def forward(self, x, label=None):
        # x: (batch_size, seq_length, num_nodes, input_dim=3)
        B,T,N,D = x.shape
        if self.args.tod_embedding_dim > 0:
            tod = x[..., 1]
        if self.args.dow_embedding_dim > 0:
            dow = x[..., 2]
        x = x[..., : self.input_dim]
        x = self.input_proj(x)  # =>(B, T, N, E=24)
        features = [x]
        if self.args.tod_embedding_dim > 0:
            tod_emb = self.tod_embedding((tod * self.args.steps_per_day).long())  # (batch_size, seq_length, num_nodes, tod_embedding_dim)
            features.append(tod_emb)
        if self.args.dow_embedding_dim > 0:
            dow_emb = self.dow_embedding(dow.long())  # (batch_size, seq_length, num_nodes, dow_embedding_dim)
            features.append(dow_emb)
        if self.args.spatial_embedding_dim > 0:
            spatial_emb = self.node_emb.expand(self.args.batch_size, self.args.seq_length, *self.node_emb.shape)
            features.append(spatial_emb)
        if self.args.adaptive_embedding_dim > 0:
            adp_emb = self.adaptive_embedding.expand(
                size=(self.args.batch_size, *self.adaptive_embedding.shape)
            )
            features.append(adp_emb)
        x = torch.cat(features, dim=-1)  # (B, T, N, model_dim=152)

        for man in self.mamba_layers:
            x = man(x)

        if self.use_mixed_proj:
            out = x.transpose(1, 2)  # (batch_size, num_nodes, seq_length, model_dim)
            out = out.reshape(self.args.batch_size, self.num_nodes, self.args.seq_length * self.args.model_dim)
            out = self.output_proj(out).view(self.args.batch_size, self.num_nodes, self.horizon, self.output_dim)
            out = out.transpose(1, 2)  # (batch_size, horizon, num_nodes, output_dim)
        else:
            out = x.transpose(1, 3)  # (batch_size, model_dim, num_nodes, seq_length)
            out = self.temporal_proj(out)# (batch_size, model_dim, num_nodes, horizon)
            out = self.output_proj(out.transpose(1, 3))  #(batch_size, horizon, num_nodes, output_dim)
        return out
