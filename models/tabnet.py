import torch
import torch.nn as nn
import torch.nn.functional as F

class GBN(nn.Module):
    def __init__(self, inp, vbs=128, momentum=0.01):
        super().__init__()
        self.bn = nn.BatchNorm1d(inp, momentum=momentum)
        self.vbs = vbs

    def forward(self, x):
        if self.training:
            chunk = torch.chunk(x, x.size(0) // self.vbs, 0)
            res = [self.bn(y) for y in chunk]
            return torch.cat(res, 0)
        else:
            return self.bn(x)

class AttentionTransformer(nn.Module):
    def __init__(self, d_in, d_out, n_shared=2, n_ind=2, vbs=128, momentum=0.02):
        super().__init__()
        self.fc = nn.Linear(d_in, d_out * (n_shared + n_ind))
        self.n_shared = n_shared
        self.n_ind = n_ind
        self.d_out = d_out
        self.gbn = GBN(d_out * (n_shared + n_ind), vbs=vbs, momentum=momentum)

    def forward(self, x, prior):
        x = self.gbn(self.fc(x))
        shared = x[:, :self.d_out * self.n_shared].view(-1, self.n_shared, self.d_out)
        ind = x[:, self.d_out * self.n_shared:].view(-1, self.n_ind, self.d_out)
        
        shared = torch.sum(shared, dim=1)
        ind = torch.sum(ind, dim=1)
        
        out = shared + ind
        out = out * prior
        out = F.softmax(out, dim=-1)
        return out

class FeatureTransformer(nn.Module):
    def __init__(self, d_in, d_out, shared_layers, n_ind=2, vbs=128, momentum=0.02):
        super().__init__()
        self.shared = nn.ModuleList()
        self.ind = nn.ModuleList()
        
        for i in range(shared_layers):
            self.shared.append(nn.Linear(d_in if i == 0 else d_out, d_out))
            self.shared.append(nn.ReLU())
            self.shared.append(GBN(d_out, vbs=vbs, momentum=momentum))
        
        for i in range(n_ind):
            self.ind.append(nn.Linear(d_out, d_out))
            self.ind.append(nn.ReLU())
            self.ind.append(GBN(d_out, vbs=vbs, momentum=momentum))

    def forward(self, x):
        # Shared layers
        for layer in self.shared:
            x = layer(x)
        
        # Independent layers
        outputs = []
        for i in range(0, len(self.ind), 3):
            out = x
            for layer in self.ind[i:i+3]:
                out = layer(out)
            outputs.append(out)
        
        return torch.stack(outputs, dim=1).sum(dim=1)

class TabNet(nn.Module):
    def __init__(self, input_dim, output_dim, n_d=64, n_a=64, n_shared=2, 
                 n_ind=2, n_steps=3, vbs=128, momentum=0.02):
        super().__init__()
        self.n_steps = n_steps
        self.output_dim = output_dim
        
        self.bn = nn.BatchNorm1d(input_dim, momentum=0.01)
        self.fc = nn.Linear(input_dim, n_d + n_a)
        
        self.att_transformers = nn.ModuleList()
        self.feat_transformers = nn.ModuleList()
        
        for i in range(n_steps):
            self.att_transformers.append(
                AttentionTransformer(n_a, input_dim, n_shared, n_ind, vbs, momentum)
            )
            self.feat_transformers.append(
                FeatureTransformer(n_d, n_d + n_a, n_shared, n_ind, vbs, momentum)
            )
        
        self.fc_out = nn.Linear(n_d, output_dim)

    def forward(self, x):
        x = self.bn(x)
        x = self.fc(x)
        
        d = x[:, :self.att_transformers[0].d_out]
        a = x[:, self.att_transformers[0].d_out:]
        
        prior = torch.ones_like(a)
        total_att = torch.zeros_like(a)
        
        for i in range(self.n_steps):
            att = self.att_transformers[i](a, prior)
            total_att += att
            
            masked_x = x * att
            feat = self.feat_transformers[i](masked_x)
            
            d = feat[:, :self.att_transformers[0].d_out]
            a = feat[:, self.att_transformers[0].d_out:]
            
            prior = prior * (0.9 - att)
        
        out = self.fc_out(d)
        return out

def create_tabnet(input_dim, output_dim):
    return TabNet(input_dim=input_dim, output_dim=output_dim)
