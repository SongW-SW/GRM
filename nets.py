import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from torch_sparse import SparseTensor, matmul
from torch_geometric.nn import GCNConv, SAGEConv, GATConv, APPNP, MessagePassing
from torch_geometric.nn.conv.gcn_conv import gcn_norm
import scipy.sparse
import numpy as np
import math
from torch.nn import Linear
from torch_geometric.nn import GCNConv, GATConv, GINConv, SAGEConv, SGConv, global_mean_pool, global_max_pool, global_add_pool, SAGPooling
from torch_geometric.nn.inits import reset
class Encoder(nn.Module):
    def __init__(self, in_channels, hidden_channels, encoder_type='GCN'):
        super(Encoder, self).__init__()
        self.hidden_channels = hidden_channels
        if encoder_type=='GCN':
            self.conv1 = GCNConv(in_channels, self.hidden_channels)
        elif encoder_type=='GAT':
            self.conv1 = GATConv(in_channels, self.hidden_channels)
        elif encoder_type=='GraphSAGE':
            self.conv1 = SAGEConv(in_channels, self.hidden_channels)
        elif encoder_type=='SGC':
            self.conv1 = SGConv(in_channels, self.hidden_channels)
        elif encoder_type=='GIN':
            self.mlp = nn.Linear(in_channels, self.hidden_channels)
            self.conv1 = GINConv(self.mlp)

        self.prelu1 = nn.PReLU(self.hidden_channels)

    def forward(self, x, edge_index, edge_attr=None):
        if edge_attr!=None:
            x1 = self.conv1(x, edge_index, edge_attr)
        else:
            x1 = self.conv1(x, edge_index)
        x1 = self.prelu1(x1)
        x1 = F.normalize(x1)
        return x1



class Pool(nn.Module):
    def __init__(self, in_channels, ratio=1.0):
        super(Pool, self).__init__()
        self.sag_pool = SAGPooling(in_channels, ratio)
        self.lin1 = torch.nn.Linear(in_channels * 2, in_channels)
    def forward(self, x, edge, batch, type='mean_pool'):
        if type == 'mean_pool':
            return global_mean_pool(x, batch)
        elif type == 'max_pool':
            return global_max_pool(x, batch)
        elif type == 'sum_pool':
            return global_add_pool(x, batch)
        elif type == 'sag_pool':
            x1, _, _, batch, _, _ = self.sag_pool(x, edge, batch=batch)
            return global_mean_pool(x1, batch)

class batch_gnn(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, encoder=None, output_type='both'):
        super(batch_gnn, self).__init__()
        self.output_type=output_type
        self.hidden_channels = hidden_channels
        if encoder==None:
            self.encoder=Encoder(in_channels, hidden_channels,encoder_type='GCN')
        else:
            self.encoder=encoder

        self.pool=Pool(in_channels=hidden_channels)
        self.sigmoid = nn.Sigmoid()
        # self.prompt = nn.parameter.Parameter(torch.rand(4))
        self.reset_parameters()

    def reset_parameters(self):
        reset(self.encoder)
        reset(self.pool)


    def forward(self, x, edge_index, batch=None, index=None, edge_attr=None, return_hidden=True):
        r""" Return node and subgraph representations of each node before and after being shuffled """
        hidden = self.encoder(x, edge_index, edge_attr, return_hidden=return_hidden)



        if index is None:
            return hidden

        if type(hidden)==tuple:
            if self.output_type=='pool':
                return self.pool(hidden[0], edge_index, batch), self.pool(hidden[1], edge_index, batch)
            else:
                return hidden

        z = hidden[index]
        summary = self.pool(hidden, edge_index, batch)

        if self.output_type=='both':
            return z, summary
        elif self.output_type=='index':
            return z
        elif self.output_type=='pool':
            return summary
        elif self.output_type=='hidden':
            return hidden




class GCN(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers,
                 dropout, save_mem=True, use_bn=True):
        super(GCN, self).__init__()

        self.convs = nn.ModuleList()
        self.bns = nn.ModuleList()

        self.convs.append(
            GCNConv(in_channels, hidden_channels, cached=not save_mem, normalize=not save_mem))
        self.bns.append(nn.BatchNorm1d(hidden_channels))
        for _ in range(num_layers - 2):
            self.convs.append(
                GCNConv(hidden_channels, hidden_channels, cached=not save_mem, normalize=not save_mem))
            self.bns.append(nn.BatchNorm1d(hidden_channels))
        self.convs.append(
            GCNConv(hidden_channels, out_channels, cached=not save_mem, normalize=not save_mem))

        self.dropout = dropout
        self.activation = F.relu
        self.use_bn = use_bn

    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()
        for bn in self.bns:
            bn.reset_parameters()


    def forward(self, x, edge_index, edge_weight=None, return_hidden=False):
        for i, conv in enumerate(self.convs[:-1]):
            x = conv(x, edge_index, edge_weight)
            if self.use_bn:
                x = self.bns[i](x)
            x = self.activation(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        if not return_hidden:
            return self.convs[-1](x, edge_index)
        else:
            return x

class SAGE(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers=2,
                 dropout=0.5, use_bn=True):
        super(SAGE, self).__init__()

        self.convs = nn.ModuleList()
        self.convs.append(
            SAGEConv(in_channels, hidden_channels))

        self.bns = nn.ModuleList()
        self.bns.append(nn.BatchNorm1d(hidden_channels))
        for _ in range(num_layers - 2):
            self.convs.append(
                SAGEConv(hidden_channels, hidden_channels))
            self.bns.append(nn.BatchNorm1d(hidden_channels))

        self.convs.append(
            SAGEConv(hidden_channels, out_channels))

        self.dropout = dropout
        self.activation = F.relu
        self.use_bn = use_bn

    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()
        for bn in self.bns:
            bn.reset_parameters()


    def forward(self, x, edge_index,return_hidden=False):
        for i, conv in enumerate(self.convs[:-1]):
            x = conv(x, edge_index)
            if self.use_bn:
                x = self.bns[i](x)
            x = self.activation(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        if not return_hidden:
            return self.convs[-1](x, edge_index)
        else:
            return x



class GAT(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers=2,
                 dropout=0.5, heads=2):
        super(GAT, self).__init__()

        self.convs = nn.ModuleList()
        self.convs.append(
            GATConv(in_channels, hidden_channels, heads=heads, concat=True))

        self.bns = nn.ModuleList()
        self.bns.append(nn.BatchNorm1d(hidden_channels*heads))
        for _ in range(num_layers - 2):

            self.convs.append(
                    GATConv(hidden_channels*heads, hidden_channels, heads=heads, concat=True) ) 
            self.bns.append(nn.BatchNorm1d(hidden_channels*heads))

        self.convs.append(
            GATConv(hidden_channels*heads, out_channels, heads=heads, concat=False))

        self.dropout = dropout
        self.activation = F.elu 

    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()
        for bn in self.bns:
            bn.reset_parameters()


    def forward(self, x, edge_index):
        for i, conv in enumerate(self.convs[:-1]):
            x = conv(x, edge_index)
            x = self.bns[i](x)
            x = self.activation(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.convs[-1](x, edge_index)
        return x

class GPR_prop(MessagePassing):
    '''
    GPRGNN, from original repo https://github.com/jianhao2016/GPRGNN
    propagation class for GPR_GNN
    '''

    def __init__(self, K, alpha, Init, Gamma=None, bias=True, **kwargs):
        super(GPR_prop, self).__init__(aggr='add', **kwargs)
        self.K = K
        self.Init = Init
        self.alpha = alpha

        assert Init in ['SGC', 'PPR', 'NPPR', 'Random', 'WS']
        if Init == 'SGC':
            # SGC-like
            TEMP = 0.0*np.ones(K+1)
            TEMP[alpha] = 1.0
        elif Init == 'PPR':
            # PPR-like
            TEMP = alpha*(1-alpha)**np.arange(K+1)
            TEMP[-1] = (1-alpha)**K
        elif Init == 'NPPR':
            # Negative PPR
            TEMP = (alpha)**np.arange(K+1)
            TEMP = TEMP/np.sum(np.abs(TEMP))
        elif Init == 'Random':
            # Random
            bound = np.sqrt(3/(K+1))
            TEMP = np.random.uniform(-bound, bound, K+1)
            TEMP = TEMP/np.sum(np.abs(TEMP))
        elif Init == 'WS':
            # Specify Gamma
            TEMP = Gamma

        self.temp = nn.Parameter(torch.tensor(TEMP))

    def reset_parameters(self):
        nn.init.zeros_(self.temp)
        for k in range(self.K+1):
            self.temp.data[k] = self.alpha*(1-self.alpha)**k
        self.temp.data[-1] = (1-self.alpha)**self.K

    def forward(self, x, edge_index, edge_weight=None):
        if isinstance(edge_index, torch.Tensor):
            edge_index, norm = gcn_norm(
                edge_index, edge_weight, num_nodes=x.size(0), dtype=x.dtype)
        elif isinstance(edge_index, SparseTensor):
            edge_index = gcn_norm(
                edge_index, edge_weight, num_nodes=x.size(0), dtype=x.dtype)
            norm = None

        hidden = x*(self.temp[0])
        for k in range(self.K):
            x = self.propagate(edge_index, x=x, norm=norm)
            gamma = self.temp[k+1]
            hidden = hidden + gamma*x
        return hidden

    def message(self, x_j, norm):
        return norm.view(-1, 1) * x_j

    def __repr__(self):
        return '{}(K={}, temp={})'.format(self.__class__.__name__, self.K,
                                          self.temp)

class GPRGNN(nn.Module):
    """GPRGNN, from original repo https://github.com/jianhao2016/GPRGNN"""

    def __init__(self, in_channels, hidden_channels, out_channels, Init='PPR', dprate=.5, dropout=.5, K=10, alpha=.1, Gamma=None, ppnp='GPR_prop'):
        super(GPRGNN, self).__init__()
        self.lin1 = nn.Linear(in_channels, hidden_channels)
        self.lin2 = nn.Linear(hidden_channels, out_channels)

        if ppnp == 'PPNP':
            self.prop1 = APPNP(K, alpha)
        elif ppnp == 'GPR_prop':
            self.prop1 = GPR_prop(K, alpha, Init, Gamma)

        self.Init = Init
        self.dprate = dprate
        self.dropout = dropout

    def reset_parameters(self):
        self.lin1.reset_parameters()
        self.lin2.reset_parameters()
        self.prop1.reset_parameters()

    def forward(self, x, edge_index,return_hidden=False):

        x = F.dropout(x, p=self.dropout, training=self.training)
        x = F.relu(self.lin1(x))

        if return_hidden:
            return x

        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.lin2(x)

        if self.dprate == 0.0:
            x = self.prop1(x, edge_index)
            return x
        else:
            x = F.dropout(x, p=self.dprate, training=self.training)
            x = self.prop1(x, edge_index)
            return x

class GCNIIConv(nn.Module):

    def __init__(self, in_features, out_features, residual=False):
        super(GCNIIConv, self).__init__()
        self.in_features = in_features

        self.out_features = out_features
        self.residual = residual
        self.weight = Parameter(torch.FloatTensor(self.in_features,self.out_features))
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.out_features)
        self.weight.data.uniform_(-stdv, stdv)

    def forward(self, x, adj, h0 , lamda, alpha, l):
        theta = math.log(lamda/l+1)
        hi = torch.spmm(adj, x)
        support = (1-alpha)*hi+alpha*h0
        output = theta*torch.mm(support, self.weight)+(1-theta)*support
        if self.residual:
            output = output+x
        return output

class GCNII(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers=2, dropout=0.5, lamda=1.0, alpha=0.1):
        super(GCNII, self).__init__()
        self.convs = nn.ModuleList()
        for _ in range(num_layers):
            self.convs.append(GCNIIConv(hidden_channels, hidden_channels))
        self.fcs = nn.ModuleList()
        self.fcs.append(nn.Linear(in_channels, hidden_channels))
        self.fcs.append(nn.Linear(hidden_channels, out_channels))
        self.act_fn = nn.ReLU()
        self.dropout = dropout
        self.alpha = alpha
        self.lamda = lamda

    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()
        for fc in self.fcs:
            fc.reset_parameters()

    def forward(self, x, edge_index):
        edge_index, norm = gcn_norm(
            edge_index, torch.ones(edge_index.size(1), dtype=torch.float).to(x.device), num_nodes=x.size(0), dtype=x.dtype)
        adj = torch.sparse.FloatTensor(
            edge_index, norm, (x.size(0), x.size(0)))
        _layers = []
        x = F.dropout(x, self.dropout, training=self.training)
        layer_inner = self.act_fn(self.fcs[0](x))
        _layers.append(layer_inner)
        for i,con in enumerate(self.convs):
            layer_inner = F.dropout(layer_inner, self.dropout, training=self.training)
            layer_inner = self.act_fn(con(layer_inner,adj,_layers[0],self.lamda,self.alpha,i+1))
        layer_inner = F.dropout(layer_inner, self.dropout, training=self.training)
        layer_inner = self.fcs[-1](layer_inner)
        return F.log_softmax(layer_inner, dim=1)


class VGCN_Encoder(torch.nn.Module):
    def __init__(self, in_channels, out_channels):
        super(VGCN_Encoder, self).__init__()
        self.conv1 = GCNConv(in_channels, 2 * out_channels, cached=False) # cached only for transductive learning
        self.conv_mu = GCNConv(2 * out_channels, out_channels, cached=False)
        self.conv_logstd = GCNConv(2 * out_channels, out_channels, cached=False)

    def forward(self, x, edge_index, edge_attr=None, return_hidden=True):

        #print(x.shape)
        #print(edge_index)

        x = self.conv1(x, edge_index).relu()
        return self.conv_mu(x, edge_index), self.conv_logstd(x, edge_index)



class prior_model(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(prior_model, self).__init__()
        self.mu = nn.Linear(in_channels, out_channels)
        self.logstd = nn.Linear(in_channels, out_channels)

    def forward(self, x):

        #print(x.shape)
        #print(edge_index)

        return self.mu(x), self.logstd(x)