from typing import Optional, Tuple

import torch
from torch import Tensor

import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch_geometric.nn import VGAE as PyGVGAE
from torch_geometric.utils import to_dense_adj, dense_to_sparse, subgraph, k_hop_subgraph
from torch_geometric.typing import OptTensor
from torch_geometric.nn import global_mean_pool

#adding GVAE for l2aug
#gcn_conv.py and res_gcn.py are from GraphCL_Automated
from gcn_conv import GCNConv
from res_gcn import ResGCN_graphcl, vgae_encoder, vgae_decoder, vgae
import re

from nets import *
from utils_mp import *

class InnerProductDecoder_Domain(torch.nn.Module):

    def forward(self, z: Tensor, edge_index: Tensor, domain_embs: Tensor =None,
                          sigmoid: bool = True) -> Tensor:

        if domain_embs!=None:
            z=z*domain_embs
        value = (z[edge_index[0]] * z[edge_index[1]]).sum(dim=1)
        return torch.sigmoid(value) if sigmoid else value





# function from GraphCL_Automated
# default values are set according to GraphCL_Automated
def get_model_with_default_configs(model_name,
                                   num_features,
                                   num_classes,
                                   num_feat_layers=1,
                                   num_conv_layers=3,
                                   num_fc_layers=2,
                                   residual=False,
                                   hidden=128):
    # More default settings.
    res_branch = "BNConvReLU"
    global_pool = "sum"
    dropout = 0.0
    edge_norm = True

    def foo():
        # return ResGCN_graphcl(dataset=dataset, hidden=hidden, 
        # num_feat_layers=num_feat_layers, num_conv_layers=num_conv_layers, 
        # num_fc_layers=num_fc_layers, gfn=False, collapse=False, 
        # residual=residual, res_branch=res_branch, global_pool=global_pool, 
        # dropout=dropout, edge_norm=edge_norm), \
        return vgae_encoder(num_features=num_features,num_classes=num_classes,
                hidden=hidden,
                num_feat_layers=num_feat_layers, 
                num_conv_layers=num_conv_layers, 
                num_fc_layers=num_fc_layers, gfn=False, collapse=False, 
                residual=residual, res_branch=res_branch, global_pool=global_pool, 
                dropout=dropout, edge_norm=edge_norm), \
                vgae_decoder()
    # else:
        # raise ValueError("Unknown model {}".format(model_name))
    return foo

class Model_L2aug_Multi(nn.Module):
    def __init__(self, args, ns, c, d, gnn, vgae, device):
        super(Model_L2aug_Multi, self).__init__()
        if gnn == 'gcn':
            self.gnn = GCN(in_channels=d,
                            hidden_channels=args.hidden_channels,
                            out_channels=c,
                            num_layers=args.num_layers,
                            dropout=args.dropout,
                            use_bn=not args.no_bn)

        elif gnn == 'sage':
            self.gnn = SAGE(in_channels=d,
                        hidden_channels=args.hidden_channels,
                        out_channels=c,
                        num_layers=args.num_layers,
                        dropout=args.dropout)

        elif gnn == 'gat':
            self.gnn = GAT(in_channels=d,
                           hidden_channels=args.hidden_channels,
                           out_channels=c,
                           num_layers=args.num_layers,
                           dropout=args.dropout,
                           heads=args.gat_heads)

        elif gnn == 'gpr':
            self.gnn = GPRGNN(in_channels=d,
                        hidden_channels=args.hidden_channels,
                        out_channels=c,
                        dropout=args.dropout,
                        alpha=args.gpr_alpha,
                        )
            
        elif gnn == 'gcnii':
            self.gnn = GCNII(in_channels=d,
                        hidden_channels=args.hidden_channels,
                        out_channels=c,
                        num_layers=args.num_layers,
                        dropout=args.dropout,
                        alpha=args.gcnii_alpha,
                        lamda=args.gcnii_lamda)

        self.p = 0.2 # what's this?
        self.ns = ns # number of nodes per tr graph
        self.device = device
        self.gnn_net = gnn
        self.args = args

        # self.gl = nn.ModuleList()
        # for n in self.ns:
            # self.gl.append(Graph_Editer(args.K, n, device))
        
        if vgae == "pyg":
            print('created pyg GVAE')
            print('d{} c{}'.format(d,c))
            #self.vgae = PyGVGAE(VGCN_Encoder(d, c))  # new line
            self.vgae = PyGVGAE(batch_gnn(in_channels=args.hidden_channels*2, hidden_channels=args.hidden_channels, encoder=VGCN_Encoder(args.hidden_channels*2, args.hidden_channels), output_type='hidden').cuda())


        self.prior_model=prior_model(args.hidden_channels, c)

        self.batch_gnn=batch_gnn(in_channels=args.hidden_channels, hidden_channels=args.hidden_channels, encoder=GCN(in_channels=args.hidden_channels,
                                                                                                                       hidden_channels=args.hidden_channels,
                                                                                                                       out_channels=c,
                                                                                                                       num_layers=2,   #args.num_layers,
                                                                                                                       dropout=args.dropout,
                                                                                                                       use_bn=not args.no_bn), output_type='pool').cuda()
        self.decoder=InnerProductDecoder_Domain()


        self.max_size=20

        self.use_torch_sub=False
        self.no_edge_gen=False


    def inference(self, data, K=3000):
        #self.vgae.training=True
        # load data to GPU

        n = data.graph['node_feat'].shape[0]
        if self.args.dataset=='ogb-arxiv':
            #sample_idx=np.random.choice(list(range(n)), size=n//20, replace=False)

            sample_idx= np.random.choice(np.where(data.test_mask == True)[0], size=n//20, replace=False)
        elif self.args.dataset=='elliptic':
            sample_idx = np.where(data.mask == True)[0]
        else:
            sample_idx=np.array(list(range(n)))

        out=self.forward(data,sample_idx=sample_idx)

        return out, sample_idx

    # Gumbel Logistic
    def gumbel_sigmoid(self, logits, tau=1):
        # from https://github.com/ElementAI/causal_discovery_toolbox/blob/master/cdt/utils/torch.py

        logistic_noise = self._sample_logistic(logits.size(), out=logits.data.new())
        y = logits + logistic_noise
        return torch.sigmoid(y / tau)
    
    def _sample_logistic(self, shape, out=None):
        # from https://github.com/ElementAI/causal_discovery_toolbox/blob/master/cdt/utils/torch.py

        U = out.resize_(shape).uniform_() if out is not None else torch.rand(shape)
        #U2 = out.resize_(shape).uniform_() if out is not None else th.rand(shape)
        return torch.log(U) - torch.log(1-U)

    # from https://pytorch-geometric.readthedocs.io/en/latest/_modules/torch_geometric/utils/dropout.html#dropout_adj
    # def filter_adj(self, row: Tensor, col: Tensor, edge_attr: OptTensor,
    #            mask: Tensor) -> Tuple[Tensor, Tensor, OptTensor]:
    #     return row[mask], col[mask                            ], None if edge_attr is None else edge_attr[mask]

    def edge_gen(self, x, edge_index, n, K, batch=None, index=None, sizes=None):


        mu, logstd = self.vgae.encoder(batch.x, batch.edge_index, batch.batch, index) #[batch, args.hidden]


        mu_total=[]    #[batch, decode_size, -1]
        logstd_total=[]

        for i in range(len(index)):
            if i!= len(index)-1:
                mu_total.append(mu[index[i]:index[i+1], :])
                logstd_total.append(logstd[index[i]:index[i+1], :])
            else:
                mu_total.append(mu[index[i]:, :])
                logstd_total.append(logstd[index[i]:, :])

        self.vgae.__mu__, self.vgae.__logstd__= mu, logstd

        #domain_prior=self.prior_model(domain_emb)



        new_batch=[]
        new_index=[]
        size=0


        for i in range(len(mu_total)):
            decode_size=mu_total[i].shape[0]

            decode_edge_index=torch.ones([decode_size,decode_size]).nonzero().t().contiguous().cuda()

            embs=self.vgae.reparametrize(mu_total[i],logstd_total[i].clamp(max=10))

            j=np.random.choice(list(range(mu_total[i].shape[0])))


            embs_temp=mu_total[i]
            embs_temp[j, :]=embs[j]
            embs=embs_temp
            #embs[j, :]=mu_total[i][j]


            #edge_prob = self.vgae.decode(embs, decode_edge_index, sigmoid=True).view(-1,1)
            edge_prob=self.decoder(embs, decode_edge_index, domain_embs=None, sigmoid=True).view(-1,1)



            #edge_prob = self.vgae.decode(z, decode_edge_index, sigmoid=True).view(-1,1)
            # logits = torch.log(edge_prob)

            if np.random.random()<0.0001 and self.vgae.training==False:
                pass
                #print(edge_prob.view(-1))

            logits = torch.cat([torch.log(1-edge_prob+1e-9),torch.log(edge_prob+1e-9)],dim=1)

            mask = F.gumbel_softmax(logits=logits,hard=True).bool()[:,1] # only take the 2nd col as the logits of edge_prob
            # mask = self.gumbel_sigmoid(logits, tau=1) # n_edges x 1

            row, col = decode_edge_index

            row, col = torch.masked_select(row, mask), torch.masked_select(col,mask)
            # row, col = row[mask], col[mask]
            new_edge_index = torch.stack([row, col], dim=0)

            xs=self.vgae.reparametrize(mu_total[i],logstd_total[i].clamp(max=10))
            embs_temp=mu_total[i]
            embs_temp[j, :]=xs[j]
            xs=embs_temp
            #xs=embs



            new_batch.append(Data(xs, new_edge_index))
            #print(new_edge_index.shape)


            new_index.append(size)
            size += embs.shape[0]

        new_batch = Batch().from_data_list(new_batch)

        #print(subgraph_edge_index.shape)
        #print(gen_edge_index.shape)
        #print(1/0)

        #print(gen_edge_index)


        # only replace the subgraph in the original graph


        # combine
        #new_edge_index=other_edge_index
        #new_edge_index = torch.cat([gen_subgraph_edge_index,other_edge_index],dim=1)

        return new_batch, new_index, 0#self.vgae.kl_loss()#self.vgae.recon_loss(z, subgraph_edge_index)#+self.vgae.kl_loss(mu=domain_prior[0], logstd=domain_prior[1])

    def random_subgraph(self, edge_index, n, K):
        # sample a subgraph with K nodes
        # randomly sample a K nodes


        node_idxes = np.arange(n)
        np.random.shuffle(node_idxes)
        sampled_node_idxes = node_idxes[:K]
        other_node_idxes = np.array([i for i in range(n) if i not in sampled_node_idxes])
        other_node_idxes = torch.LongTensor(other_node_idxes).to(self.device)
        sampled_node_idxes = torch.LongTensor(sampled_node_idxes).to(self.device)

        # print(edge_index)

        subgraph_edge_index, _ = subgraph(sampled_node_idxes, edge_index)

        subgraph_edge_index = subgraph_edge_index

        #print("subgraph is created")
        # subgraph_x = x[sampled_node_idxes,:]
        # subgraph_y = y[sampled_node_idxes,:]

        #print("subgraph edge index shape", subgraph_edge_index.shape)

        return subgraph_edge_index, sampled_node_idxes, other_node_idxes

    def adjust_edge(self, idx, neighbors):
        # Generate edges for subgraphs
        dic = {}
        for i in range(len(idx)):
            dic[idx[i]] = i

        new_index = [[], []]
        nodes = set(idx)
        for i in idx:
            edge = list(neighbors[i] & nodes)
            edge = [dic[_] for _ in edge]
            # edge = [_ for _ in edge if _ > i]
            new_index[0] += len(edge) * [dic[i]]
            new_index[1] += edge
        return torch.LongTensor(new_index).cuda()

    def forward(self, data, criterion=None, sample_idx=None, K=3000):

        # load data to GPU
        x, y = data.graph['node_feat'].to(self.device), data.label.to(self.device)

        #print(y)


        edge_index = data.graph['edge_index'].to(self.device)

        x=self.gnn(x,edge_index,return_hidden=True)



        n = x.shape[0]

        if sample_idx is None:
            if self.args.dataset!='elliptic' and self.args.dataset!='ogb-arxiv':
                sample_idx=np.random.choice(list(range(n)), size=500 if n>500 else n, replace=False)
            elif self.args.dataset=='ogb-arxiv':
                sample_idx = np.random.choice(list(range(n)), size=1000, replace=False)
                #sample_idx = np.random.choice(np.where(data.test_mask == True)[0], size=500, replace=False)
            else:
                sample_idx = np.random.choice(np.where(data.mask==True)[0], size=100, replace=False)



        size = 0
        index=[]
        subgraphs = []
        for idx in sample_idx:




            nodes = [idx] + np.random.choice(list(data.graph['neighbors'][idx]),
                                             size=min(len(data.graph['neighbors'][idx]), self.max_size),
                                             replace=False).tolist()
            sub_edges = self.adjust_edge(nodes, data.graph['neighbors'])
            index.append(size)

            x_ori=x[nodes]


            xs=[x_ori.mean(0)]
            for node_idx in nodes[1:]:
                nodes = [node_idx]+np.random.choice(list(data.graph['neighbors'][node_idx]), size=min(len(data.graph['neighbors'][node_idx]),self.max_size), replace=False).tolist()
                xs.append(x[nodes].mean(0) if len(x[nodes].shape)==2 else x[nodes])
            xs=torch.stack(xs,0)


            sub_x = torch.cat([x_ori,xs],-1)
            subgraphs.append(Data(sub_x, sub_edges))
            size += sub_x.size(0)

        batch=Batch().from_data_list(subgraphs)



        new_batch, new_index, loss_recon = self.edge_gen(x, edge_index, n, K, batch, index)



        out=self.batch_gnn(new_batch.x, new_batch.edge_index, new_batch.batch, new_index, return_hidden=False)

        #if self.args.dataset=='elliptic':
        #    out=out[:,0]

        if criterion:

            loss = self.sup_loss(y[sample_idx], out, criterion)
            loss+=loss_recon

            return loss

        else:
            return out


    def inference_no_aug(self, data):
        x = data.graph['node_feat'].to(self.device)
        edge_index = data.graph['edge_index'].to(self.device)
        out = self.gnn(x, edge_index)
        return out

    def sup_loss(self, y, pred, criterion):
        if self.args.rocauc or self.args.dataset in ('twitch-e', 'fb100', 'elliptic'):
            if y.shape[1] == 1:
                true_label = F.one_hot(y, max(2, y.max() + 1)).squeeze(1)
                #print(true_label)

            else:
                true_label = y

            loss = criterion(pred, true_label.squeeze(1).to(torch.float))
        else:
            out = F.log_softmax(pred, dim=1)
            target = y.squeeze(1)
            loss = criterion(out, target)
        return loss