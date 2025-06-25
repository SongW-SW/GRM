from collections import defaultdict
import numpy as np
import torch
import torch.nn.functional as F
import scipy
import scipy.io
from sklearn.preprocessing import label_binarize
from ogb.nodeproppred import NodePropPredDataset


from load_data import load_twitch, load_fb100
from data_utils import rand_train_test_idx, even_quantile_labels, to_sparse_tensor, dataset_drive_url

from torch_geometric.datasets import MixHopSyntheticDataset
from torch_geometric.transforms import NormalizeFeatures
from dgl.data import SBMMixtureDataset

from torch_geometric.datasets import Planetoid, Amazon
from os import path

import pickle as pkl

from torch_sparse import SparseTensor

class NCDataset(object):
    def __init__(self, name):
        """
        based off of ogb NodePropPredDataset
        https://github.com/snap-stanford/ogb/blob/master/ogb/nodeproppred/dataset.py
        Gives torch tensors instead of numpy arrays
            - name (str): name of the dataset
            - root (str): root directory to store the dataset folder
            - meta_dict: dictionary that stores all the meta-information about data. Default is None, 
                    but when something is passed, it uses its information. Useful for debugging for external contributers.
        
        Usage after construction: 
        
        split_idx = dataset.get_idx_split()
        train_idx, valid_idx, test_idx = split_idx["train"], split_idx["valid"], split_idx["test"]
        graph, label = dataset[0]
        
        Where the graph is a dictionary of the following form: 
        dataset.graph = {'edge_index': edge_index,
                         'edge_feat': None,
                         'node_feat': node_feat,
                         'num_nodes': num_nodes}
        For additional documentation, see OGB Library-Agnostic Loader https://ogb.stanford.edu/docs/nodeprop/
        
        """

        self.name = name
        self.graph = {}
        self.label = None

    def __getitem__(self, idx):
        assert idx == 0, 'This dataset has only one graph'
        return self.graph, self.label

    def __len__(self):
        return 1

    def __repr__(self):  
        return '{}({})'.format(self.__class__.__name__, len(self))

def load_nc_dataset(data_dir, dataname, sub_dataname='',  gen_model='gcn',year=2020):
    """ Loader for NCDataset
        Returns NCDataset
    """
    if dataname == 'twitch-e':
        # twitch-explicit graph
        if sub_dataname not in ('DE', 'ENGB', 'ES', 'FR', 'PTBR', 'RU', 'TW'):
            print('Invalid sub_dataname, deferring to DE graph')
            sub_dataname = 'DE'
        dataset = load_twitch_dataset(data_dir, sub_dataname)
    elif dataname == 'fb100':
        if sub_dataname not in ('Penn94', 'Amherst41', 'Cornell5', 'Johns Hopkins55', 
        # 'Reed98', 
        'Caltech36', 
        # 'Berkeley13', 
        'Brown11', 
        # 'Columbia2', 
        'Yale4', 
        # 'Virginia63', 
        'Texas80',
        'Bingham82', 'Duke14', 'Princeton12', 'WashU32', 'Brandeis99', 'Carnegie49'):
            print('Invalid sub_dataname, deferring to Penn94 graph')
            sub_dataname = 'Penn94'
        dataset = load_fb100_dataset(data_dir, sub_dataname)

    elif dataname in  ('cora', 'amazon-photo'):
        dataset = load_synthetic_dataset(data_dir, dataname, sub_dataname, gen_model)
    elif dataname == 'ogb-arxiv':
        dataset = load_ogb_arxiv(data_dir=data_dir, year_bound=year, proportion = 1.0)
    elif dataname == 'elliptic':
        if sub_dataname not in range(0, 49):
            print('Invalid sub_dataname, deferring to graph1')
            sub_dataname = 0
        dataset = load_elliptic_dataset(data_dir, sub_dataname)
    else:
        raise ValueError('Invalid dataname')
    return dataset

def load_twitch_dataset(data_dir, lang):
    assert lang in ('DE', 'ENGB', 'ES', 'FR', 'PTBR', 'RU', 'TW'), 'Invalid dataset'
    A, label, features = load_twitch(data_dir, lang)
    dataset = NCDataset(lang)
    edge_index = torch.tensor(A.nonzero(), dtype=torch.long)
    node_feat = torch.tensor(features, dtype=torch.float)
    num_nodes = node_feat.shape[0]
    dataset.graph = {'edge_index': edge_index,
                     'edge_feat': None,
                     'node_feat': node_feat,
                     'num_nodes': num_nodes}
    dataset.label = torch.tensor(label)
    return dataset


def load_fb100_dataset(data_dir, filename):
    feature_vals_all = np.empty((0, 6))
    for f in ['Penn94', 'Amherst41', 'Cornell5', 'Johns Hopkins55', 
    # 'Reed98', 
    'Caltech36', 
    # 'Berkeley13', 
    'Brown11', 
    # 'Columbia2', 
    'Yale4', 
    # 'Virginia63', 
    'Texas80',
    'Bingham82', 'Duke14', 'Princeton12', 'WashU32', 'Brandeis99', 'Carnegie49']:
        A, metadata = load_fb100(data_dir, f)
        metadata = metadata.astype(np.int)
        feature_vals = np.hstack(
            (np.expand_dims(metadata[:, 0], 1), metadata[:, 2:]))
        feature_vals_all = np.vstack(
            (feature_vals_all, feature_vals)
        )

    A, metadata = load_fb100(data_dir, filename)
    dataset = NCDataset(filename)
    edge_index = torch.tensor(A.nonzero(), dtype=torch.long)
    metadata = metadata.astype(np.int)
    label = metadata[:, 1] - 1  # gender label, -1 means unlabeled

    # make features into one-hot encodings
    feature_vals = np.hstack(
        (np.expand_dims(metadata[:, 0], 1), metadata[:, 2:]))
    features = np.empty((A.shape[0], 0))
    for col in range(feature_vals.shape[1]):
        feat_col = feature_vals[:, col]
        # feat_onehot = label_binarize(feat_col, classes=np.unique(feat_col))
        feat_onehot = label_binarize(feat_col, classes=np.unique(feature_vals_all[:, col]))
        features = np.hstack((features, feat_onehot))

    node_feat = torch.tensor(features, dtype=torch.float)
    num_nodes = metadata.shape[0]
    dataset.graph = {'edge_index': edge_index,
                     'edge_feat': None,
                     'node_feat': node_feat,
                     'num_nodes': num_nodes}
    dataset.label = torch.tensor(label)
    # binary label
    dataset.label = torch.where(dataset.label > 0, 1, 0)
    return dataset

def load_synthetic_dataset(data_dir, name, lang, gen_model='gcn'):
    dataset = NCDataset(lang)

    assert lang in range(0, 10), 'Invalid dataset'

    if name == 'cora':
        node_feat, y = pkl.load(open('{}/Planetoid/cora/gen/{}-{}.pkl'.format(data_dir, lang, gen_model), 'rb'))
        torch_dataset = Planetoid(root='{}/Planetoid'.format(data_dir),
                                  name='cora')
    elif name == 'amazon-photo':
        node_feat, y = pkl.load(open('{}/Amazon/Photo/gen/{}-{}.pkl'.format(data_dir, lang, gen_model), 'rb'))
        torch_dataset = Amazon(root='{}/Amazon'.format(data_dir),
                               name='Photo')
    data = torch_dataset[0]

    edge_index = data.edge_index
    # label = data.y
    label = y
    num_nodes = node_feat.size(0)

    dataset.graph = {'edge_index': edge_index,
                     'node_feat': node_feat,
                     'edge_feat': None,
                     'num_nodes': num_nodes}

    dataset.label = label

    return dataset

def take_second(element):
    return element[1]

def load_ogb_arxiv(data_dir, year_bound = [2018, 2020], proportion = 1.0):
    import ogb.nodeproppred

    dataset = ogb.nodeproppred.NodePropPredDataset(name='ogbn-arxiv', root=data_dir)
    graph = dataset.graph

    node_years = graph['node_year']
    n = node_years.shape[0]
    node_years = node_years.reshape(n)

    d = np.zeros(len(node_years))

    edges = graph['edge_index']
    for i in range(edges.shape[1]):
        if node_years[edges[0][i]] <= year_bound[1] and node_years[edges[1][i]] <= year_bound[1]:
            d[edges[0][i]] += 1
            d[edges[1][i]] += 1

    nodes = []
    for i, year in enumerate(node_years):
        if year <= year_bound[1]:
            nodes.append([i, d[i]])

    nodes.sort(key = take_second, reverse = True)

    nodes = nodes[: int(proportion * len(nodes))]

    result_edges = []
    result_features = []
    result_labels = []

    for node in nodes:
        result_features.append(graph['node_feat'][node[0]])
    result_features = np.array(result_features)

    ids = {}
    for i, node in enumerate(nodes):
        ids[node[0]] = i

    for i in range(edges.shape[1]):
        if edges[0][i] in ids and edges[1][i] in ids:
            result_edges.append([ids[edges[0][i]], ids[edges[1][i]]])
    result_edges = np.array(result_edges).transpose(1, 0)

    result_labels = dataset.labels[[node[0] for node in nodes]]

    edge_index = torch.tensor(result_edges, dtype=torch.long)
    node_feat = torch.tensor(result_features, dtype=torch.float)
    dataset.graph = {'edge_index': edge_index,
                     'edge_feat': None,
                     'node_feat': node_feat,
                     'num_nodes': node_feat.size(0)}
    dataset.label = torch.tensor(result_labels)
    node_years_new = [node_years[node[0]] for node in nodes]
    dataset.test_mask = (torch.tensor(node_years_new) > year_bound[0])

    return dataset
def load_elliptic_dataset(data_dir, lang):
    assert lang in range(0, 49), 'Invalid dataset'
    result = pkl.load(open('{}/elliptic/{}.pkl'.format(data_dir, lang), 'rb'))
    A, label, features = result
    dataset = NCDataset(lang)
    edge_index = torch.tensor(A.nonzero(), dtype=torch.long)
    node_feat = torch.tensor(features, dtype=torch.float)
    num_nodes = node_feat.shape[0]
    dataset.graph = {'edge_index': edge_index,
                     'edge_feat': None,
                     'node_feat': node_feat,
                     'num_nodes': num_nodes}
    dataset.label = torch.tensor(label)
    dataset.label= torch.relu(dataset.label)

    dataset.mask = (dataset.label >= 0)
    return dataset