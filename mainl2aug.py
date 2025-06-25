import os

os.environ['CUDA_LAUNCH_BLOCKING'] = "1"

import torch
os.environ['TORCH'] = torch.__version__
print(torch.__version__)

import argparse
import sys
import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.utils import to_undirected
from torch_scatter import scatter
import time
from logger import Logger, SimpleLogger
from dataset import load_nc_dataset
from data_utils import normalize, gen_normalized_adjs, evaluate, evaluate_whole_graph, evaluate_whole_graph_ogb, evaluate_whole_graph_multi, evaluate_whole_graph_elliptic, eval_acc, eval_rocauc, eval_f1, to_sparse_tensor, load_fixed_splits
from parse import parser_add_main_args, parse_method_l2aug_multi

import warnings
warnings.filterwarnings("ignore")

from copy import deepcopy

from utils_mp import Subgraph


### Parse args ###
parser = argparse.ArgumentParser(description='General Training Pipeline')
parser_add_main_args(parser)
# args = parser.parse_args()
args, unknown = parser.parse_known_args()
print(args)

device = torch.device("cuda:" + str(args.device)) if torch.cuda.is_available() else torch.device("cpu")
print(device)
#device=torch.device("cpu")



# NOTE: for consistent data splits, see data_utils.rand_train_test_idx

def fix_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    #torch.backends.cudnn.deterministic = True


def get_dataset(dataset, sub_dataset=None, gen_model=None, year=None):
    ### Load and preprocess data ###
    if dataset == 'twitch-e':
        dataset = load_nc_dataset(args.data_dir, 'twitch-e', sub_dataset)
    elif dataset == 'fb100':
        dataset = load_nc_dataset(args.data_dir, 'fb100', sub_dataset)
    elif dataset == 'cora':
        dataset = load_nc_dataset(args.data_dir, 'cora', sub_dataset, gen_model)
    elif dataset == 'amazon-photo':
        dataset = load_nc_dataset(args.data_dir, 'amazon-photo', sub_dataset, gen_model)
    elif dataset == 'ogb-arxiv':
        dataset = load_nc_dataset(args.data_dir, 'ogb-arxiv', year=year)
    elif dataset == 'elliptic':
        dataset = load_nc_dataset(args.data_dir, 'elliptic', sub_dataset)
    else:
        raise ValueError('Invalid dataname')

    if len(dataset.label.shape) == 1:
        dataset.label = dataset.label.unsqueeze(1)

    dataset.n = dataset.graph['num_nodes']
    dataset.c = max(dataset.label.max().item() + 1, dataset.label.shape[1])
    dataset.d = dataset.graph['node_feat'].shape[1]

    dataset.graph['edge_index'], dataset.graph['node_feat'] = \
        dataset.graph['edge_index'], dataset.graph['node_feat']

    neighbors=[set() for _ in range(dataset.n)]
    #for i in range(dataset.n):
    #    neighbors[i].add(i)

    for edge_index in dataset.graph['edge_index'].t().numpy():
        neighbors[edge_index[0]].add(edge_index[1])

    dataset.graph['neighbors']=neighbors

    return dataset

def main():
    fix_seed(0)


    if args.dataset == 'twitch-e':
        twitch_sub_name = ['DE', 'ENGB', 'ES', 'FR', 'RU', 'TW']
        tr_subs, val_subs, te_subs = ['DE'], ['ENGB'], ['ES','FR','PTBR','RU', 'TW']
        datasets_tr = [get_dataset(dataset='twitch-e', sub_dataset=tr_subs[i]) for i in range(len(tr_subs))]
        datasets_val = [get_dataset(dataset='twitch-e', sub_dataset=val_subs[i]) for i in range(len(val_subs))]
        datasets_te = [get_dataset(dataset='twitch-e', sub_dataset=te_subs[i]) for i in range(len(te_subs))]
    
    elif args.dataset == 'fb100':
        '''
        Configure different training sub-graphs
        '''
        tr_subs, val_subs, te_subs = ['Johns Hopkins55', 'Caltech36', 'Amherst41'], ['Cornell5', 'Yale4'],  ['Penn94', 'Brown11', 'Texas80']
        # tr_subs, val_subs, te_subs = ['Bingham82', 'Duke14', 'Princeton12'], ['Cornell5', 'Yale4'],  ['Penn94', 'Brown11', 'Texas80']
        # tr_subs, val_subs, te_subs = ['WashU32', 'Brandeis99', 'Carnegie49'], ['Cornell5', 'Yale4'], ['Penn94', 'Brown11', 'Texas80']
        datasets_tr = [get_dataset(dataset='fb100', sub_dataset=tr_subs[i]) for i in range(len(tr_subs))]
        datasets_val = [get_dataset(dataset='fb100', sub_dataset=val_subs[i]) for i in range(len(val_subs))]
        datasets_te = [get_dataset(dataset='fb100', sub_dataset=te_subs[i]) for i in range(len(te_subs))]
    elif args.dataset == 'cora':
        tr_sub, val_sub, te_subs = [0], [1], list(range(2, 10))
        gen_model = args.gnn_gen
        datasets_tr = [get_dataset(dataset='cora', sub_dataset=tr_sub[0], gen_model=gen_model)]
        datasets_val = [get_dataset(dataset='cora', sub_dataset=val_sub[0], gen_model=gen_model)]
        datasets_te = [get_dataset(dataset='cora', sub_dataset=te_subs[i], gen_model=gen_model) for i in range(len(te_subs))]
    elif args.dataset == 'amazon-photo':
        tr_sub, val_sub, te_subs = [0], [1], list(range(2, 10))
        gen_model = args.gnn_gen
        datasets_tr = [get_dataset(dataset='amazon-photo', sub_dataset=tr_sub[0], gen_model=gen_model)]
        datasets_val = [get_dataset(dataset='amazon-photo', sub_dataset=val_sub[0], gen_model=gen_model)]
        datasets_te = [get_dataset(dataset='amazon-photo', sub_dataset=te_subs[i], gen_model=gen_model) for i in range(len(te_subs))]
    elif args.dataset == 'ogb-arxiv':
        tr_year, val_year, te_years = [[1950, 2011]], [[2011, 2014]], [[2014, 2016], [2016, 2018], [2018, 2020]]
        datasets_tr = [get_dataset(dataset='ogb-arxiv', year=tr_year[0])]
        datasets_val = [get_dataset(dataset='ogb-arxiv', year=val_year[0])]
        datasets_te = [get_dataset(dataset='ogb-arxiv', year=te_years[i]) for i in range(len(te_years))]
    elif args.dataset == 'elliptic':
        tr_subs, val_subs, te_subs = [i for i in range(6, 11)], [i for i in range(11, 16)], [i for i in range(16, 49)]
        datasets_tr = [get_dataset(dataset='elliptic', sub_dataset=tr_subs[i]) for i in range(len(tr_subs))]
        datasets_val = [get_dataset(dataset='elliptic', sub_dataset=val_subs[i]) for i in range(len(val_subs))]
        datasets_te = [get_dataset(dataset='elliptic', sub_dataset=te_subs[i]) for i in range(len(te_subs))]
    else:
        raise ValueError('Invalid dataname')

    # if args.dataset == 'fb100':
    for i in range(len(datasets_tr)):
        dataset_tr = datasets_tr[i]
        print(f"Train num nodes {dataset_tr.n} | num classes {dataset_tr.c} | num node feats {dataset_tr.d}")

    dataset_val = datasets_val[0]
    # print(f"Train num nodes {dataset_tr.n} | num classes {dataset_tr.c} | num node feats {dataset_tr.d}")
    print(f"Val num nodes {dataset_val.n} | num classes {dataset_val.c} | num node feats {dataset_val.d}")

    for i in range(len(datasets_te)):
        dataset_te = datasets_te[i]
        print(f"Test {i} num nodes {dataset_te.n} | num classes {dataset_te.c} | num node feats {dataset_te.d}")


    model = parse_method_l2aug_multi(args, datasets_tr, device)


    # using rocauc as the eval function
    if args.dataset=='twitch-e' or args.dataset=='fb100' or args.dataset=='elliptic':
        criterion = nn.BCEWithLogitsLoss()
    else:
        criterion= nn.NLLLoss()

    if args.dataset == 'fb100' or args.dataset=='cora' or args.dataset=='amazon-photo' or args.dataset=='ogb-arxiv':
        eval_func = eval_acc
    elif args.dataset=='elliptic':
        eval_func = eval_f1
    else:
        eval_func = eval_rocauc

    logger = Logger(args.runs, args)

    # model.train()
    print("Method:", args.method)
    #print('MODEL:', model)
    print('DATASET:', args.dataset)

    ### Subgraph Extraction ###
    ppr_path = './subgraph/' + args.dataset
    subgraphs_tr = []
    for i in range(len(datasets_tr)):

        data = datasets_tr[i].graph
        #subgraph = Subgraph(data.x, data.edge_index, ppr_path, args.subgraph_size, args.n_order)
        subgraph = Subgraph(data['node_feat'], data['edge_index'], ppr_path, args.subgraph_size, args.n_order)
        subgraph.build()
        subgraphs_tr.append(subgraph)

    ### Training loop ###

    # for run in range(args.runs):
    for run in range(1):
        # we do not run multiple times here
        # model.reset_parameters()
        # model = deepcopy(model_)
        output_file=open('./results/{}.txt'.format(time.time()),'w')


        model.train()


        optimizer = torch.optim.AdamW(model.parameters(),
                                        lr=args.lr, weight_decay=args.weight_decay)
        
        best_val = float('-inf')
        for epoch in range(args.epochs):
            model.train()


            # optimizer_gnn.zero_grad()
            # optimizer_aug.zero_grad()
            optimizer.zero_grad()
            if args.dataset == 'twitch-e' or args.dataset=='cora' or args.dataset=='amazon-photo' or args.dataset=='ogb-arxiv' or args.dataset=='elliptic':
                for i in range(len(datasets_tr)):
                    dataset_tr = datasets_tr[i]
                    loss = model(dataset_tr, criterion)
                    loss.backward()

                    #if epoch % args.display_step == 0:
                    #    print('loss:', loss)

                    optimizer.step()
            elif args.dataset == 'fb100':
                for dataset_tr in datasets_tr:
                    loss = model(dataset_tr, criterion)
                    loss.backward()
                    optimizer.step()



            if epoch % args.display_step == 0:

                if args.dataset == 'twitch-e' or args.dataset=='cora' or args.dataset=='amazon-photo' :
                #if epoch>1000:
                    accs, test_outs = evaluate_whole_graph(args, model, dataset_tr, dataset_val, datasets_te, eval_func)
                elif args.dataset=='ogb-arxiv':
                    accs, test_outs = evaluate_whole_graph_ogb(args, model, dataset_tr, dataset_val, datasets_te, eval_func)
                elif args.dataset == 'fb100':
                    accs, test_outs = evaluate_whole_graph_multi(args, model, datasets_tr, datasets_val, datasets_te, eval_func)
                elif args.dataset=='elliptic':
                    accs, test_outs = evaluate_whole_graph_elliptic(args, model, datasets_tr, datasets_val, datasets_te,
                                                                 eval_func)

                logger.add_result(run, accs)




                test_info = ''
                for test_acc in accs[2:]:
                    test_info += f'Test: {100 * test_acc:.2f}% '
                #print(test_info)
                print(f'Epoch: {epoch:02d}, '
                      f'Train: {100 * accs[0]:.2f}%, '
                      f'Valid: {100 * accs[1]:.2f}%, '+test_info)

                output_file.write(f'Epoch: {epoch:02d}, '
                                       f'Train: {100 * accs[0]:.2f}%, '
                                       f'Valid: {100 * accs[1]:.2f}%, '+test_info+'\n')




        logger.print_statistics(run)

main()