import argparse
import torch
import torch.nn.functional as F
import sys
import os
import logging
from dgl import DGLGraph
from dmt.models import EdgePropGAT, GAT, GAT_EdgeAT, MiniBatchEdgeProp, MiniBatchEdgePropInfer, MiniBatchGCNInfer, MiniBatchGCNSampling, MiniBatchGraphSAGEInfer, MiniBatchGraphSAGESampling
from dmt.models import MiniBatchEdgePropPlus, MiniBatchEdgePropPlusInfer
from dmt.trainer import Trainer
from dmt.mini_batch_trainer import MiniBatchTrainer
from dmt.data import register_data_args, load_data
from dmt.utils import Params, set_logger

# logger = logging.getLogger(__name__)
# logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s') # include timestamp

def main(parserrams):
    # load and preprocess dataset
    data = load_data(params.dataset)
    features = torch.FloatTensor(data.features)
    labels = torch.LongTensor(data.labels)
    train_mask = torch.ByteTensor(data.train_mask)
    val_mask = torch.ByteTensor(data.val_mask)
    test_mask = torch.ByteTensor(data.test_mask)
    num_feats = features.shape[1]
    # num_edge_feats = edge_features.shape[1]
    num_edge_feats = data.num_edge_feats
    n_classes = data.num_labels
    n_edges = data.graph.number_of_edges()
    n_nodes = data.graph.number_of_nodes()
    logging.info("""----Data statistics------'
      #Edges %d
      #Classes %d
      #Train samples %d
      #Val samples %d
      #Test samples %d""" %
          (n_edges, n_classes,
           train_mask.sum().item(),
           val_mask.sum().item(),
           test_mask.sum().item()))
    if params.gpu < 0:
        cuda = False
        cuda_context = None
    else:
        cuda = True
        torch.cuda.set_device(params.gpu)
        cuda_context = torch.device('cuda:{}'.format(params.gpu))
        # features = features.cuda()
        labels = labels.cuda()
        train_mask = train_mask.cuda()
        val_mask = val_mask.cuda()
        test_mask = test_mask.cuda()

    # create DGL graph
    g = data.graph
    n_edges = g.number_of_edges()
    # add self loop
    # print(g.edata)
    #g.add_edges(g.nodes(), g.nodes(), data={'edge_features': torch.zeros((n_nodes, num_edge_feats))})

    # create model
    if params.model == "EdgePropAT":
        heads = [params.num_heads] * params.num_layers
        model = EdgePropGAT(g,
                    params.num_layers,
                    num_feats,
                    num_edge_feats, 
                    params.num_hidden,
                    n_classes,
                    heads,
                    F.elu,
                    params.in_drop,
                    params.attn_drop,
                    params.alpha,
                    params.residual, 
                    params.use_batch_norm)
        
    elif params.model == "GAT":
        heads = [params.num_heads] * params.num_layers
        model = GAT(g,
                    params.num_layers,
                    num_feats,
                    params.num_hidden,
                    n_classes,
                    heads,
                    F.elu,
                    params.in_drop,
                    params.attn_drop,
                    params.alpha,
                    params.residual)
        
    elif params.model == "GAT_EdgeAT":
        heads = [params.num_heads] * params.num_layers
        model = GAT_EdgeAT(g,
                    params.num_layers,
                    num_feats,
                    num_edge_feats, 
                    params.num_hidden,
                    n_classes,
                    heads,
                    F.elu,
                    params.in_drop,
                    params.attn_drop,
                    params.alpha,
                    params.residual)
        
    elif params.model == "MiniBatchEdgeProp":
        model = MiniBatchEdgeProp(
                    g, 
                    params.num_layers,
                    num_feats,
                    num_edge_feats, 
                    params.num_hidden,
                    n_classes,
                    F.elu,
                    params.in_drop,
                    params.residual, 
                    params.use_batch_norm)
        model_infer = MiniBatchEdgePropInfer(
                    g, 
                    params.num_layers,
                    num_feats,
                    num_edge_feats, 
                    params.num_hidden,
                    n_classes,
                    F.elu,
                    0, #params.in_drop,
                    params.residual, 
                    params.use_batch_norm)
        
    elif params.model == 'MiniBatchGCN':
        model = MiniBatchGCNSampling(
                    in_feats=num_feats,
                    n_hidden=params.num_hidden,
                    n_classes=n_classes,
                    n_layers=params.num_layers,
                    activation=F.relu,
                    dropout=params.in_drop
        )
        model_infer = MiniBatchGCNInfer(
                    in_feats=num_feats,
                    n_hidden=params.num_hidden,
                    n_classes=n_classes,
                    n_layers=params.num_layers,
                    activation=F.relu
        )
        
    elif params.model == 'MiniBatchGraphSAGE':
        model = MiniBatchGraphSAGESampling(
                    in_feats=num_feats,
                    n_hidden=params.num_hidden,
                    n_classes=n_classes,
                    n_layers=params.num_layers,
                    activation=F.relu,
                    dropout=params.in_drop
        )
        model_infer = MiniBatchGraphSAGEInfer(
                    in_feats=num_feats,
                    n_hidden=params.num_hidden,
                    n_classes=n_classes,
                    n_layers=params.num_layers,
                    activation=F.relu
        )
        
    elif params.model == 'MiniBatchEdgePropPlus':
        model = MiniBatchEdgePropPlus(
                    g, 
                    params.num_layers,
                    num_feats,
                    num_edge_feats, 
                    params.node_hidden_dim,
                    params.edge_hidden_dim,
                    params.fc_hidden_dim,
                    n_classes,
                    F.elu,
                    params.in_drop,
                    params.residual, 
                    params.use_batch_norm)        

        model_infer = MiniBatchEdgePropPlusInfer(
                    g, 
                    params.num_layers,
                    num_feats,
                    num_edge_feats, 
                    params.node_hidden_dim,
                    params.edge_hidden_dim,
                    params.fc_hidden_dim,
                    n_classes,
                    F.elu,
                    0, #params.in_drop,
                    params.residual, 
                    params.use_batch_norm)


    if cuda:
        model.cuda()   
        if 'model_infer' in locals():
            model_infer.cuda()

    logging.info(model)
    loss_fcn = torch.nn.CrossEntropyLoss()

    # use optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=params.lr, weight_decay=params.weight_decay)

    if "minibatch" in params.model.lower():
        g.readonly()
        # initialize the history for control variate
        # see control variate in https://arxiv.org/abs/1710.10568
        for i in range(params.num_layers):
            g.ndata['history_{}'.format(i)] = torch.zeros((features.shape[0], params.node_hidden_dim))
        g.ndata['node_features'] = features
        #g.edata['edge_features'] = data.graph.edata['edge_features']
        norm = 1./g.in_degrees().unsqueeze(1).float()
        g.ndata['norm'] = norm
        print('graph node features', g.ndata['node_features'].shape)
        print('graph edge features', g.edata['edge_features'].shape)

        degs = g.in_degrees().numpy()
        degs[degs > params.num_neighbors] = params.num_neighbors
        g.ndata['subg_norm'] = torch.FloatTensor(1./degs).unsqueeze(1)  # for calculating P_hat

        trainer = MiniBatchTrainer(
                        g=g, 
                        model=model, 
                        model_infer=model_infer,
                        loss_fn=loss_fcn, 
                        optimizer=optimizer, 
                        epochs=params.epochs, 
                        features=features, 
                        labels=labels, 
                        train_mask=train_mask, 
                        val_mask=val_mask, 
                        test_mask=test_mask, 
                        fast_mode=params.fastmode, 
                        n_edges=n_edges, 
                        patience=params.patience, 
                        batch_size=params.batch_size, 
                        test_batch_size=params.test_batch_size, 
                        num_neighbors=params.num_neighbors, 
                        n_layers=params.num_layers, 
                        model_dir=params.model_dir, 
                        num_cpu=params.num_cpu, 
                        cuda_context=cuda_context)
    else:
        if cuda:
            g.edata['edge_features'] = data.graph.edata['edge_features'].cuda()
        trainer = Trainer(
                        model=model, 
                        loss_fn=loss_fcn, 
                        optimizer=optimizer, 
                        epochs=params.epochs, 
                        features=features, 
                        labels=labels, 
                        train_mask=train_mask, 
                        val_mask=val_mask, 
                        test_mask=test_mask, 
                        fast_mode=params.fast_mode, 
                        n_edges=n_edges, 
                        patience=params.patience, 
                        model_dir=params.model_dir)
    trainer.train()

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Examples')
    # register_data_args(parser)
    parser.add_argument("--model-dir", type=str, required=True, 
                        help="Directory containing params.json")
    parser.add_argument('--restore_file', default=None,
                        help="Optional, name of the file in --model_dir containing weights to reload before \
                        training")  # 'best' or 'train'
    args = parser.parse_args()


    # Set the logger
    set_logger(os.path.join(args.model_dir, 'train.log'))
    logging.info(args)

    # load params
    json_path = os.path.join(args.model_dir, 'params.json')
    assert os.path.isfile(json_path), "No json configuration file found at {}".format(json_path)
    params = Params(json_path)

    params.model_dir = args.model_dir

    # models asssertions
    current_models = {'EdgePropAT', 'GAT', 'GAT_EdgeAT', 'MiniBatchEdgeProp', 'MiniBatchGCN', 'MiniBatchGraphSAGE', 'MiniBatchEdgePropPlus'}
    assert params.model in current_models, "The model \"{}\" is not implemented, please chose from {}".format(params.model, current_models)


    # params.cuda = torch.cuda.is_available()

    main(params)
