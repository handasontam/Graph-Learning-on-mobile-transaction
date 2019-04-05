import argparse
import torch
import torch.nn.functional as F
import sys
import os
import logging
from dgl import DGLGraph
from models import EdgePropGAT, GAT, GAT_EdgeAT
from trainer import Trainer
from data import register_data_args, load_data
from utils import Params, set_logger

# logger = logging.getLogger(__name__)
# logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s') # include timestamp

def main(params):
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
    else:
        cuda = True
        torch.cuda.set_device(params.gpu)
        features = features.cuda()
        data.graph.edata['e'] = data.graph.edata['e'].cuda()
        labels = labels.cuda()
        train_mask = train_mask.cuda()
        val_mask = val_mask.cuda()
        test_mask = test_mask.cuda()

    # create DGL graph
    g = data.graph
    n_edges = g.number_of_edges()
    # add self loop
    g.add_edges(g.nodes(), g.nodes())
    # create model
    heads = ([params.num_heads] * params.num_layers) + [params.num_out_heads]
    if params.model == "EdgePropAT":
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
                    params.residual)
    elif params.model == "GAT":
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

    logging.info(model)
    if cuda:
        model.cuda()
    loss_fcn = torch.nn.CrossEntropyLoss()

    # use optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=params.lr, weight_decay=params.weight_decay)

    trainer = Trainer(model, loss_fcn, optimizer, params.epochs, features, 
                    labels, train_mask, val_mask, test_mask, params.fastmode, n_edges)
    trainer.train()

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Examples')
    register_data_args(parser)
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

    # models asssertions
    current_models = {'EdgePropAT', 'GAT', 'GAT_EdgeAT'}
    assert params.model in current_models, "The model \"{}\" is not implemented, please chose from {}".format(params.model, current_models)


    # params.cuda = torch.cuda.is_available()

    main(params)
