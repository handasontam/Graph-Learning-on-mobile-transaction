"""Data related package."""
from __future__ import absolute_import

from .drug_data_loader import DrugDataset
from .eth_data_loader import EthDataset
from .simple_data_loader import SimpleDataset

def register_data_args(parser):
    parser.add_argument("--dataset", type=str, required=True,
            help="The input dataset.")

def load_data(dataset):
    if dataset == 'eth':
        return EthDataset()
    elif dataset == 'drug':
        return DrugDataset()
    elif dataset == 'simple':
        return SimpleDataset()
    else:
        raise ValueError('Unknown dataset: {}'.format(dataset))
