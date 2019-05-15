"""Data related package."""
from __future__ import absolute_import

from .drug_data_loader import DrugDataset
from .eth_data_loader import EthDataset
from .simple_data_loader import SimpleDataset
from .wechat_data_loader import WeChatDataset

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
    elif dataset == 'wechat':
        return WeChatDataset()
    else:
        raise ValueError('Unknown dataset: {}'.format(dataset))
