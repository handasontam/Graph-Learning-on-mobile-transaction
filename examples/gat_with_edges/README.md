Graph Attention Networks (GAT)
============

- Paper link: [https://arxiv.org/abs/1710.10903](https://arxiv.org/abs/1710.10903)
- Author's code repo (in Tensorflow):
  [https://github.com/PetarV-/GAT](https://github.com/PetarV-/GAT).
- Popular pytorch implementation:
  [https://github.com/Diego999/pyGAT](https://github.com/Diego999/pyGAT).

Dependencies
------------
- torch v1.0: the autograd support for sparse mm is only available in v1.0.
- requests
- sklearn

```bash
pip install torch==1.0.0 requests
```

How to run
----------

Run with following:

```bash

python train.py \
--dataset=cora \
--gpu=1 \
--epochs 10000 \
--num-heads 2 \
--num-hidden 5 \
--in-drop 0 \
--attn-drop 0 \
--lr 0.008 \
--edges_path /home/handason/data/eth/adj.csv \
--node_features_path /home/handason/data/eth/node_features.csv \
--label_path /home/handason/data/eth/label.csv \
--vertex_map_path /home/handason/data/eth/node_id_map.txt 
```

Results
-------

| Dataset | Test Accuracy | Time(s) | Baseline#1 times(s) | Baseline#2 times(s) |
| ------- | ------------- | ------- | ------------------- | ------------------- |
| Cora | 84.0% | 0.0127 | 0.0982 (**7.7x**) | 0.0424 (**3.3x**) |
| Citeseer | 70.7% | 0.0123 | n/a | n/a |
| Pubmed | 78.1% | 0.0302 | n/a | n/a |

* All the accuracy numbers are obtained after 300 epochs.
* The time measures how long it takes to train one epoch.
* All time is measured on EC2 p3.2xlarge instance w/ V100 GPU.
* Baseline#1: [https://github.com/PetarV-/GAT](https://github.com/PetarV-/GAT).
* Baseline#2: [https://github.com/Diego999/pyGAT](https://github.com/Diego999/pyGAT).
