# Run experiment
- First, create a new experiment directory
```bash
mkdir -p experiments/test
```
- create the params.json (see the following section (copy one of the below and modify))
```bash
vim experiments/test/params.json
```

- prepare the dataset (adj.txt, features,txt, label.txt), the format for each of the file is listed below (Also see data/test_data for the example):

## adj.txt:
```
from_node_id, to_node_id, edge_features1, edge_features2, ...
```

## features.txt
```
node_id, feature1, feature2, ...
```

## label.txt
```
node_id, label_id
```

- start the experiment using the following command
```bash
python main.py --data-dir ./dmt/data/test_data/ --preprocess --model-dir experiments/test
```

# params.json
## General parameter
- gpu (int): The gpu id. gpu should be set to -1 if you only want to use cpu
- epochs (int): The max number of epochs.
- lr (float): learning rate
- patience (int): patience for early stopping, if the loss stops decreasing for consecutive (patience) epochs, it will early stop.
- weight_decay (float): an L2 regularization. Higher value will result in stronger regularization
- use_batch_norm (bool): Use batch normalization
- in_drop (float): dropout probability for the input features
- model (str): The model name. (See below for details)

## Additional parameter for minibatch model
- batch_size (int): 
- test_batch_size (int):
- num_neighbors (int): number of samples used for neighbor sampling
- num_cpu: number of cpu used for minibatch sampling and neighbor sampling

## Model
### DGI
- conv_model can be 'GCN' or 'GAT'

```json
{
    "model": "DGI", 
    "gpu": -1,
    "conv_model": "GCN", 
    "epochs": 10000, 
    "num_layers": 1, 
    "node_hidden_dim": 128,
    "in_drop": 0.0, 
    "lr": 0.0002, 
    "weight_decay": 0.01,
    "fastmode": false,
    "patience": 100,
    "num_cpu": 10
}
```

### MiniBatchGraphSAGE
```json
{
    "model": "MiniBatchGraphSAGE",
    "gpu": 0,
    "epochs": 10000,
    "num_layers": 1,
    "num_hidden": 128,
    "residual": true,
    "in_drop": 0.0,
    "lr": 0.0003,
    "batch_size": 32,
    "test_batch_size": 512,
    "num_neighbors": 30,
    "weight_decay": 0.001,
    "use_batch_norm": false,
    "fastmode": false,
    "patience": 100,
    "num_cpu": 10
}
```


### MiniBatchEdgeProp
```json
{
    "model": "MiniBatchEdgeProp", 
    "gpu": 0,
    "epochs": 10000, 
    "num_layers": 1, 
    "num_hidden": 5, 
    "in_drop": 0.0, 
    "lr": 0.0002, 
    "batch_size": 128, 
    "test_batch_size": 512, 
    "num_neighbors": 5, 
    "weight_decay": 0.01,
    "fastmode": false,
    "patience": 100,
    "num_cpu": 10
}
```

### MiniBatchDGI
- Currently, conv_model only supports EdgeProp
```json
{
    "model": "MiniBatchDGI", 
    "gpu": 0,
    "conv_model": "EdgeProp", 
    "epochs": 10000, 
    "num_layers": 1, 
    "node_hidden_dim": 3,
    "in_drop": 0.0, 
    "lr": 0.0003,
    "batch_size": 8192, 
    "test_batch_size": 8192,  
    "num_neighbors": 25, 
    "weight_decay": 0.01,
    "fastmode": false,
    "patience": 10,
    "num_cpu": 10
}
```
