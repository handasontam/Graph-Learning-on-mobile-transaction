Dependencies
------------
- torch v1.0: the autograd support for sparse mm is only available in v1.0.
- requests
- sklearn
- tensorflow (for tensorboard)
```bash
$ pip install -r requirements.txt
```

```bash
$ pip install torch==1.0.0 requests
$ pip install dgl
$ pip install tensorboardX
```
<!-- 
JSON configuration file:

| parameter     | description                         | type    |
|---------------|-------------------------------------|---------|
| gpu           | which GPU to use. Set -1 to use CPU | int     |
| epochs        | number of training epochs           | int     |
| num_heads     | number of hidden attention heads    | int     |
| num_out_heads | number of output attention heads    | int     |
| num_layers    | number of hidden layers             | int     |
| num_hidden    | number of hidden units              | int     |
| residual      | use residual connection             | int     |
| in_drop       | input feature dropout               | float   |
| attn_drop     | attention dropout                   | float   |
| lr            | learning rate                       | float   |
| weight_decay  | weight decay                        | float   |
| alpha         | the negative slop of leaky relu     | float   |
| fastmod       | skip re-evaluate the validation set | boolean | -->



