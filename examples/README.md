Dependencies
------------
- torch v1.0: the autograd support for sparse mm is only available in v1.0.
- requests
- sklearn

```bash
pip install torch==1.0.0 requests
pip install dgl
```

```bash
# download eth dataset
mkdir data
cd data
curl https://transfer.sh/c9ZKx/eth.tar.gz -o eth.tar.gz  # md5: 674f5875c8d2271fcd5f36607194762e
tar -zxvf eth.tar.gz
```



| model           | node attr.     | edge attr.  |  Result (max test accuracy |
| -------------   |:--------------:| -----------:| -------------: |
| Edge prop + GAT | yes            |  yes        | 0.9261|
| Edge prop + GAT | no            |  yes        | 0.6774|
| Edge prop + GIN (originally used for graph isomorphism testing) | yes            |  yes        | |
| Edge prop + SAGE| yes            |  yes        | |
| GAT Edge attention | yes         |  yes(only used for attention)        | 0.8602|
| GAT Edge attention | no         |  yes(only used for attention)        | 0.4370|
| GAT             | yes            |  no         | 0.8302|
| Deep Graph Infomax| yes            |  no         | |

