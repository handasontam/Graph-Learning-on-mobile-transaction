![equation](./equation.png)

## with node features
```bash
python train.py \
--gpu=0 \
--epochs 10000 \
--num-heads 2 \
--num-hidden 5 \
--in-drop 0 \
--attn-drop 0 \
--lr 0.0015 \
--edges_path ../data/eth/adj.csv \
--node_features_path ../data/eth/node_features.csv \
--label_path ../data/eth/label.csv \
--vertex_map_path ../data/eth/node_id_map.txt
```

## without node features

```bash
python train.py \
--gpu=1 \
--epochs 10000 \
--num-heads 2 \
--num-hidden 5 \
--in-drop 0 \
--attn-drop 0 \
--lr 0.0015 \
--edges_path ../data/eth/adj.csv \
--label_path ../data/eth/label.csv \
--vertex_map_path ../data/eth/node_id_map.txt
```


# Resutls
learning both edge embedding and node embeddings  
## node features + edge features
max test accuracy = 0.9261 over 2000 epochs
## edge features (dummy node features: [1,1,1,1,1,1,1,1,1,1] for all nodes)
max test accuracy = 0.5914 over 2000 epochs