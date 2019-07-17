import numpy as np
import dgl

def get_graph_from_data(data_path, directed, edge_dim):
    """
    Process the data into networkx DiGraph and run the algorithm
    :param data_path:
    :param directed:
    :return:
    """
    # process the data
    with open(data_path, 'r') as f:
        data = f.readlines()
        dg = dgl.DGLGraph()
        i=0
        for edges in data:
            s = edges.strip().split(',')
            if len(s) < 2:
                continue
            u = int(s[0])
            v = int(s[1])
            edge_fts = np.array([x for x in s[2:]]).astype(np.float32)
            if directed:
                if dg.has_edge(u, v):
                    dg[u][v]['edge_features'][0:edge_dim] = edge_fts
                else:
                    dg.add_edges_from([(u, v, {'edge_features': np.append(edge_fts, np.zeros(edge_dim))})])
                # dg.add_edges_from([(int(u), int(v), {'edge_features': weight})])
                if dg.has_edge(v, u):
                    dg[v][u]['edge_features'][edge_dim:] = edge_fts
                else:
                    dg.add_edges_from([(v, u, {'edge_features': np.append(np.zeros(edge_dim), edge_fts)})])
            else:
                # for undirected graph, we add the reverse path for each edge
                dg.add_edges_from([(u, v, {'edge_features': edge_fts})])
                dg.add_edges_from([(v, u, {'edge_features': edge_fts})])
            i+=1
    # print(len(list(DG.selfloop_edges())))
    # remove self loop
    # DG.remove_edges_from(DG.selfloop_edges())
    #print('original graph: ', G.edges(data=True))
    print('Graph loaded success')
    print('number of vertex: ', dg.number_of_nodes())
    print('number of edges: ', dg.number_of_edges())
    print()

    return dg