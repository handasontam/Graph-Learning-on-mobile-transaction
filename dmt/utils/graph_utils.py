import networkx as nx
from networkx.relabel import relabel_nodes

def convert_node_labels_to_integers(G, first_label=0):
    """Return a copy of the graph G with the nodes relabeled using
    consecutive integers.

    Parameters
    ----------
    G : graph
       A NetworkX graph

    first_label : int, optional (default=0)
       An integer specifying the starting offset in numbering nodes.
       The new integer labels are numbered first_label, ..., n-1+first_label.

    label_attribute : string, optional (default=None)
       Name of node attribute to store old label.  If None no attribute
       is created.

    Notes
    -----
    Node and edge attribute data are copied to the new (relabeled) graph.

    See Also
    --------
    relabel_nodes
    """
    N = G.number_of_nodes()+first_label
    mapping = dict(zip(G.nodes(), range(first_label, N)))
    H = relabel_nodes(G, mapping)
    H.name = "("+G.name+")_with_int_labels"
    return H

def get_vertex_mapping_dict(nodes, first_label=0):
    N = len(nodes)+first_label
    mapping = dict(zip(list(nodes), range(first_label, N)))
    return mapping