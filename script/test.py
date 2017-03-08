import networkx as nx
import matplotlib.pyplot as plt
import pandas as pd
# from scipy import stats
# import numpy as np
# from collections import Counter

# plotting the correlations
def plot_correlations(axis, kendalltau, spearman, title = 'Correlation'):
    """
    dokumentáció
    """
    fig = plt.figure()
    plt.plot(axis, kendalltau, label = 'Kendalltau')
    plt.plot(axis, spearman, label = 'Spearman')
    plt.ylabel('correlation')
    plt.xlabel('number of processed edges [thousand]')
    plt.legend(loc = 'best')
    plt.title(title)
    plt.show()
    
# gives a dataframe from data
def get_data(filepath, nrows = None):
    edges_df = pd.read_csv(filepath, sep=' ', names=["time","source","target"], nrows = nrows)  
    return edges_df

# gives a static, weighted, directed graph from the dataframe
# weights: the number of interactions
def get_graph_from_df(df):
    G = nx.DiGraph()
    for index, edge in df.iterrows():
        if G.has_edge(edge[1], edge[2]):
            G[edge[1]][edge[2]]['weight'] += 1.0
        else:
            G.add_edge(edge[1], edge[2], weight = 1.0)
    return G