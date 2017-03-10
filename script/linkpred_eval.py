import pandas as pd
import math
import numpy as np

# minden napról csak az első topK db predikciót tartja meg
def get_topK_predicts(predicted_links, topK):
    if (topK == 0):
        return predicted_links
    
    links_per_day = len(predicted_links[predicted_links['day'] == 1])
    if (links_per_day < topK):
        return predicted_links
    
    # ez jó
    ndays = len(predicted_links['day'].unique())
    indices = []
    for i in range(0,ndays):
        for j in range(topK,links_per_day):
            indices.append(i*links_per_day + j)

    return predicted_links.drop(indices)

# az eltalált élek halmaza
# ha topK = 0, akkor az egész predicted_links kell, ha nem 0, akkor meg kell nyesni
def get_TP_edges(edges_df, predicted_links, topK = 0):
    predicted_links_topK = get_topK_predicts(predicted_links, topK)
    true_positive_edges = pd.merge(edges_df, predicted_links_topK, on=['day', 'source', 'target'], left_index=True)
    return true_positive_edges
    
# a betippelt nem létező élek
def get_FP_edges(edges_df, predicted_links, topK = 0):
    predicted_links_topK = get_topK_predicts(predicted_links, topK)
    true_positive_edges = pd.merge(edges_df, predicted_links_topK, on=['day', 'source', 'target'], left_index=True)
    false_positive_edges = predicted_links.drop(true_positive_edges.index.values)
    return false_positive_edges

# a nem tippelt, de felbukanó élek
def get_FN_edges(edges_df, predicted_links, topK = 0):
    predicted_links_topK = get_topK_predicts(predicted_links, topK)
    true_positive_edges = pd.merge(edges_df, predicted_links_topK, on=['day', 'source', 'target'], right_index=True)
    false_negative_edges = edges_df.drop(true_positive_edges.index.values)
    return false_negative_edges

def get_precision(edges_df, predicted_links, topK = 0):
    TP = len(get_TP_edges(edges_df, predicted_links, topK))
    FP = len(get_FP_edges(edges_df, predicted_links, topK))
    return TP / (TP + FP)

def get_recall(edges_df, predicted_links, topK = 0):
    TP = len(get_TP_edges(edges_df, predicted_links, topK))
    FN = len(get_FN_edges(edges_df, predicted_links, topK))
    return TP / (TP + FN)

def get_f1score(edges_df, predicted_links, topK = 0):
    prec = get_precision(edges_df, predicted_links, topK)
    rec = get_recall(edges_df, predicted_links, topK)
    return 2*prec*rec / (prec + rec)

# a topK paraméter ne legyen nagyobb, mint a config fájlban
def get_NDCG(edges_df, predicted_links, topK = 0):
    predicted_links_topK = get_topK_predicts(predicted_links, topK)
    ndays = len(predicted_links_topK['day'].unique())
    NDCG = []
    
    links_per_day = len(predicted_links_topK[predicted_links_topK['day'] == 1])
    IDCG = 0.0
    for i in range(1,links_per_day+1):
        IDCG += 1 / math.log2(1+i)
        
    true_positive_edges = get_TP_edges(edges_df, predicted_links_topK)
    for day in range(1,ndays+1):
        DCG_daily = 0.0
        pred = predicted_links_topK[predicted_links_topK['day'] == day]
        i = 0
        for index, link in pred.iterrows():
            i += 1
            if (((true_positive_edges['day'] == link['day']) & (true_positive_edges['source'] == link['source']) & (true_positive_edges['target'] == link['target'])).any()):
                DCG_daily += 1 / math.log2(1+i)
        NDCG.append(DCG_daily/IDCG)

    return np.mean(NDCG)

def get_all_metrics(edges_df, predicted_links, topK = 0):
    prec = get_precision(edges_df, predicted_links, topK = 0)
    rec = get_recall(edges_df, predicted_links, topK = 0)
    f1sco = get_f1score(edges_df, predicted_links, topK = 0)
    NDCG = get_NDCG(edges_df, predicted_links, topK = 0)
    return prec, rec, f1sco, NDCG