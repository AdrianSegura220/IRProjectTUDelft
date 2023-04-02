import numpy as np
import pandas as pd

from fast_forward.ranking import Ranking


# implementation of reciprocal rank fusion
# query: query id
# document: document id
# ranks: a list of lists of ranks, since each query has a list of documents and each document has a semantic score and a lexical score
# eta: a smoothing parameter, by default 60
def reciprocal_rank_fusion(ranks,eta=60):
    # returns the reciprocal rank of a document for a query
    return 1/sum([1/rank+eta for rank in ranks])

def reciprocal_rank_fusion_all(r1: Ranking, r2: Ranking,eta=60,isNorm = False,rankingName = 'rrf',sort=True):
    for q_id in r1.q_ids:
        # Get the ranks of the documents from both rankings
        r1_ranks = {doc_id: rank + 1 for rank, doc_id in enumerate(r1[q_id].keys())}
        r2_ranks = {doc_id: rank + 1 for rank, doc_id in enumerate(r2[q_id].keys())}

        # min and max 
        minScore, maxScore = np.inf(), -np.inf()
        fusion = {}
        # Calculate the RRF score for each document that appears in both rankings
        for doc_id in set(r1_ranks.keys()) & set(r2_ranks.keys()):
            ranks = r1_ranks[doc_id]+r2_ranks[doc_id]
            fusion[q_id] = {}
            fusion[q_id][doc_id] = reciprocal_rank_fusion(ranks,eta)

            if fusion[q_id][doc_id] > maxScore:
                maxScore = fusion[q_id][doc_id]
            
            if fusion[q_id][doc_id] < minScore:
                minScore = fusion[q_id][doc_id]
    # returns a dictionary with the reciprocal rank of each document for each query
    return Ranking(min_max_normalization(minScore,maxScore,fusion) if isNorm else fusion, name=rankingName, sort=sort, copy=False)

def min_max_normalization(minScore,maxScore,ranks):
    # then, we normalize the ranks
    for q_id in ranks.keys():
        for doc_id in ranks[q_id].keys():
            ranks[q_id][doc_id] = (ranks[q_id][doc_id] - minScore)/(maxScore - minScore)
    
    return ranks