import logging
from collections import defaultdict
from ranking import Ranking
import numpy as np


LOGGER = logging.getLogger(__name__)

def normalize_ranking(ranking: Ranking) -> Ranking:
    """Normalize the scores in a ranking using minimax normalization.

    Args:
        ranking (Ranking): The input ranking.

    Returns:
        Ranking: The normalized ranking.
    """
    normalized = defaultdict(dict)
    for q_id in ranking:
        min_score = min(ranking[q_id].values())
        max_score = max(ranking[q_id].values())

        if max_score == min_score:
            for doc_id in ranking[q_id]:
                normalized[q_id][doc_id] = 1  # or any other fixed value, as they're all the same
        else:
            for doc_id in ranking[q_id]:
                normalized[q_id][doc_id] = (
                    (ranking[q_id][doc_id] - min_score) / (max_score - min_score)
                )

    return Ranking(normalized, name=ranking.name, sort=True, copy=False)

def interpolate(
    r1: Ranking, r2: Ranking, alpha: float, name: str = None, sort: bool = True, normalise: bool = False
) -> Ranking:
    """Interpolate scores. For each query-doc pair:
        * If the pair has only one score, ignore it.
        * If the pair has two scores, interpolate: r1 * alpha + r2 * (1 - alpha).

    Args:
        r1 (Ranking): Scores from the first retriever.
        r2 (Ranking): Scores from the second retriever.
        alpha (float): Interpolation weight.
        name (str, optional): Ranking name. Defaults to None.
        sort (bool, optional): Whether to sort the documents by score. Defaults to True.

    Returns:
        Ranking: Interpolated ranking.
    """

    if normalise:
        r1 = normalize_ranking(r1)
        r2 = normalize_ranking(r2)

    assert r1.q_ids == r2.q_ids
    # print name of the two rankings
    print(r1.name, r2.name)
    results = defaultdict(dict)
    for q_id in r1:
        for doc_id in r1[q_id].keys() & r2[q_id].keys():
            results[q_id][doc_id] = (
                alpha * r1[q_id][doc_id] + (1 - alpha) * r2[q_id][doc_id]
            )
    
    
    return Ranking(results, name=name, sort=sort, copy=False)

def reciprocal_rank_fusion(r1: Ranking, r2: Ranking, name: str = None, sort: bool = True, normalise: bool = False, eta=60, eta2 = 0) -> Ranking:
    """RRF For each query-doc pair:
        * If the pair has only one document, ignore it.
        * If the pair has two documents, then do rrf on both ranks

    Args:
        r1 (Ranking): Scores from the first retriever.
        r2 (Ranking): Scores from the second retriever.
        alpha (float): Interpolation weight.
        name (str, optional): Ranking name. Defaults to None.
        sort (bool, optional): Whether to sort the documents by score. Defaults to True.

    Returns:
        Ranking: RRF ranking.
    """

    if normalise:
        r1 = normalize_ranking(r1)
        r2 = normalize_ranking(r2)

    if r1.q_ids != r2.q_ids:
        raise ValueError("Ranking instances must have the same query IDs.")
    
    fused_run = defaultdict(dict)

    if eta2 == 0:
        eta2 = eta
    
    for q_id in r1:
        # Get the document rankings for each query in both ranking instances
        r1_ranks = r1.__getitem__(q_id)
        r2_ranks = r2.__getitem__(q_id)

        # sort the ranks based on score
        r1_ranks = {k: v for k, v in sorted(r1_ranks.items(), key=lambda item: item[1], reverse=True)}
        r2_ranks = {k: v for k, v in sorted(r2_ranks.items(), key=lambda item: item[1], reverse=True)}

        # Calculate the RRF score for each document that appears in both rankings
        for doc_id in r1_ranks.keys() & r2_ranks.keys():
            # get the index of the doc_id
            value1 = list(r1_ranks.keys()).index(doc_id)
            value2 = list(r2_ranks.keys()).index(doc_id)
            rrf_score = (1 / (eta + value1)) + (1 / (eta2 + value2))
            fused_run[q_id][doc_id] = rrf_score

    # Create a new Ranking instance with the fused rankings
    fused_ranking = Ranking(fused_run, name=name, sort=sort, copy=False)
    return fused_ranking

def reciprocal_rank_fusion_2(ranks,eta=60):
    # returns the reciprocal rank of a document for a query
    return sum([1/(rank+eta) for rank in ranks])

def reciprocal_rank_fusion_all(r1: Ranking, r2: Ranking,eta=60,isNorm = False,rankingName = 'rrf',sort=True):
    for q_id in r1.q_ids:
        # Get the ranks of the documents from both rankings
        r1_ranks = {doc_id: rank + 1 for rank, doc_id in enumerate(r1[q_id].keys())}
        r2_ranks = {doc_id: rank + 1 for rank, doc_id in enumerate(r2[q_id].keys())}
        
        # min and max 
        minScore, maxScore = np.Inf, -np.Inf
        fusion = {}
        # Calculate the RRF score for each document that appears in both rankings
        for doc_id in set(r1_ranks.keys()) & set(r2_ranks.keys()):
            ranks = [r1_ranks[doc_id],r2_ranks[doc_id]]
            fusion[q_id] = {}
            fusion[q_id][doc_id] = reciprocal_rank_fusion_2(ranks,eta)

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