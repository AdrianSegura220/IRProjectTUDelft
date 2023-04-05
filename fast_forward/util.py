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

def reciprocal_rank_fusion(r1: Ranking, r2: Ranking, name: str = None, sort: bool = True, normalise: bool = False) -> Ranking:
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
    k = 60

    # print the names of the two rankings
    print(r1.name, r2.name)

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
            rrf_score = (1 / (k + value1)) + (1 / (k + value2))
            fused_run[q_id][doc_id] = rrf_score

    # Create a new Ranking instance with the fused rankings
    fused_ranking = Ranking(fused_run, name=name, sort=sort, copy=False)
    return fused_ranking


def sigmoid(x, beta=1):
    return 1 / (1 + np.exp(-beta*x))

def sigmoid_rank_fusion(r1: Ranking, r2: Ranking, name: str = None, sort: bool = True) -> Ranking:
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

    if r1.q_ids != r2.q_ids:
        raise ValueError("Ranking instances must have the same query IDs.")
    
    fused_run = defaultdict(dict)
    k = 60

    # print the names of the two rankings
    print(r1.name, r2.name)

    beta = 1

    for q_id in r1:
        # Get the document rankings for each query in both ranking instances
        r1_ranks = r1.__getitem__(q_id)
        r2_ranks = r2.__getitem__(q_id)

        # sort the ranks based on score
        r1_ranks = {k: v for k, v in sorted(r1_ranks.items(), key=lambda item: item[1], reverse=True)}
        r2_ranks = {k: v for k, v in sorted(r2_ranks.items(), key=lambda item: item[1], reverse=True)}

        # Calculate the RRF score for each document that appears in both rankings
        for doc_id in r1_ranks.keys() & r2_ranks.keys():
            #get score of doc_id in r1
            score1 = r1_ranks[doc_id]
            # sum over all the ranks for r1_ranks
            sum_r1 = 0.5
            for i in range(len(r1_ranks)):
                sum_r1 += sigmoid(x=r1_ranks[i]-score1, beta=beta)

            #get score of doc_id in r2
            score2 = r2_ranks[doc_id]
            # sum over all the ranks for r2_ranks
            sum_r2 = 0.5
            for i in range(len(r2_ranks)):
                sum_r2 += sigmoid(x=r2_ranks[i]-score2, beta=beta)
            
            # fuse sum_r1 and sum_r2 with k
            fused_score = (1 / (k + sum_r1)) + (1 / (k + sum_r2))
            fused_run[q_id][doc_id] = fused_score

    # Create a new Ranking instance with the fused rankings
    fused_ranking = Ranking(fused_run, name=name, sort=sort, copy=False)
    return fused_ranking