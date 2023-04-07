import logging
from collections import defaultdict
from ranking import Ranking
import numpy as np

LOGGER = logging.getLogger(__name__)

def z_score_normalisation(ranking: Ranking) -> Ranking:
    """Normalize the scores in a ranking using z-score normalization.

    Args:
        ranking (Ranking): The input ranking.

    Returns:
        Ranking: The normalized ranking.
    """
    normalized = defaultdict(dict)
    print("z-score normalization done")
    for q_id in ranking:
        mean = np.mean(list(ranking[q_id].values()))
        std = np.std(list(ranking[q_id].values()))

        if std == 0:
            for doc_id in ranking[q_id]:
                normalized[q_id][doc_id] = 1  # or any other fixed value, as they're all the same
        else:
            for doc_id in ranking[q_id]:
                normalized[q_id][doc_id] = (ranking[q_id][doc_id] - mean) / std
          
    return Ranking(normalized, name=ranking.name, sort=True, copy=False)

def normalize_ranking(ranking: Ranking) -> Ranking:
    """Normalize the scores in a ranking using minimax normalization.

    Args:
        ranking (Ranking): The input ranking.

    Returns:
        Ranking: The normalized ranking.
    """
    normalized = defaultdict(dict)
    print("minimax normalization done")
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
    r1: Ranking, r2: Ranking, alpha: float, name: str = None, sort: bool = True, normalise: str = "none"
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

    if normalise == "minimax":
        r1 = normalize_ranking(r1)
        r2 = normalize_ranking(r2)
    elif normalise == "zscore":
        r1 = z_score_normalisation(r1)
        r2 = z_score_normalisation(r2)

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

def reciprocal_rank_fusion(r1: Ranking, r2: Ranking, name: str = None, sort: bool = True, eta=60, eta2 = 0) -> Ranking:
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

    if eta2 == 0:
        eta2 = eta

    for q_id in r1:
        r1_ranks = r1.__getitem__(q_id)
        r2_ranks = r2.__getitem__(q_id)

        r1_ranks = {k: v for k, v in sorted(r1_ranks.items(), key=lambda item: item[1], reverse=True)}
        r2_ranks = {k: v for k, v in sorted(r2_ranks.items(), key=lambda item: item[1], reverse=True)}

        r1_indices = {doc_id: idx for idx, doc_id in enumerate(r1_ranks.keys())}
        r2_indices = {doc_id: idx for idx, doc_id in enumerate(r2_ranks.keys())}

        common_doc_ids = set(r1_ranks.keys()) & set(r2_ranks.keys())
        fused_run[q_id] = {doc_id: (1 / (eta + r1_indices[doc_id])) + (1 / (eta2 + r2_indices[doc_id])) for doc_id in common_doc_ids}

    # Create a new Ranking instance with the fused rankings
    fused_ranking = Ranking(fused_run, name=name, sort=sort, copy=False)
    return fused_ranking