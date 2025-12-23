"""
Clustering analysis using TF-IDF + K-Means
"""

from typing import Any, Tuple, Iterable

import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.sparse import csr_matrix
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score


def vectorize_titles_tfidf(
    df: pd.DataFrame,
    *,
    max_features: int = 800,
    ngram_range: tuple[int, int] = (1, 2),
    min_df: int = 2,
    stop_words: str | Any = "english",
) -> Tuple[csr_matrix, TfidfVectorizer]:
    """
    Vectorize paper titles using TF-IDF.

    Parameters
    ----------
    df:
        Cleaned DataFrame containing a 'title' column.
    max_features:
        Maximum number of TF-IDF features to keep.
    ngram_range:
        N-gram range. (1, 2) is recommended for short texts like titles.
    min_df:
        Ignore terms that appear in fewer than `min_df` documents.
        Helps reduce noise from rare words.
    stop_words:
        Stop-word list. Use "english" or custom stop words.

    Returns
    -------
    X:
        TF-IDF feature matrix (documents × features).
    vectorizer:
        Fitted TfidfVectorizer instance.
    """
    if "title" not in df.columns:
        raise ValueError("DataFrame must contain a 'title' column.")

    titles = df["title"].astype(str).str.strip().fillna("").tolist()

    vectorizer = TfidfVectorizer(
        max_features=max_features,
        ngram_range=ngram_range,
        min_df=min_df,
        stop_words=stop_words,
    )

    X = vectorizer.fit_transform(titles)

    return X, vectorizer


def choose_k_with_silhouette(
    X,
    k_candidates: Iterable[int] = range(3, 8),
    random_state: int = 42,
    prefer_smaller_k: bool = True,
    delta: float = 0.02,
) -> Tuple[int, dict]:
    """
    Choose K for KMeans within a semantically reasonable range using silhouette score.

    Parameters
    ----------
    X:
        Feature matrix (e.g., TF-IDF).
    k_candidates:
        Iterable of candidate k values (e.g., range(3, 8)).
    random_state:
        Random state for KMeans.
    prefer_smaller_k:
        If True, prefer smaller k when silhouette scores are very close.
    delta:
        Threshold below which silhouette score differences are considered negligible.

    Returns
    -------
    best_k:
        Selected number of clusters.
    scores:
        Dict mapping k -> silhouette score.
    """
    scores = {}

    for k in k_candidates:
        if k >= X.shape[0]:
            continue  # cannot have more clusters than samples

        labels = KMeans(
            n_clusters=k,
            random_state=random_state,
            n_init="auto",
        ).fit_predict(X)

        score = silhouette_score(X, labels)
        scores[k] = score

    if not scores:
        raise ValueError("No valid k candidates for silhouette evaluation.")

    # Sort by silhouette score (descending)
    sorted_items = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    best_k, best_score = sorted_items[0]

    if prefer_smaller_k:
        # Find all k whose score is close to the best score
        close_ks = [k for k, s in sorted_items if abs(s - best_score) <= delta]
        best_k = min(close_ks)

    return best_k, scores


def top_terms_by_cluster(df, X, vectorizer, top_n=10):
    terms = vectorizer.get_feature_names_out()
    labels = df["cluster"].to_numpy()
    out = {}

    for c in sorted(np.unique(labels)):
        idx = np.where(labels == c)[0]
        mean_tfidf = X[idx].mean(axis=0).A1
        top_idx = mean_tfidf.argsort()[::-1][:top_n]
        out[int(c)] = [terms[i] for i in top_idx]

    return out
