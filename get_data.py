"""
Get paper data from Crossref -> pandas DataFrame

Notes:
- Crossref max page size is 1000, so we page with offset if rows > page_size.
- For polite usage, pass your email via `mailto`.
"""

from typing import List, Optional, Dict, Any
import re
import requests

import pandas as pd

from util import date_parts_to_datetime


CROSSREF_WORKS_URL = "https://api.crossref.org/works"


def _escape_query_term(term: str) -> str:
    """
    Escape a term for a simple Crossref 'query' string.
    Wrap multi-word phrases in quotes.
    """
    term = term.strip()
    if not term:
        return ""
    # Avoid double quotes breaking the query
    term = term.replace('"', '\\"')
    if " " in term:
        return f'"{term}"'
    return term


def _build_query_from_keywords(keywords: List[str]) -> str:
    """
    Build a single query string from a list of keyword terms.
    Example: ["indoor environmental quality", "IEQ"] -> '"indoor environmental quality" OR IEQ'
    """
    cleaned = [_escape_query_term(k) for k in keywords if k and k.strip()]
    if not cleaned:
        raise ValueError("keywords must contain at least one non-empty string.")
    return " OR ".join(cleaned)


def _build_filter(types: List[str], from_date: str) -> str:
    """
    Build the Crossref 'filter' parameter.

    Crossref filter is AND across different keys.
    Repeating the same key (e.g. type:...) is treated as OR.
    """
    # parts = [f"from-index-date:{from_date}"]
    parts = [f"from-created-date:{from_date}"]
    for t in types:
        t = (t or "").strip()
        if t:
            parts.append(f"type:{t}")
    return ",".join(parts)


def _build_filter_for_orcid(orcid: str, types: List[str], from_date: str) -> str:
    """
    Build Crossref filter for searching works by ORCID within a date range + types.
    Crossref supports: filter=orcid:<id>,from-created-date:<YYYY-MM-DD>,type:<...>
    """
    parts = [f"orcid:{orcid}", f"from-created-date:{from_date}"]
    for t in types:
        t = (t or "").strip()
        if t:
            parts.append(f"type:{t}")
    return ",".join(parts)


def _clean_abstract(text: Optional[str]) -> Optional[str]:
    """
    Clean Crossref abstract.
    Crossref abstracts are often JATS/HTML-like.
    This removes tags and collapses whitespace.
    """
    if not text:
        return None
    # Remove XML/HTML tags
    text = re.sub(r"<[^>]+>", " ", text)
    # Normalize whitespace
    text = re.sub(r"\s+", " ", text).strip()
    return text


def _flatten_crossref_item(item: Dict[str, Any]) -> Dict[str, Any]:
    """
    Flatten a Crossref item into a compact row suitable for analysis.
    Keep the raw item as well for advanced downstream use.
    """
    title_list = item.get("title") or []
    title = title_list[0] if title_list else None

    container_list = item.get("container-title") or []
    container_title = container_list[0] if container_list else None

    doi = item.get("DOI")
    cr_type = item.get("type")

    indexed_dt = (item.get("indexed") or {}).get("date-time")
    created_dt = (item.get("created") or {}).get("date-time")
    published_ol_parts = (item.get("published-online") or {}).get("date-parts")
    published_online_dt = date_parts_to_datetime(published_ol_parts)
    issued_parts = (item.get("issued") or {}).get("date-parts")
    issued_dt = date_parts_to_datetime(issued_parts)

    # Authors: list of dicts {given, family, ORCID, ...}
    authors = item.get("author") or []
    author_names = []
    orcids = []
    for a in authors:
        given = (a.get("given") or "").strip()
        family = (a.get("family") or "").strip()
        name = " ".join([p for p in [given, family] if p])
        if name:
            author_names.append(name)
        if a.get("ORCID"):
            orcids.append(a["ORCID"])

    # Abstract if available
    abstract_raw = item.get("abstract")
    abstract = _clean_abstract(abstract_raw)

    row = {
        "doi": doi,
        "type": cr_type,
        "title": title,
        "container_title": container_title,
        "publisher": item.get("publisher"),
        "indexed_date": indexed_dt,
        "created_date": created_dt,
        "published_online_date": published_online_dt,
        "issued_date": issued_dt,
        "author_count": len(authors),
        "authors": "; ".join(author_names) if author_names else None,
        "orcids": "; ".join(orcids) if orcids else None,
        "url": item.get("URL"),
        "score": item.get("score"),  # relevance score (when using `query`)
        "abstract": abstract,  # optional content if available
        "raw": item,  # keep full JSON item for later analysis
    }
    return row


def get_data(
    keywords: List[str],
    types: Optional[List[str]] = None,
    from_date: Optional[str] = None,
    rows: int = 20,
    mailto: Optional[str] = None,
    sort: str = "score",
    order: str = "desc",
    timeout: int = 30,
    page_size: int = 200,
) -> pd.DataFrame:
    """
    Fetch Crossref works and return a pandas DataFrame.

    Notes
    -----
    from_date uses Crossref filter `from-created-date`.

    Parameters
    ----------
    keywords:
        List of keyword terms. Will be combined with OR into the Crossref `query` parameter.
    types:
        List of Crossref work types. Default: journal-article + proceedings-article.
        See https://api.crossref.org/types
        Examples: "journal-article", "proceedings-article", "posted-content", "report",
            "book-chapter", "database", "standard", "dissertation", "dataset"
    from_date:
        Start date for `from-created-date` filter in YYYY-MM-DD.
        If None, no date filter is applied (not recommended for large queries).
    rows:
        Total number of records to fetch (not per page). Default 20.
    mailto:
        Contact email for polite usage (recommended).
    sort:
        Sorting controls. "score" (relevance) / "indexed" / "created" / "updated" / "published" etc.
    order:
        Sorting controls. "asc" or "desc".
    timeout:
        Request timeout.
    page_size:
        Number of records per request (Crossref allows up to 1000).

    Returns
    -------
    pd.DataFrame
    """
    if rows <= 0:
        return pd.DataFrame()

    if types is None:
        types = ["journal-article", "proceedings-article"]

    query = _build_query_from_keywords(keywords)

    params_base: Dict[str, Any] = {
        "query": query,
        "sort": sort,
        "order": order,
    }

    # Include email address for polite requests with higher rate limit
    if mailto:
        params_base["mailto"] = mailto

    if from_date:
        params_base["filter"] = _build_filter(types=types, from_date=from_date)
    else:
        # If user does not provide from_date, still apply type filters if any.
        # (Otherwise large queries can be very slow / huge.)
        if types:
            params_base["filter"] = ",".join([f"type:{t}" for t in types if t])

    # Pagination via offset
    collected: List[Dict[str, Any]] = []
    offset = 0
    remaining = rows
    page_size = max(1, min(int(page_size), 1000))

    headers = {
        # Optional but good practice; Crossref also accepts mailto param
        "User-Agent": (
            f"weekly-paper-report (mailto:{mailto})"
            if mailto
            else "weekly-paper-report"
        ),
        "Accept": "application/json",
    }

    while remaining > 0:
        batch = min(page_size, remaining)
        params = dict(params_base)
        params["rows"] = batch
        params["offset"] = offset

        r = requests.get(
            CROSSREF_WORKS_URL, params=params, headers=headers, timeout=timeout
        )
        r.raise_for_status()
        payload = r.json()

        items = payload.get("message", {}).get("items", []) or []
        if not items:
            break

        for it in items:
            collected.append(_flatten_crossref_item(it))

        got = len(items)
        remaining -= got
        offset += got

        # If server returns fewer than requested, we reached the end.
        if got < batch:
            break

    df = pd.DataFrame(collected)

    # Convenience: parse timestamps where possible
    for col in [
        "indexed_date",
        "created_date",
        "published_online_date",
        "issued_date",
    ]:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors="coerce", utc=True)

    # De-duplicate by DOI (Crossref DOI is case-insensitive; normalize to lower for stable dedup)
    if "doi" in df.columns:
        df["doi"] = df["doi"].astype(str).str.strip()
        df["doi_norm"] = df["doi"].str.lower()
        df = df.loc[~df["doi_norm"].duplicated()].copy()

    return df


def get_data_by_orcid(
    orcid: str,
    types: Optional[List[str]] = None,
    from_date: Optional[str] = None,
    rows: int = 20,
    mailto: Optional[str] = None,
    sort: str = "created",
    order: str = "desc",
    timeout: int = 30,
    page_size: int = 200,
) -> pd.DataFrame:
    """
    Fetch Crossref works for a given ORCID and return a pandas DataFrame.

    Notes
    -----
    Results only include works where publishers deposited this ORCID in Crossref metadata.

    from_date uses Crossref filter `from-created-date`.

     Parameters
    ----------
    orcid:
        ORCID of the author, from https://orcid.org/.
    types:
        List of Crossref work types. Default: journal-article + proceedings-article.
        See https://api.crossref.org/types
        Examples: "journal-article", "proceedings-article", "posted-content", "report",
            "book-chapter", "database", "standard", "dissertation", "dataset"
    from_date:
        Start date for `from-created-date` filter in YYYY-MM-DD.
        If None, no date filter is applied (not recommended for large queries).
    rows:
        Total number of records to fetch (not per page). Default 20.
    mailto:
        Contact email for polite usage (recommended).
    sort:
        Sorting controls. "score" (relevance) / "indexed" / "created" / "updated" / "published" etc.
    order:
        Sorting controls. "asc" or "desc".
    timeout:
        Request timeout.
    page_size:
        Number of records per request (Crossref allows up to 1000).

    Returns
    -------
    pd.DataFrame
    """
    if not orcid or not isinstance(orcid, str):
        return pd.DataFrame()

    orcid = orcid.strip()
    orcid = orcid.replace("https://orcid.org/", "").replace("http://orcid.org/", "")

    if rows <= 0:
        return pd.DataFrame()

    if types is None:
        types = ["journal-article", "proceedings-article"]

    params_base: Dict[str, Any] = {
        "sort": sort,
        "order": order,
    }

    if mailto:
        params_base["mailto"] = mailto

    if from_date:
        params_base["filter"] = _build_filter_for_orcid(
            orcid=orcid, types=types, from_date=from_date
        )
    else:
        # Still apply orcid + types if no from_date
        parts = [f"orcid:{orcid}"] + [f"type:{t}" for t in types if t]
        params_base["filter"] = ",".join(parts)

    collected: List[Dict[str, Any]] = []
    offset = 0
    remaining = rows
    page_size = max(1, min(int(page_size), 1000))

    headers = {
        "User-Agent": (
            f"weekly-paper-report (mailto:{mailto})"
            if mailto
            else "weekly-paper-report"
        ),
        "Accept": "application/json",
    }

    while remaining > 0:
        batch = min(page_size, remaining)
        params = dict(params_base)
        params["rows"] = batch
        params["offset"] = offset

        r = requests.get(
            CROSSREF_WORKS_URL, params=params, headers=headers, timeout=timeout
        )
        r.raise_for_status()
        payload = r.json()

        items = payload.get("message", {}).get("items", []) or []
        if not items:
            break

        for it in items:
            collected.append(_flatten_crossref_item(it))

        got = len(items)
        remaining -= got
        offset += got
        if got < batch:
            break

    df = pd.DataFrame(collected)

    # Parse timestamps
    for col in [
        "indexed_date",
        "created_date",
        "published_online_date",
        "issued_date_date",
    ]:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors="coerce", utc=True)

    # Dedup DOI
    if "doi" in df.columns:
        df["doi"] = df["doi"].astype(str).str.strip()
        df["doi_norm"] = df["doi"].str.lower()
        df = df.loc[~df["doi_norm"].duplicated()].copy()

    return df
