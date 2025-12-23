"""
Weekly Paper Report

- Author: hqrrr
- License: MIT
- Homepage: https://github.com/hqrrr/weekly-paper-report
"""

import os
from datetime import date
from dotenv import load_dotenv

from get_data import get_data
from util import clean_df, last_n_days_iso, load_keywords, i18n
from analysis import cluster_analysis, publisher_analysis, followed_authors_analysis
from report import report_html, PlotEmbed

# Configuration

## Which type of papers to search
TYPES = ["journal-article", "proceedings-article"]

## Leave an email for polite query
## Use 'Secrets' in your GitHub repository
## Settings -> Secrets and variables -> Actions -> New repository secret
## Add:
##    Name: WPR_MAILTO
##    Secret: your.email@email.com
load_dotenv()  # reads variables from a .env file for local development
EMAIL = os.getenv("WPR_MAILTO", "")

## Keywords and Followed Authors
KEYWORDS_PATH = "./config/keywords.yaml"
FOLLOWED_AUTHORS_PATH = "./config/followed_authors.yaml"

## Number of retrievals
ROWS_KEYWORD_SEARCH = 100  # rows
ROWS_PER_AUTHOR = 20  # rows

## Search date
DAYS_BACK = 7  # days

## Report theme
THEME = "light"

if __name__ == "__main__":
    today = date.today().isoformat()
    print(f"::group::Weekly Paper Report Log ({today})")
    print("=== weekly-paper-report ===")
    # Environment check
    if EMAIL:
        print("WPR_MAILTO loaded OK.")
    else:
        print("WPR_MAILTO not set (mailto will be omitted).")

    # Load keywords from YAML
    keywords = load_keywords(KEYWORDS_PATH)
    print("\nKeywords:")
    print(keywords)

    # Compute rolling date range
    from_date = last_n_days_iso(DAYS_BACK)

    # Keyword search
    df = get_data(
        keywords=keywords,
        types=TYPES,
        from_date=from_date,
        rows=ROWS_KEYWORD_SEARCH,
        mailto=EMAIL,
    )
    # Clean data
    df_cleaned = clean_df(df)

    # Cluster analysis
    df_clustered, cluster_terms, k = cluster_analysis(df_cleaned)

    # Publisher analysis
    publisher_analysis(df_cleaned)

    # Followed authors analysis
    (df_followed_in_results, df_followed_recent_by_orcid) = followed_authors_analysis(
        df_clustered,
        types=TYPES,
        from_date=from_date,
        rows_per_author=ROWS_PER_AUTHOR,
        mailto=EMAIL,
        followed_authors_path=FOLLOWED_AUTHORS_PATH,
    )

    # Report
    report_path = report_html(
        df_results=df_clustered,
        df_followed_in_results=df_followed_in_results,
        df_followed_recent_by_orcid=df_followed_recent_by_orcid,
        plots=[
            PlotEmbed(i18n("Cluster map", "聚类图"), "./html/clusters.html"),
            PlotEmbed(
                i18n("Publishers and article types", "出版商与文章类型"),
                "./html/publishers.html",
            ),
        ],
        report_title=i18n("Weekly Paper Report", "每周论文报告"),
        subtitle=f"{today}",
        top_picks_n=10,
        cluster_top_picks_n=5,
        theme=THEME,
        keywords=keywords,
        followed_authors_path=FOLLOWED_AUTHORS_PATH,
    )

    print(f"\n=== Report generated at {report_path} ===")
    print("::endgroup::")
    print("::notice::Report successfully generated")
