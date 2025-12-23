# Weekly Paper Report

## Overview

<!-- TOC -->
* [Weekly Paper Report](#weekly-paper-report)
  * [Overview](#overview)
  * [Key features](#key-features)
  * [Demo](#demo)
  * [How to use](#how-to-use)
    * [1) Fork & enable GitHub Actions](#1-fork--enable-github-actions)
    * [2) Configure GitHub Pages](#2-configure-github-pages)
    * [3) Configure your keywords and followed authors](#3-configure-your-keywords-and-followed-authors)
      * [Example: `keywords.yaml`](#example-keywordsyaml)
      * [Example: `followed_authors.yaml`](#example-followed_authorsyaml)
    * [4) (Recommended) Set `WPR_MAILTO` as a GitHub Actions secret](#4-recommended-set-wpr_mailto-as-a-github-actions-secret)
  * [Responsible use and limitations](#responsible-use-and-limitations)
  * [Customizing the report theme](#customizing-the-report-theme)
    * [Using an existing theme](#using-an-existing-theme)
    * [Adding a new theme](#adding-a-new-theme)
  * [License](#license)
<!-- TOC -->

## Key features

- **Automated weekly literature monitoring:** 

    Automatically queries recent publications (rolling time window, default: 7 days) based on user-defined keywords and followed authors.

- **Crossref-based search with ORCID support:** 

    Uses the Crossref API for open bibliographic data retrieval and supports author tracking via ORCID identifiers.

- **Keyword-driven relevance ranking:** 

    Results are ranked by Crossref relevance score, helping surface the most relevant papers first.

- **Topic clustering of search results:** 

    Applies TF-IDF-based clustering on paper titles to group results into thematic clusters and highlight key research directions.

- **Self-contained static HTML report:** 

    Outputs a fully self-contained report (`index.html` + assets) that can be hosted on GitHub Pages or shared offline.

## Demo

An example weekly report is available at: [Weekly Paper Report](https://hqrrr.github.io/weekly-paper-report/)

![Demo Screenshot](_pics/demo_screenshot.png)

## How to use

### 1) Fork & enable GitHub Actions

1. Fork this repository to your own GitHub account.
2. Open your fork -> **Actions** tab -> enable workflows if GitHub asks you to  
   (**scheduled workflows are often disabled by default on forks**).

### 2) Configure GitHub Pages

In your fork:
1. Go to **Settings -> Pages**
2. Set **Source** to **GitHub Actions**
3. After the workflow finishes, your report will be published to your GitHub Pages site.
4. (Optional) Trigger a manual run once: **Actions -> Build and Deploy Report to GitHub Pages -> Run workflow**, to verify everything works.

> **Where to find your report (GitHub Pages URL)**
> 
> After the workflow has finished successfully, your report will be available at:
> 
> ```
> https://<your-github-username>.github.io/<repository-name>/
> ```
>
> You can also find the exact URL in:
> - **Settings -> Pages** (shown under "Your site is live at …"), or
> - the **Deployments** page (shown under "github-pages").


### 3) Configure your keywords and followed authors

Edit these files in your fork (examples see below):
- `./config/keywords.yaml` - keywords used for Crossref search
- `./config/followed_authors.yaml` - followed authors (recommended: include ORCID)

Commit and push the changes to your default branch to regenerate the report.

#### Example: `keywords.yaml`

```yaml
# List of keywords used for literature search
keywords:
  - indoor environmental quality
  - IEQ
  - thermal comfort
  - indoor air quality
  - user behavior
```
> Each keyword is queried against Crossref. 
> Use full terms and common abbreviations where appropriate.

#### Example: `followed_authors.yaml`

```yaml
# List of followed authors
authors:
  - name: Andrew Persily
    orcid: "0000-0002-9542-3318"
    names:
      - Andrew Persily
      - A. Persily
```

> - `name` is used for display and matching in keyword search results.
> - `orcid` enables precise author-based lookup via Crossref and is **strongly recommended**. 
> - If `orcid` is not provided, the author will only be matched against keyword search results
  using the names listed in `names`. In this case, author-based lookup via Crossref
  is not performed and results may be incomplete.

### 4) (Recommended) Set `WPR_MAILTO` as a GitHub Actions secret

Crossref recommends providing a contact email (`mailto`) for polite API usage.

In your fork:
1. Go to **Settings -> Secrets and variables -> Actions**
2. Create a new **Repository secret**:
   - Name: `WPR_MAILTO`
   - Secret: your email address

> **Weekly schedule notes (important)**
> - The workflow is configured to run on a weekly schedule (via `on: schedule`).
> - **GitHub may automatically disable scheduled runs for repositories with no activity for ~60 days.**  
>  If your weekly updates stop, simply:
> - make a small commit (e.g., edit README), and/or
> - re-enable the workflow in the **Actions** tab.

## Responsible use and limitations

This project is intended as a **lightweight literature monitoring and exploration tool**.  
Please consider the following points when using or extending it:

- **Avoid excessive request frequency**

  Do not configure very short update intervals (e.g., hourly or daily runs). Excessive automated requests may violate the fair-use expectations of public APIs such as Crossref. A weekly schedule is strongly recommended.

- **Results may be incomplete**

  The report relies on open bibliographic metadata and keyword-based queries. Not all relevant publications may be indexed, linked, or returned by the data sources. Absence of a paper in the report does not imply absence in the literature.

- **Author matching is imperfect**
  
  Author-based tracking is most reliable when ORCID identifiers are available. Name-based matching may produce false positives or miss relevant works.

- **Clustering and ranking are heuristic**
  
  Topic clustering and relevance ranking are automatically derived from titles and metadata using statistical methods. These results are approximate and should be interpreted as exploratory aids rather than authoritative classifications.

- **Not a substitute for systematic review**
  
  This tool is designed to support awareness and discovery, not to replace systematic literature reviews or expert judgment.

Users are encouraged to treat the generated reports as **decision-support material**
and to verify important findings through primary sources.

## Customizing the report theme

The visual appearance of the report is controlled by a CSS theme.

### Using an existing theme

By default, the report uses the `light` theme.  
You can select a theme when generating the report in `app.py`:

```python
# app.py
## Report theme
THEME = "light"
```

Available themes are loaded from the `themes/` directory.

### Adding a new theme

To add a custom theme:

1. Copy an existing theme file, for example: `themes/light.css -> themes/dark.css`
2. Modify the CSS styles in the new file
3. Set the theme name accordingly in `app.py`: `THEME = "dark"`

> If the specified theme name cannot be found, the report will automatically fall back to the default `light` theme.

## License

This project is licensed under the **MIT License**. See the [LICENSE](./LICENSE) file for details.

The generated reports and retrieved bibliographic metadata are subject to the terms of their respective data providers.

