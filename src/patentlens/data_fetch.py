"""Pull a larger G06N3 patent corpus from the public Google Patents BigQuery dataset.

`patents-public-data.patents.publications` is free to *query* (no license fee), but you
still need a GCP project with billing enabled to run BigQuery jobs against it, and every
query counts against BigQuery's free tier (1 TB scanned/month as of writing). ALWAYS run
`estimate_query_cost()` before `fetch_patents()` on a project you care about the bill for --
the WHERE clause below filters by country + CPC prefix, but BigQuery still has to scan the
title/abstract/cpc/citation columns for every US patent to evaluate that filter, which is a
few hundred GB, not a few MB.

Setup:
    pip install google-cloud-bigquery db-dtypes

    # In Colab:
    from google.colab import auth
    auth.authenticate_user()

    # Locally:
    gcloud auth application-default login
"""

_QUERY_TEMPLATE = """
SELECT
  publication_number,
  filing_date,
  publication_date,
  (SELECT text FROM UNNEST(title_localized) WHERE language = 'en' LIMIT 1) AS title,
  (SELECT text FROM UNNEST(abstract_localized) WHERE language = 'en' LIMIT 1) AS abstract,
  ARRAY_TO_STRING(
    ARRAY(SELECT code FROM UNNEST(cpc) WHERE code LIKE '{cpc_prefix}%'), '|'
  ) AS cpc_codes,
  ARRAY_TO_STRING(
    ARRAY(SELECT publication_number FROM UNNEST(citation) WHERE publication_number != ''), '|'
  ) AS cited_patents,
  ARRAY_TO_STRING(
    ARRAY(SELECT name FROM UNNEST(inventor_harmonized) WHERE name != ''), '|'
  ) AS inventors,
  ARRAY_TO_STRING(
    ARRAY(SELECT name FROM UNNEST(assignee_harmonized) WHERE name != ''), '|'
  ) AS assignees
FROM `patents-public-data.patents.publications`
WHERE country_code = '{country}'
  AND EXISTS (SELECT 1 FROM UNNEST(cpc) AS c WHERE c.code LIKE '{cpc_prefix}%')
  AND (SELECT text FROM UNNEST(title_localized) WHERE language = 'en' LIMIT 1) IS NOT NULL
  AND (SELECT text FROM UNNEST(abstract_localized) WHERE language = 'en' LIMIT 1) IS NOT NULL
{limit_clause}
"""


def build_query(cpc_prefix="G06N3", country="US", row_cap=None):
    limit_clause = f"LIMIT {int(row_cap)}" if row_cap else ""
    return _QUERY_TEMPLATE.format(cpc_prefix=cpc_prefix, country=country, limit_clause=limit_clause)


def estimate_query_cost(project_id, cpc_prefix="G06N3", country="US", row_cap=None):
    """Dry-runs the query and returns (bytes_scanned, estimated_usd).

    BigQuery on-demand pricing is $6.25/TiB scanned (first 1 TiB/month free) as of
    writing -- check https://cloud.google.com/bigquery/pricing before relying on this
    number for a real budget decision.
    """
    from google.cloud import bigquery

    client = bigquery.Client(project=project_id)
    query = build_query(cpc_prefix, country, row_cap)
    job_config = bigquery.QueryJobConfig(dry_run=True, use_query_cache=False)
    job = client.query(query, job_config=job_config)

    bytes_scanned = job.total_bytes_processed
    tib_scanned = bytes_scanned / (1024 ** 4)
    estimated_usd = max(0.0, tib_scanned - 1.0) * 6.25  # first 1 TiB/month is free
    return bytes_scanned, estimated_usd


def fetch_patents(project_id, cpc_prefix="G06N3", country="US", row_cap=None):
    """Runs the query for real and returns a DataFrame with:
    publication_number, filing_date, publication_date, title, abstract, cpc_codes
    (pipe-joined), cited_patents (pipe-joined), inventors (pipe-joined), assignees
    (pipe-joined).

    `row_cap=None` pulls every US patent BigQuery finds under the CPC prefix -- for
    G06N3 that's on the order of tens of thousands. Always run `estimate_query_cost()`
    first. If you only want a bounded sample, pass e.g. `row_cap=50000`.
    """
    from google.cloud import bigquery

    client = bigquery.Client(project=project_id)
    query = build_query(cpc_prefix, country, row_cap)
    return client.query(query).to_dataframe()
