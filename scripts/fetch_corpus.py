"""Fetch a larger G06N3 patent corpus from Google Patents' public BigQuery dataset.

`src/patentlens/data_fetch.py` holds the query and returns a DataFrame; this script is
the command-line wrapper around it — it dry-runs the cost, asks before spending quota,
and writes the CSV to the exact path `artifacts.find_raw_csv()` looks for, so
`scripts/train.py` picks it up on the next run with no further configuration.

Requires a billing-enabled Google Cloud project and two dependencies that are commented
out of requirements.txt by default (nothing else in the project needs them):

    pip install google-cloud-bigquery db-dtypes
    gcloud auth application-default login

Querying `patents-public-data` carries no license fee, but BigQuery still scans the
title/abstract/cpc/citation columns of every US patent to evaluate the CPC filter — a
few hundred GB against a 1 TiB/month free tier. Estimate first; that is the default.

Run from the repo root:

    python scripts/fetch_corpus.py --project-id my-gcp-project --estimate-only
    python scripts/fetch_corpus.py --project-id my-gcp-project
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from patentlens import artifacts, data_fetch  # noqa: E402
from patentlens.artifacts import MODELS_DIR, RAW_CSV_CANDIDATES, log  # noqa: E402

# Where train.py expects a large corpus to be. Writing anywhere else means it silently
# falls back to the committed 3,000-patent pilot CSV.
DEFAULT_OUTPUT = RAW_CSV_CANDIDATES[0]


def require_bigquery():
    """Fail with the install command rather than a bare ImportError traceback."""
    try:
        import google.cloud.bigquery  # noqa: F401
    except ImportError:
        sys.exit(
            "google-cloud-bigquery is not installed (it is commented out of "
            "requirements.txt by default).\n\n"
            "    pip install google-cloud-bigquery db-dtypes\n"
            "    gcloud auth application-default login\n"
        )


def report_cost(project_id, cpc_prefix, country, row_cap):
    """Dry-run the query and print what it would scan. Returns the estimated USD."""
    log("Dry-running the query to estimate cost...")
    scanned_bytes, estimated_usd = data_fetch.estimate_query_cost(
        project_id=project_id, cpc_prefix=cpc_prefix, country=country, row_cap=row_cap
    )
    gib = scanned_bytes / (1024 ** 3)
    log(f"Would scan {gib:,.1f} GiB — estimated ${estimated_usd:,.2f} "
        "(assumes the 1 TiB/month free tier is untouched; see data_fetch for pricing caveats)")
    if row_cap:
        log("Note: LIMIT caps rows returned, not bytes scanned — the filter still reads "
            "every US patent's text columns, so --row-cap does not reduce the figure above.")
    return estimated_usd


def warn_about_stale_models():
    """train.py checkpoints on file existence, not corpus identity — a models/ left over
    from a previous corpus is reused silently, producing results for the wrong dataset."""
    if MODELS_DIR.exists() and any(MODELS_DIR.iterdir()):
        log("")
        log(f"WARNING: {MODELS_DIR} already exists and is non-empty.")
        log("  scripts/train.py resumes from whatever is on disk and will NOT notice the")
        log("  corpus changed — it would reuse artifacts built from the previous dataset.")
        log(f"  Delete it before retraining:  rm -rf {MODELS_DIR}")


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--project-id", required=True,
                        help="Google Cloud project to bill the BigQuery job to")
    parser.add_argument("--cpc-prefix", default="G06N3",
                        help="CPC prefix to filter on (default: G06N3, neural networks)")
    parser.add_argument("--country", default="US",
                        help="patent country code (default: US)")
    parser.add_argument("--row-cap", type=int, default=100_000,
                        help="max rows to return; 0 means no limit (default: 100000)")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT,
                        help=f"destination CSV (default: {DEFAULT_OUTPUT})")
    parser.add_argument("--estimate-only", action="store_true",
                        help="print the cost estimate and exit without running the query")
    parser.add_argument("--force", action="store_true",
                        help="overwrite the destination CSV if it already exists")
    parser.add_argument("--yes", action="store_true",
                        help="skip the confirmation prompt (for non-interactive runs)")
    args = parser.parse_args()

    require_bigquery()
    row_cap = args.row_cap or None

    if args.output.exists() and not (args.force or args.estimate_only):
        sys.exit(f"{args.output} already exists. Pass --force to overwrite it.")

    report_cost(args.project_id, args.cpc_prefix, args.country, row_cap)

    if args.estimate_only:
        log("Estimate only — no query run.")
        return

    if not args.yes:
        if input("\nRun the query for real? [y/N] ").strip().lower() not in ("y", "yes"):
            log("Aborted. Nothing was queried.")
            return

    log(f"Querying {args.cpc_prefix} / {args.country}...")
    df = data_fetch.fetch_patents(
        project_id=args.project_id, cpc_prefix=args.cpc_prefix,
        country=args.country, row_cap=row_cap,
    )
    log(f"Fetched {len(df):,} rows, {len(df.columns)} columns")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.output, index=False)
    log(f"Wrote {args.output} ({args.output.stat().st_size / 1024**2:,.1f} MB)")

    found = artifacts.find_raw_csv()
    if found == args.output:
        log("scripts/train.py will use this corpus on its next run.")
    else:
        log(f"NOTE: train.py reads {found}, not this file — it only looks at "
            f"{[str(p) for p in RAW_CSV_CANDIDATES]}.")

    warn_about_stale_models()


if __name__ == "__main__":
    main()
