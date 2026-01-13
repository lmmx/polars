"""
Benchmark Polars CDC vs PyArrow CDC on HuggingFace dataset revisions.

Datasets available (approximate sizes per revision):
- cfahlgren1/hub-stats/datasets.parquet: ~85MB, 194 revisions
- cfahlgren1/hub-stats/spaces.parquet: ~125MB, 193 revisions
- openfoodfacts/product-database/food.parquet: ~6GB, 32 revisions (use with caution!)

Default: 10 revisions of datasets.parquet = ~850MB download, ~3.5GB disk with all variants

Usage:

```
# Smallest test (~425MB download, ~1.7GB disk)
python benchmark_hf_datasets.py --preset small

# Medium test (~1.7GB download, ~7GB disk)
python benchmark_hf_datasets.py --preset medium

# Custom: 10 revisions
python benchmark_hf_datasets.py --revisions 10

# Re-run without re-downloading
python benchmark_hf_datasets.py --preset small --skip-download
```
"""

import json
import os
import shutil
import subprocess
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import polars as pl
import pyarrow.parquet as pq
from de import estimate_de
from tqdm import tqdm

PRESETS = {
    "small": {
        "repo": "cfahlgren1/hub-stats",
        "file": "datasets.parquet",
        "max_revisions": 5,
        "description": "~425MB download",
    },
    "medium": {
        "repo": "cfahlgren1/hub-stats",
        "file": "datasets.parquet",
        "max_revisions": 20,
        "description": "~1.7GB download",
    },
    "large": {
        "repo": "cfahlgren1/hub-stats",
        "file": "datasets.parquet",
        "max_revisions": 50,
        "description": "~4.2GB download",
    },
}


def fetch_revisions_hf_hub(
    repo: str, file_path: str, target_dir: Path, max_revisions: int
) -> list[Path]:
    """Fetch file revisions using huggingface_hub."""
    from huggingface_hub import HfApi, hf_hub_download

    api = HfApi()
    output_dir = target_dir / "originals"
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Fetching commit history for {repo}/{file_path}...")
    commits = list(api.list_repo_commits(repo, repo_type="dataset"))
    print(f"Found {len(commits)} total commits, using first {max_revisions}")

    outputs = []
    for commit in tqdm(commits[:max_revisions], desc="Downloading"):
        rev_short = commit.commit_id[:7]
        dst = output_dir / f"{Path(file_path).stem}-{rev_short}.parquet"

        if dst.exists():
            outputs.append(dst)
            continue

        try:
            local_path = hf_hub_download(
                repo_id=repo,
                filename=file_path,
                repo_type="dataset",
                revision=commit.commit_id,
                cache_dir=target_dir / ".hf_cache",
            )
            shutil.copy(local_path, dst)
            outputs.append(dst)
        except Exception as e:
            print(f"  Skipping {rev_short}: {e}")
            continue

    return sorted(outputs)


def rewrite_file(args: tuple[Path, Path, bool, str]) -> Path:
    """Rewrite a single parquet file."""
    src, dst, use_cdc, writer = args

    if writer == "polars":
        df = pl.read_parquet(src)
        df.write_parquet(dst, compression="zstd", use_content_defined_chunking=use_cdc)
    else:
        table = pq.read_table(src)
        pq.write_table(
            table, dst, compression="zstd", use_content_defined_chunking=use_cdc
        )

    return dst


def rewrite_parallel(
    src_paths: list[Path],
    dst_dir: Path,
    use_cdc: bool,
    writer: str,
    max_workers: int = 4,
) -> list[Path]:
    """Rewrite files in parallel."""
    dst_dir.mkdir(parents=True, exist_ok=True)

    tasks = [(src, dst_dir / src.name, use_cdc, writer) for src in src_paths]
    outputs = []

    if len(tasks) <= 3:
        for task in tqdm(tasks, desc=f"{writer} CDC={use_cdc}"):
            outputs.append(rewrite_file(task))
    else:
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            futures = {executor.submit(rewrite_file, t): t for t in tasks}
            for future in tqdm(
                as_completed(futures),
                total=len(futures),
                desc=f"{writer} CDC={use_cdc}",
            ):
                outputs.append(future.result())

    return sorted(outputs)


def measure_dedup(paths: list[Path], label: str) -> dict:
    """Measure deduplication stats."""
    result = estimate_de([str(p) for p in paths])

    return {
        "label": label,
        "num_files": len(paths),
        "total_bytes": result["total_len"],
        "unique_bytes": result["chunk_bytes"],  # Pre-compression unique
        "unique_compressed_bytes": result[
            "compressed_chunk_bytes"
        ],  # Post-compression unique
    }


def fmt_size(b: float) -> str:
    """Format bytes."""
    for u in ["B", "KB", "MB", "GB"]:
        if b < 1024:
            return f"{b:.1f}{u}"
        b /= 1024
    return f"{b:.1f}TB"


def print_results(results: list[dict]):
    """Print formatted results with CORRECT metrics."""

    # Sort by what actually matters: unique compressed bytes (storage cost)
    sorted_results = sorted(results, key=lambda r: r["unique_compressed_bytes"])

    print("\n" + "=" * 85)
    print("WHAT ACTUALLY MATTERS: Absolute bytes stored after deduplication")
    print("=" * 85)
    print("(Lower 'Unique Compressed' = less storage needed = BETTER)\n")

    print(
        f"{'Rank':<6} {'Method':<20} {'Total':>10} {'Unique':>12} {'Unique Comp':>14}"
    )
    print("-" * 85)

    best = sorted_results[0]["unique_compressed_bytes"]

    for i, r in enumerate(sorted_results, 1):
        unique_comp = r["unique_compressed_bytes"]
        overhead = ((unique_comp / best) - 1) * 100 if best > 0 else 0

        marker = " ← BEST" if i == 1 else f" (+{overhead:.1f}%)" if overhead > 0 else ""

        print(
            f"{i:<6} "
            f"{r['label']:<20} "
            f"{fmt_size(r['total_bytes']):>10} "
            f"{fmt_size(r['unique_bytes']):>12} "
            f"{fmt_size(unique_comp):>14}"
            f"{marker}"
        )

    # Now show the MISLEADING ratio metric for comparison
    print("\n" + "-" * 85)
    print("FOR REFERENCE: The misleading 'dedup ratio' metric (unique/total)")
    print("-" * 85)

    for r in results:
        ratio = r["unique_compressed_bytes"] / r["total_bytes"] * 100
        print(f"{r['label']:<20} {ratio:>6.1f}%  (but total size differs!)")

    # Analysis
    print("\n" + "=" * 85)
    print("ANALYSIS")
    print("=" * 85)

    # Find each variant
    pa_nocdc = next((r for r in results if r["label"] == "PyArrow NO CDC"), None)
    pa_cdc = next((r for r in results if r["label"] == "PyArrow CDC"), None)
    pl_nocdc = next((r for r in results if r["label"] == "Polars NO CDC"), None)
    pl_cdc = next((r for r in results if r["label"] == "Polars CDC"), None)

    if all([pa_nocdc, pa_cdc, pl_nocdc, pl_cdc]):
        print("\n1. CDC BENEFIT (same library, CDC vs no-CDC):")

        pa_savings = (
            pa_nocdc["unique_compressed_bytes"] - pa_cdc["unique_compressed_bytes"]
        )
        pl_savings = (
            pl_nocdc["unique_compressed_bytes"] - pl_cdc["unique_compressed_bytes"]
        )

        print(
            f"   PyArrow: CDC saves {fmt_size(pa_savings)} ({pa_savings/pa_nocdc['unique_compressed_bytes']*100:.1f}% reduction)"
        )
        print(
            f"   Polars:  CDC saves {fmt_size(pl_savings)} ({pl_savings/pl_nocdc['unique_compressed_bytes']*100:.1f}% reduction)"
        )

        print("\n2. LIBRARY COMPARISON (Polars vs PyArrow):")

        # Without CDC
        nocdc_diff = (
            pl_nocdc["unique_compressed_bytes"] - pa_nocdc["unique_compressed_bytes"]
        )
        print(
            f"   Without CDC: Polars uses {fmt_size(abs(nocdc_diff))} {'less' if nocdc_diff < 0 else 'more'} than PyArrow"
        )

        # With CDC
        cdc_diff = pl_cdc["unique_compressed_bytes"] - pa_cdc["unique_compressed_bytes"]
        print(
            f"   With CDC:    Polars uses {fmt_size(abs(cdc_diff))} {'less' if cdc_diff < 0 else 'more'} than PyArrow"
        )

        print("\n3. OVERALL WINNER:")
        winner = sorted_results[0]
        worst = sorted_results[-1]
        total_savings = (
            worst["unique_compressed_bytes"] - winner["unique_compressed_bytes"]
        )
        print(
            f"   {winner['label']} stores {fmt_size(total_savings)} less than {worst['label']}"
        )
        print(
            f"   ({total_savings/worst['unique_compressed_bytes']*100:.1f}% storage reduction)"
        )


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Benchmark Polars CDC - FIXED METRICS")
    parser.add_argument("--preset", choices=PRESETS.keys(), default="small")
    parser.add_argument("--repo", type=str, help="Override preset repo")
    parser.add_argument("--file", type=str, help="Override preset file")
    parser.add_argument("--revisions", type=int, help="Override preset revisions")
    parser.add_argument("--work-dir", type=Path, default=Path("./cdc_benchmark"))
    parser.add_argument("--skip-download", action="store_true")
    parser.add_argument("--workers", type=int, default=4)

    args = parser.parse_args()

    preset = PRESETS[args.preset]
    repo = args.repo or preset["repo"]
    file_path = args.file or preset["file"]
    max_revisions = args.revisions or preset["max_revisions"]

    print("=" * 60)
    print("POLARS CDC BENCHMARK (Fixed Metrics)")
    print("=" * 60)
    print(f"Dataset: {repo}/{file_path}")
    print(f"Revisions: {max_revisions}")
    print("=" * 60)

    args.work_dir.mkdir(parents=True, exist_ok=True)

    # Step 1: Get files
    if args.skip_download:
        originals = sorted((args.work_dir / "originals").glob("*.parquet"))
        print(f"\nUsing {len(originals)} existing files")
    else:
        print("\nStep 1: Downloading revisions...")
        originals = fetch_revisions_hf_hub(
            repo, file_path, args.work_dir, max_revisions
        )

    if not originals:
        print("ERROR: No files found!")
        return 1

    total_size = sum(p.stat().st_size for p in originals)
    print(f"Got {len(originals)} files, {fmt_size(total_size)} total")

    # Step 2: Rewrite
    print("\nStep 2: Rewriting files...")

    variants = {}
    for label, writer, use_cdc in [
        ("PyArrow NO CDC", "pyarrow", False),
        ("PyArrow CDC", "pyarrow", True),
        ("Polars NO CDC", "polars", False),
        ("Polars CDC", "polars", True),
    ]:
        subdir = label.lower().replace(" ", "_")
        variants[label] = rewrite_parallel(
            originals,
            args.work_dir / subdir,
            use_cdc=use_cdc,
            writer=writer,
            max_workers=args.workers,
        )

    # Step 3: Measure
    print("\nStep 3: Measuring deduplication...")

    results = []
    for label, paths in variants.items():
        results.append(measure_dedup(paths, label))

    print_results(results)

    # Save
    results_file = args.work_dir / "results.json"
    with open(results_file, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to: {results_file}")

    return 0


if __name__ == "__main__":
    exit(main())
