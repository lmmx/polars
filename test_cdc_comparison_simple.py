import tempfile
from pathlib import Path

import polars as pl
import pyarrow as pa
import pyarrow.parquet as pq
from de import estimate, estimate_de

n_rows = 100_000
base_data = list(range(n_rows))

with tempfile.TemporaryDirectory() as tmp:
    tmp = Path(tmp)

    table_original = pa.table({"data": base_data})
    table_prefixed = pa.table({"data": [999999999] + base_data})

    df_original = pl.DataFrame({"data": base_data})
    df_prefixed = pl.DataFrame({"data": [999999999] + base_data})

    # PyArrow NO CDC
    pq.write_table(table_original, tmp / "pyarrow_nocdc_1.parquet", compression="zstd")
    pq.write_table(table_prefixed, tmp / "pyarrow_nocdc_2.parquet", compression="zstd")

    # PyArrow CDC
    pq.write_table(
        table_original,
        tmp / "pyarrow_cdc_1.parquet",
        compression="zstd",
        use_content_defined_chunking=True,
    )
    pq.write_table(
        table_prefixed,
        tmp / "pyarrow_cdc_2.parquet",
        compression="zstd",
        use_content_defined_chunking=True,
    )

    # Polars NO CDC
    df_original.write_parquet(
        tmp / "polars_nocdc_1.parquet", use_content_defined_chunking=False
    )
    df_prefixed.write_parquet(
        tmp / "polars_nocdc_2.parquet", use_content_defined_chunking=False
    )

    # Polars CDC
    df_original.write_parquet(
        tmp / "polars_cdc_1.parquet", use_content_defined_chunking=True
    )
    df_prefixed.write_parquet(
        tmp / "polars_cdc_2.parquet", use_content_defined_chunking=True
    )

    results = []

    for name, f1, f2 in [
        ("PyArrow NO CDC", "pyarrow_nocdc_1.parquet", "pyarrow_nocdc_2.parquet"),
        ("PyArrow CDC", "pyarrow_cdc_1.parquet", "pyarrow_cdc_2.parquet"),
        ("Polars NO CDC", "polars_nocdc_1.parquet", "polars_nocdc_2.parquet"),
        ("Polars CDC", "polars_cdc_1.parquet", "polars_cdc_2.parquet"),
    ]:
        estimated = estimate_de([str(tmp / f1), str(tmp / f2)])
        total = estimated["total_len"]
        deduped = estimated["compressed_chunk_bytes"]
        # Pre-compression (we don't need care about this) is chunk_bytes
        # unique_bytes = estimated["chunk_bytes"]
        saved_pct = (1 - deduped / total) * 100
        results.append((name, total, deduped, saved_pct))

    print(f"{'Method':<20} {'Total':>12} {'After Dedupe':>14} {'Saved':>8}")
    print("-" * 58)
    for name, total, deduped, saved in results:
        print(
            f"{name:<20} {total/1024:>10.1f} kB {deduped/1024:>12.1f} kB {saved:>7.1f}%"
        )

    print("\n--- Interpretation ---")
    print("'Saved' = % reduction when storing both files in Xet.")
    print("Higher is better. CDC should improve this vs no-CDC.")
