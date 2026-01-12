# test_cdc_utf8view_short_strings.py
import tempfile
from pathlib import Path

import polars as pl
import pyarrow as pa
import pyarrow.parquet as pq
from de import estimate_de

n_rows = 200_000

# SHORT strings (≤12 bytes) - these get INLINED in Utf8View
# "u_000001" = 8 bytes, fits in inline storage
base_data = [f"u_{i:06d}" for i in range(n_rows)]

# Insert 100 rows in the middle
mid = n_rows // 2
inserted = [f"NEW_{i:04d}" for i in range(100)]  # Also short strings
mutated_data = base_data[:mid] + inserted + base_data[mid:]

with tempfile.TemporaryDirectory() as tmp:
    tmp = Path(tmp)

    table_original = pa.table({"data": base_data})
    table_mutated = pa.table({"data": mutated_data})

    df_original = pl.DataFrame({"data": base_data})
    df_mutated = pl.DataFrame({"data": mutated_data})

    # PyArrow NO CDC
    pq.write_table(table_original, tmp / "pyarrow_nocdc_1.parquet", compression="zstd")
    pq.write_table(table_mutated, tmp / "pyarrow_nocdc_2.parquet", compression="zstd")

    # PyArrow CDC
    pq.write_table(
        table_original,
        tmp / "pyarrow_cdc_1.parquet",
        compression="zstd",
        use_content_defined_chunking=True,
    )
    pq.write_table(
        table_mutated,
        tmp / "pyarrow_cdc_2.parquet",
        compression="zstd",
        use_content_defined_chunking=True,
    )

    # Polars NO CDC
    df_original.write_parquet(
        tmp / "polars_nocdc_1.parquet", use_content_defined_chunking=False
    )
    df_mutated.write_parquet(
        tmp / "polars_nocdc_2.parquet", use_content_defined_chunking=False
    )

    # Polars CDC
    df_original.write_parquet(
        tmp / "polars_cdc_1.parquet", use_content_defined_chunking=True
    )
    df_mutated.write_parquet(
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
        saved_pct = (1 - deduped / total) * 100
        results.append((name, total, deduped, saved_pct))

    print("=== SHORT STRINGS TEST (≤12 bytes, inlined in views) ===\n")
    print(f"Original: {n_rows:,} rows")
    print(f"Mutated: {len(mutated_data):,} rows (+{len(inserted)} inserted in middle)")
    print(f"String length: {len(base_data[0])} bytes (inlined)\n")
    
    print(f"{'Method':<20} {'Total':>12} {'After Dedupe':>14} {'Saved':>8}")
    print("-" * 58)
    for name, total, deduped, saved in results:
        print(
            f"{name:<20} {total/1024:>10.1f} kB {deduped/1024:>12.1f} kB {saved:>7.1f}%"
        )

    print("\n--- Interpretation ---")
    print("For short strings (≤12 bytes), content is inlined in views.")
    print("CDC should help Polars resync after insertion.")
    print("Look for: Polars CDC > Polars NO CDC")