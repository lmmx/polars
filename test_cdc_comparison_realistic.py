# test_cdc_comparison_wide_payload.py
import tempfile
from pathlib import Path
import json
import random
import string

import polars as pl
import pyarrow as pa
import pyarrow.parquet as pq
from de import estimate_de

N = 200_000
PAYLOAD_SIZE = 512  # bytes per row

def make_payload(i):
    # semi-structured, stable content
    return json.dumps({
        "user": f"user_{i:06d}",
        "tags": [f"tag_{j}" for j in range(i % 5)],
        "payload": "x" * PAYLOAD_SIZE,
    })

base = {
    "user_id": list(range(N)),
    "ts": list(range(N)),
    "payload": [make_payload(i) for i in range(N)],
}

# Insert 100 rows in the middle (keys change, payload mostly stable)
mid = N // 2
insert = {
    "user_id": [-1] * 100,
    "ts": [-1] * 100,
    "payload": [make_payload(42)] * 100,
}

mutated = {
    k: v[:mid] + insert[k] + v[mid:]
    for k, v in base.items()
}

with tempfile.TemporaryDirectory() as tmp:
    tmp = Path(tmp)

    # Arrow tables
    pa_orig = pa.table(base)
    pa_mut = pa.table(mutated)

    # Polars frames
    pl_orig = pl.DataFrame(base)
    pl_mut = pl.DataFrame(mutated)

    # Write all variants
    for name, writer in [
        ("pyarrow_nocdc", lambda t, p: pq.write_table(t, p, compression="zstd")),
        ("pyarrow_cdc", lambda t, p: pq.write_table(
            t, p, compression="zstd", use_content_defined_chunking=True
        )),
    ]:
        writer(pa_orig, tmp / f"{name}_1.parquet")
        writer(pa_mut, tmp / f"{name}_2.parquet")

    for name, use_cdc in [
        ("polars_nocdc", False),
        ("polars_cdc", True),
    ]:
        pl_orig.write_parquet(
            tmp / f"{name}_1.parquet",
            use_content_defined_chunking=use_cdc,
        )
        pl_mut.write_parquet(
            tmp / f"{name}_2.parquet",
            use_content_defined_chunking=use_cdc,
        )

    results = []
    for name in [
        "pyarrow_nocdc",
        "pyarrow_cdc",
        "polars_nocdc",
        "polars_cdc",
    ]:
        est = estimate_de([
            str(tmp / f"{name}_1.parquet"),
            str(tmp / f"{name}_2.parquet"),
        ])
        total = est["total_len"]
        dedup = est["compressed_chunk_bytes"]
        saved = (1 - dedup / total) * 100
        results.append((name, total, dedup, saved))

    print(f"{'Method':<20} {'Total':>12} {'After Dedupe':>14} {'Saved':>8}")
    print("-" * 58)
    for name, total, dedup, saved in results:
        print(
            f"{name:<20} {total/1024/1024:>8.2f} MB "
            f"{dedup/1024/1024:>10.2f} MB {saved:>7.1f}%"
        )
