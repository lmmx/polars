# Create two dataframes with long strings, one with an insertion
import polars as pl

# Long strings (>12 bytes, won't inline)
base = [f"this_is_a_long_string_number_{i:06d}" for i in range(100_000)]
inserted = ["INSERTED_ROW_HERE_ABCDEF"] + base

df1 = pl.DataFrame({"text": base})
df2 = pl.DataFrame({"text": inserted})

# Write with CDC
df1.write_parquet("/tmp/test1_cdc.parquet", use_content_defined_chunking=True)
df2.write_parquet("/tmp/test2_cdc.parquet", use_content_defined_chunking=True)

# Check dedup
from de import estimate_de
result = estimate_de(["/tmp/test1_cdc.parquet", "/tmp/test2_cdc.parquet"])
print(f"Dedup ratio: {result['compressed_chunk_bytes']/result['total_len']*100:.1f}%")
# Expected: poor dedup (~90%+) because views don't resync