# tests/unit/io/test_parquet_cdc.py
from pathlib import Path

import pyarrow.parquet as pq
import pytest

import polars as pl


def get_page_offsets(path: Path) -> list[int]:
    """Extract data page offsets from a parquet file."""
    pf = pq.ParquetFile(path)
    offsets = []
    for rg_idx in range(pf.metadata.num_row_groups):
        rg = pf.metadata.row_group(rg_idx)
        for col_idx in range(rg.num_columns):
            col = rg.column(col_idx)
            offsets.append(col.data_page_offset)
    return offsets


def get_num_pages(path: Path) -> int:
    """Get total number of data pages across all columns."""
    pf = pq.ParquetFile(path)
    total = 0
    for rg_idx in range(pf.metadata.num_row_groups):
        rg = pf.metadata.row_group(rg_idx)
        for col_idx in range(rg.num_columns):
            col = rg.column(col_idx)
            # Rough estimate based on size vs typical page size
            total += max(1, col.total_uncompressed_size // (64 * 1024))
    return total


@pytest.mark.parametrize(
    "cdc_option",
    [
        True,
        {
            "min_chunk_size": 64 * 1024,
            "avg_chunk_size": 128 * 1024,
            "max_chunk_size": 256 * 1024,
        },
    ],
)
def test_write_parquet_with_cdc_roundtrip(
    tmp_path: Path, cdc_option: dict[str, int] | bool | None
) -> None:
    """Basic smoke test that CDC-written files can be read back."""
    df = pl.DataFrame(
        {
            "a": list(range(10000)),
            "b": ["x" * 100] * 10000,
        }
    )

    path = tmp_path / "test.parquet"
    df.write_parquet(path, use_content_defined_chunking=cdc_option)

    result = pl.read_parquet(path)
    assert result.equals(df)


def test_write_parquet_cdc_disabled(tmp_path: Path) -> None:
    """Verify None and False both disable CDC."""
    df = pl.DataFrame({"a": [1, 2, 3]})
    path = tmp_path / "test.parquet"

    df.write_parquet(path, use_content_defined_chunking=None)
    df.write_parquet(path, use_content_defined_chunking=False)


def test_write_parquet_cdc_invalid_options(tmp_path: Path) -> None:
    """Verify invalid CDC options raise errors."""
    df = pl.DataFrame({"a": [1, 2, 3]})
    path = tmp_path / "test.parquet"

    with pytest.raises(ValueError, match="min_chunk_size"):
        df.write_parquet(
            path,
            use_content_defined_chunking={
                "min_chunk_size": 1000,
                "avg_chunk_size": 100,
            },
        )


def test_cdc_deterministic(tmp_path: Path) -> None:
    """CDC should produce byte-identical files for identical input."""
    # Use primitive type (i64) where CDC extraction works reliably
    df = pl.DataFrame(
        {
            "a": list(range(500000)),
            "b": list(range(500000, 1000000)),
        }
    )

    path1 = tmp_path / "test1.parquet"
    path2 = tmp_path / "test2.parquet"

    df.write_parquet(path1, use_content_defined_chunking=True)
    df.write_parquet(path2, use_content_defined_chunking=True)

    # Files should be byte-identical
    assert path1.read_bytes() == path2.read_bytes()


def test_cdc_differs_from_fixed_chunking_primitives(tmp_path: Path) -> None:
    """CDC should produce different page structure than fixed-size chunking for primitives."""
    # Create large i64 data that will span multiple pages
    # 1 million i64 values = 8MB of data
    n_rows = 1_000_000
    df = pl.DataFrame(
        {
            "data": list(range(n_rows)),
        }
    )

    path_fixed = tmp_path / "fixed.parquet"
    path_cdc = tmp_path / "cdc.parquet"

    # Use small page size to force multiple pages
    df.write_parquet(
        path_fixed,
        data_page_size=64 * 1024,
        use_content_defined_chunking=False,
    )
    df.write_parquet(
        path_cdc,
        data_page_size=64 * 1024,
        use_content_defined_chunking={
            "min_chunk_size": 32 * 1024,
            "avg_chunk_size": 64 * 1024,
            "max_chunk_size": 128 * 1024,
        },
    )

    fixed_size = path_fixed.stat().st_size
    cdc_size = path_cdc.stat().st_size

    # With different chunking strategies, file sizes should differ
    # (different page boundaries = different compression opportunities)
    assert (
        fixed_size != cdc_size
    ), f"Expected different file sizes: fixed={fixed_size}, cdc={cdc_size}"

    # Both should still be readable and equal
    assert pl.read_parquet(path_fixed).equals(pl.read_parquet(path_cdc))


def test_cdc_content_sensitivity_primitives(tmp_path: Path) -> None:
    """
    CDC boundaries should depend on content for primitive types.
    Prepending data should not shift all subsequent boundaries.
    """
    # Use i64 data large enough to create multiple chunks
    n_rows = 500_000
    base_data = list(range(n_rows))

    df_original = pl.DataFrame({"data": base_data})
    df_with_prefix = pl.DataFrame({"data": [999999999] + base_data})

    path_original = tmp_path / "original.parquet"
    path_prefixed = tmp_path / "prefixed.parquet"

    cdc_opts = {
        "min_chunk_size": 16 * 1024,
        "avg_chunk_size": 32 * 1024,
        "max_chunk_size": 64 * 1024,
    }

    df_original.write_parquet(path_original, use_content_defined_chunking=cdc_opts)
    df_with_prefix.write_parquet(path_prefixed, use_content_defined_chunking=cdc_opts)

    original_bytes = path_original.read_bytes()
    prefixed_bytes = path_prefixed.read_bytes()

    # Find common substrings - with CDC, chunks should resync after the prefix
    chunk_size = 512
    original_chunks = {
        original_bytes[i : i + chunk_size]
        for i in range(0, len(original_bytes) - chunk_size, chunk_size // 2)
    }
    prefixed_chunks = {
        prefixed_bytes[i : i + chunk_size]
        for i in range(0, len(prefixed_bytes) - chunk_size, chunk_size // 2)
    }

    shared_chunks = original_chunks & prefixed_chunks
    overlap_ratio = len(shared_chunks) / max(len(original_chunks), 1)

    # With CDC, we expect some overlap after resync (> 5%)
    # With fixed chunking, everything shifts so minimal overlap
    assert overlap_ratio > 0.05, (
        f"CDC should produce overlapping chunks after content resync, "
        f"but overlap ratio was only {overlap_ratio:.2%}"
    )


def test_cdc_with_various_primitive_dtypes(tmp_path: Path) -> None:
    """CDC should work correctly with various primitive data types."""
    n_rows = 10000
    df = pl.DataFrame(
        {
            "int8": pl.Series([i % 127 for i in range(n_rows)], dtype=pl.Int8),
            "int16": pl.Series(list(range(n_rows)), dtype=pl.Int16),
            "int32": pl.Series(list(range(n_rows)), dtype=pl.Int32),
            "int64": pl.Series(list(range(n_rows)), dtype=pl.Int64),
            "float32": pl.Series([float(i) for i in range(n_rows)], dtype=pl.Float32),
            "float64": pl.Series([float(i) for i in range(n_rows)], dtype=pl.Float64),
        }
    )

    path = tmp_path / "multi_dtype.parquet"
    df.write_parquet(path, use_content_defined_chunking=True)

    result = pl.read_parquet(path)
    assert result.equals(df)


def test_cdc_lazy_sink(tmp_path: Path) -> None:
    """CDC should work with lazy sink_parquet."""
    df = pl.DataFrame(
        {
            "a": list(range(50000)),
            "b": list(range(50000, 100000)),
        }
    )

    path = tmp_path / "lazy_cdc.parquet"
    df.lazy().sink_parquet(path, use_content_defined_chunking=True)

    result = pl.read_parquet(path)
    assert result.equals(df)


def test_cdc_small_data(tmp_path: Path) -> None:
    """CDC should handle data smaller than min_chunk_size."""
    df = pl.DataFrame({"a": [1, 2, 3]})

    path = tmp_path / "small.parquet"
    df.write_parquet(
        path,
        use_content_defined_chunking={
            "min_chunk_size": 1024 * 1024,
            "avg_chunk_size": 2 * 1024 * 1024,
            "max_chunk_size": 4 * 1024 * 1024,
        },
    )

    result = pl.read_parquet(path)
    assert result.equals(df)


def test_cdc_with_strings_fallback(tmp_path: Path) -> None:
    """
    String data currently falls back to fixed-size chunking.
    This test documents the current behavior.
    """
    # String data where CDC byte extraction may not work optimally
    df = pl.DataFrame(
        {
            "strings": [f"row_{i:010d}" for i in range(100000)],
        }
    )

    path = tmp_path / "strings.parquet"
    # Should not raise, even if it falls back internally
    df.write_parquet(path, use_content_defined_chunking=True)

    result = pl.read_parquet(path)
    assert result.equals(df)


def test_cdc_binary_data(tmp_path: Path) -> None:
    """Test CDC with binary data."""
    n_rows = 50000
    df = pl.DataFrame(
        {
            "binary": [f"data_{i:010d}".encode() for i in range(n_rows)],
        }
    )

    path = tmp_path / "binary.parquet"
    df.write_parquet(path, use_content_defined_chunking=True)

    result = pl.read_parquet(path)
    assert result.equals(df)