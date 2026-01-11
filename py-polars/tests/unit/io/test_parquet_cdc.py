# tests/unit/io/test_parquet_cdc.py
from pathlib import Path

import pytest

import polars as pl


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
def test_write_parquet_with_cdc(
    tmp_path: Path, cdc_option: dict[str, int] | bool | None
) -> None:
    df = pl.DataFrame(
        {
            "a": list(range(10000)),
            "b": ["x" * 100] * 10000,
        }
    )

    path = tmp_path / "test.parquet"
    df.write_parquet(path, use_content_defined_chunking=cdc_option)

    # Verify file is readable
    result = pl.read_parquet(path)
    assert result.equals(df)


def test_write_parquet_cdc_disabled(tmp_path: Path) -> None:
    df = pl.DataFrame({"a": [1, 2, 3]})
    path = tmp_path / "test.parquet"

    # Both None and False should work
    df.write_parquet(path, use_content_defined_chunking=None)
    df.write_parquet(path, use_content_defined_chunking=False)


def test_write_parquet_cdc_invalid_options(tmp_path: Path) -> None:
    df = pl.DataFrame({"a": [1, 2, 3]})
    path = tmp_path / "test.parquet"

    # min > avg should fail
    with pytest.raises(ValueError, match="min_chunk_size"):
        df.write_parquet(
            path,
            use_content_defined_chunking={
                "min_chunk_size": 1000,
                "avg_chunk_size": 100,
            },
        )
