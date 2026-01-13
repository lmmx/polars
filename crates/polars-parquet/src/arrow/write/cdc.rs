//! Content-defined chunking for Parquet page boundaries.
//!
//! CDC operates on encoded bytes (post-encoding, pre-compression) to ensure
//! identical content produces identical chunk boundaries across files.

use polars_error::PolarsResult;

use crate::parquet::encoding::Encoding;
use crate::parquet::page::{DataPage, DataPageHeader, DataPageHeaderV1, DataPageHeaderV2};
use crate::parquet::schema::types::PrimitiveType as ParquetPrimitiveType;
use crate::parquet::statistics::ParquetStatistics;
use crate::parquet::CowBuffer;
use crate::write::{Descriptor, WriteOptions};

/// Represents encoded column data before page splitting.
pub struct EncodedColumnData {
    /// The encoded values (what would be written to a single page)
    pub data: Vec<u8>,
    /// Number of values encoded
    pub num_values: usize,
    /// The encoding used
    pub encoding: Encoding,
    /// Definition levels (already encoded)
    pub def_levels: Option<Vec<u8>>,
    /// Repetition levels (already encoded)
    pub rep_levels: Option<Vec<u8>>,
    /// Value boundary tracker for mapping byte offsets to value indices
    pub value_boundaries: ValueBoundaryTracker,
}

/// Tracks byte offset → value index mapping during encoding.
#[derive(Clone, Default)]
pub struct ValueBoundaryTracker {
    /// byte_offsets[i] = byte position where value i starts in encoded buffer
    offsets: Vec<usize>,
}

impl ValueBoundaryTracker {
    pub fn new() -> Self {
        Self { offsets: vec![0] }
    }

    /// Create a tracker for fixed-size elements
    pub fn for_fixed_size(num_values: usize, element_size: usize) -> Self {
        Self {
            offsets: (0..=num_values).map(|i| i * element_size).collect(),
        }
    }

    /// Record the end position of the current value
    pub fn record(&mut self, current_byte_offset: usize) {
        self.offsets.push(current_byte_offset);
    }

    /// Get the byte offset where a given value starts
    pub fn byte_at_value(&self, value_idx: usize) -> usize {
        self.offsets.get(value_idx).copied().unwrap_or(0)
    }

    /// Find the value index at or after a byte offset
    pub fn value_at_byte(&self, byte_offset: usize) -> usize {
        match self.offsets.binary_search(&byte_offset) {
            Ok(idx) => idx,
            Err(idx) => idx.min(self.offsets.len().saturating_sub(1)),
        }
    }

    pub fn num_values(&self) -> usize {
        self.offsets.len().saturating_sub(1)
    }
}

/// Information about a page's value range (for statistics computation)
pub struct PageValueRange {
    pub start_value: usize,
    pub num_values: usize,
}

/// Data for a single CDC-split page, before statistics are attached
pub struct CdcPageData {
    pub encoded_data: Vec<u8>,
    pub def_levels: Option<Vec<u8>>,
    pub rep_levels: Option<Vec<u8>>,
    pub value_range: PageValueRange,
}

/// Split encoded column data into page chunks using CDC.
/// Returns the raw page data and value ranges; caller is responsible for
/// computing statistics and creating final DataPage objects.
pub fn split_encoded_to_page_data(
    encoded: EncodedColumnData,
    cdc_options: super::ContentDefinedChunkingOptions,
) -> Vec<CdcPageData> {
    use fastcdc::v2020::FastCDC;

    let chunker = FastCDC::new(
        &encoded.data,
        cdc_options.min_size as u32,
        cdc_options.avg_size as u32,
        cdc_options.max_size as u32,
    );

    let chunks: Vec<_> = chunker.collect();

    if chunks.is_empty() {
        return vec![CdcPageData {
            encoded_data: encoded.data,
            def_levels: encoded.def_levels,
            rep_levels: encoded.rep_levels,
            value_range: PageValueRange {
                start_value: 0,
                num_values: encoded.num_values,
            },
        }];
    }

    split_at_chunk_boundaries(encoded, &chunks)
}

fn split_at_chunk_boundaries(
    encoded: EncodedColumnData,
    chunks: &[fastcdc::v2020::Chunk],
) -> Vec<CdcPageData> {
    let tracker = &encoded.value_boundaries;
    let mut pages = Vec::with_capacity(chunks.len());
    let mut current_value_start = 0usize;

    for (i, chunk) in chunks.iter().enumerate() {
        let chunk_end_byte = chunk.offset + chunk.length;

        let end_value = if i == chunks.len() - 1 {
            tracker.num_values()
        } else {
            tracker.value_at_byte(chunk_end_byte)
        };

        let num_values = end_value.saturating_sub(current_value_start);
        if num_values == 0 {
            continue;
        }

        let actual_start_byte = tracker.byte_at_value(current_value_start);
        let actual_end_byte = tracker.byte_at_value(end_value);

        let page_data = encoded.data[actual_start_byte..actual_end_byte].to_vec();

        // TODO: Proper level slicing for nested types
        let page_def_levels = encoded.def_levels.clone();
        let page_rep_levels = encoded.rep_levels.clone();

        pages.push(CdcPageData {
            encoded_data: page_data,
            def_levels: page_def_levels,
            rep_levels: page_rep_levels,
            value_range: PageValueRange {
                start_value: current_value_start,
                num_values,
            },
        });

        current_value_start = end_value;
    }

    // Handle remaining values
    if current_value_start < tracker.num_values() {
        let remaining_values = tracker.num_values() - current_value_start;
        let start_byte = tracker.byte_at_value(current_value_start);
        let page_data = encoded.data[start_byte..].to_vec();

        pages.push(CdcPageData {
            encoded_data: page_data,
            def_levels: encoded.def_levels.clone(),
            rep_levels: encoded.rep_levels.clone(),
            value_range: PageValueRange {
                start_value: current_value_start,
                num_values: remaining_values,
            },
        });
    }

    pages
}

/// Create a DataPage from CDC-split encoded data with pre-computed statistics.
pub fn create_data_page(
    page_data: CdcPageData,
    encoding: Encoding,
    type_: ParquetPrimitiveType,
    options: WriteOptions,
    statistics: Option<ParquetStatistics>,
) -> PolarsResult<DataPage> {
    let def_levels_byte_length = page_data.def_levels.as_ref().map(|d| d.len()).unwrap_or(0);
    let rep_levels_byte_length = page_data.rep_levels.as_ref().map(|r| r.len()).unwrap_or(0);
    let num_values = page_data.value_range.num_values;

    let mut buffer = Vec::with_capacity(
        rep_levels_byte_length + def_levels_byte_length + page_data.encoded_data.len(),
    );

    if let Some(rep) = page_data.rep_levels {
        buffer.extend_from_slice(&rep);
    }
    if let Some(def) = page_data.def_levels {
        buffer.extend_from_slice(&def);
    }
    buffer.extend_from_slice(&page_data.encoded_data);

    let header = match options.version {
        crate::write::Version::V1 => DataPageHeader::V1(DataPageHeaderV1 {
            num_values: num_values as i32,
            encoding: encoding.into(),
            definition_level_encoding: Encoding::Rle.into(),
            repetition_level_encoding: Encoding::Rle.into(),
            statistics,
        }),
        crate::write::Version::V2 => DataPageHeader::V2(DataPageHeaderV2 {
            num_values: num_values as i32,
            encoding: encoding.into(),
            num_nulls: 0, // TODO: compute from def_levels
            num_rows: num_values as i32,
            definition_levels_byte_length: def_levels_byte_length as i32,
            repetition_levels_byte_length: rep_levels_byte_length as i32,
            is_compressed: Some(false),
            statistics,
        }),
    };

    let descriptor = Descriptor {
        primitive_type: type_,
        max_def_level: if def_levels_byte_length > 0 { 1 } else { 0 },
        max_rep_level: if rep_levels_byte_length > 0 { 1 } else { 0 },
    };

    Ok(DataPage::new(
        header,
        CowBuffer::Owned(buffer),
        descriptor,
        num_values,
    ))
}