//! APIs to write to Parquet format.
//!
//! # Arrow/Parquet Interoperability
//! As of [parquet-format v2.9](https://github.com/apache/parquet-format/blob/master/LogicalTypes.md)
//! there are Arrow [DataTypes](arrow::datatypes::ArrowDataType) which do not have a parquet
//! representation. These include but are not limited to:
//! * `ArrowDataType::Timestamp(TimeUnit::Second, _)`
//! * `ArrowDataType::Int64`
//! * `ArrowDataType::Duration`
//! * `ArrowDataType::Date64`
//! * `ArrowDataType::Time32(TimeUnit::Second)`
//!
//! The use of these arrow types will result in no logical type being stored within a parquet file.

mod binary;
mod binview;
mod boolean;
mod dictionary;
mod file;
mod fixed_size_binary;
mod nested;
mod pages;
mod primitive;
mod row_group;
mod schema;
mod utils;

use arrow::array::*;
use arrow::bitmap::Bitmap;
use arrow::datatypes::*;
use arrow::types::{NativeType, days_ms, i256};
#[cfg(feature = "content_defined_chunking")]
use fastcdc::v2020::FastCDC;
pub use nested::{num_values, write_rep_and_def};
pub use pages::{to_leaves, to_nested, to_parquet_leaves};
use polars_utils::float16::pf16;
use polars_utils::pl_str::PlSmallStr;
pub use utils::write_def_levels;

pub use crate::parquet::compression::{BrotliLevel, CompressionOptions, GzipLevel, ZstdLevel};
pub use crate::parquet::encoding::Encoding;
pub use crate::parquet::metadata::{
    Descriptor, FileMetadata, KeyValue, SchemaDescriptor, ThriftFileMetadata,
};
pub use crate::parquet::page::{CompressedDataPage, CompressedPage, Page};
use crate::parquet::schema::Repetition;
use crate::parquet::schema::types::PrimitiveType as ParquetPrimitiveType;
pub use crate::parquet::schema::types::{
    FieldInfo, ParquetType, PhysicalType as ParquetPhysicalType,
};
pub use crate::parquet::write::{
    Compressor, DynIter, DynStreamingIterator, RowGroupIterColumns, Version, compress,
    write_metadata_sidecar,
};
pub use crate::parquet::{FallibleStreamingIterator, fallible_streaming_iterator};
use crate::write::fixed_size_binary::build_statistics_float16;

/// The statistics to write
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[cfg_attr(feature = "dsl-schema", derive(schemars::JsonSchema))]
pub struct StatisticsOptions {
    pub min_value: bool,
    pub max_value: bool,
    pub distinct_count: bool,
    pub null_count: bool,
}

impl Default for StatisticsOptions {
    fn default() -> Self {
        Self {
            min_value: true,
            max_value: true,
            distinct_count: false,
            null_count: true,
        }
    }
}

/// Options for content-defined chunking of Parquet pages.
///
/// When enabled, page boundaries are determined by content hashing rather than
/// fixed sizes, enabling efficient deduplication on content-addressable storage
/// systems like HuggingFace Hub's Xet storage layer.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[cfg_attr(feature = "dsl-schema", derive(schemars::JsonSchema))]
pub struct ContentDefinedChunkingOptions {
    /// Minimum chunk size in bytes.
    pub min_size: usize,
    /// Average/target chunk size in bytes.
    pub avg_size: usize,
    /// Maximum chunk size in bytes.
    pub max_size: usize,
}

impl ContentDefinedChunkingOptions {
    /// Minimum allowed avg_size (fastcdc requirement)
    pub const MIN_AVG_SIZE: usize = 256;
    /// Minimum allowed min_size (fastcdc requirement)
    pub const MIN_MIN_SIZE: usize = 64;

    pub fn validate(&self) -> Result<(), String> {
        if self.min_size < Self::MIN_MIN_SIZE {
            return Err(format!(
                "min_size ({}) must be >= {}",
                self.min_size,
                Self::MIN_MIN_SIZE
            ));
        }
        if self.avg_size < Self::MIN_AVG_SIZE {
            return Err(format!(
                "avg_size ({}) must be >= {}",
                self.avg_size,
                Self::MIN_AVG_SIZE
            ));
        }
        if self.min_size > self.avg_size {
            return Err(format!(
                "min_size ({}) must be <= avg_size ({})",
                self.min_size,
                self.avg_size
            ));
        }
        if self.avg_size > self.max_size {
            return Err(format!(
                "avg_size ({}) must be <= max_size ({})",
                self.avg_size,
                self.max_size
            ));
        }
        Ok(())
    }
}

impl Default for ContentDefinedChunkingOptions {
    fn default() -> Self {
        Self {
            min_size: 256 * 1024,  // 256 KiB
            avg_size: 512 * 1024,  // 512 KiB
            max_size: 1024 * 1024, // 1 MiB
        }
    }
}

/// Options to encode an array
#[derive(Clone, Copy)]
pub enum EncodeNullability {
    Required,
    Optional,
}

/// Currently supported options to write to parquet
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct WriteOptions {
    /// Whether to write statistics
    pub statistics: StatisticsOptions,
    /// The page and file version to use
    pub version: Version,
    /// The compression to apply to every page
    pub compression: CompressionOptions,
    /// The size to flush a page, defaults to 1024 * 1024 if None
    pub data_page_size: Option<usize>,
    /// Content-defined chunking options. When Some, uses CDC for page boundaries.
    /// When None, uses traditional fixed-size chunking.
    pub content_defined_chunking: Option<ContentDefinedChunkingOptions>,
}

#[derive(Clone)]
pub struct ColumnWriteOptions {
    pub field_id: Option<i32>,
    pub metadata: Vec<KeyValue>,
    pub required: Option<bool>,
    pub children: ChildWriteOptions,
}

#[derive(Clone)]
pub enum ChildWriteOptions {
    Leaf(FieldWriteOptions),
    ListLike(Box<ListLikeFieldWriteOptions>),
    Struct(Box<StructFieldWriteOptions>),
}

impl ColumnWriteOptions {
    pub fn to_leaves<'a>(&'a self, out: &mut Vec<&'a FieldWriteOptions>) {
        match &self.children {
            ChildWriteOptions::Leaf(o) => out.push(o),
            ChildWriteOptions::ListLike(o) => o.child.to_leaves(out),
            ChildWriteOptions::Struct(o) => {
                for o in &o.children {
                    o.to_leaves(out);
                }
            },
        }
    }
}

#[derive(Clone)]
pub struct FieldWriteOptions {
    pub encoding: Encoding,
}

impl ColumnWriteOptions {
    pub fn default_with(children: ChildWriteOptions) -> Self {
        Self {
            field_id: None,
            metadata: Vec::new(),
            required: None,
            children,
        }
    }
}

impl FieldWriteOptions {
    pub fn default_with_encoding(encoding: Encoding) -> Self {
        Self { encoding }
    }

    pub fn into_default_column_write_options(self) -> ColumnWriteOptions {
        ColumnWriteOptions::default_with(ChildWriteOptions::Leaf(self))
    }
}

#[derive(Clone)]
pub struct ListLikeFieldWriteOptions {
    pub child: ColumnWriteOptions,
}

#[derive(Clone)]
pub struct StructFieldWriteOptions {
    pub children: Vec<ColumnWriteOptions>,
}

use arrow::compute::aggregate::estimated_bytes_size;
use arrow::match_integer_type;
pub use file::FileWriter;
pub use pages::{Nested, array_to_columns, arrays_to_columns};
use polars_error::{PolarsResult, polars_bail};
pub use row_group::{RowGroupIterator, row_group_iter};
pub use schema::{schema_to_metadata_key, to_parquet_type};

use self::pages::{FixedSizeListNested, PrimitiveNested, StructNested};
use crate::write::dictionary::encode_as_dictionary_optional;

impl StatisticsOptions {
    pub fn empty() -> Self {
        Self {
            min_value: false,
            max_value: false,
            distinct_count: false,
            null_count: false,
        }
    }

    pub fn full() -> Self {
        Self {
            min_value: true,
            max_value: true,
            distinct_count: true,
            null_count: true,
        }
    }

    pub fn is_empty(&self) -> bool {
        !(self.min_value || self.max_value || self.distinct_count || self.null_count)
    }

    pub fn is_full(&self) -> bool {
        self.min_value && self.max_value && self.distinct_count && self.null_count
    }
}

impl WriteOptions {
    pub fn has_statistics(&self) -> bool {
        !self.statistics.is_empty()
    }
}

impl EncodeNullability {
    const fn new(is_optional: bool) -> Self {
        if is_optional {
            Self::Optional
        } else {
            Self::Required
        }
    }

    fn is_optional(self) -> bool {
        matches!(self, Self::Optional)
    }
}

/// returns offset and length to slice the leaf values
pub fn slice_nested_leaf(nested: &[Nested]) -> (usize, usize) {
    // find the deepest recursive dremel structure as that one determines how many values we must
    // take
    let mut out = (0, 0);
    for nested in nested.iter().rev() {
        match nested {
            Nested::LargeList(l_nested) => {
                let start = *l_nested.offsets.first();
                let end = *l_nested.offsets.last();
                return (start as usize, (end - start) as usize);
            },
            Nested::List(l_nested) => {
                let start = *l_nested.offsets.first();
                let end = *l_nested.offsets.last();
                return (start as usize, (end - start) as usize);
            },
            Nested::FixedSizeList(nested) => return (0, nested.length * nested.width),
            Nested::Primitive(nested) => out = (0, nested.length),
            Nested::Struct(_) => {},
        }
    }
    out
}

fn decimal_length_from_precision(precision: usize) -> usize {
    // digits = floor(log_10(2^(8*n - 1) - 1))
    // ceil(digits) = log10(2^(8*n - 1) - 1)
    // 10^ceil(digits) = 2^(8*n - 1) - 1
    // 10^ceil(digits) + 1 = 2^(8*n - 1)
    // log2(10^ceil(digits) + 1) = (8*n - 1)
    // log2(10^ceil(digits) + 1) + 1 = 8*n
    // (log2(10^ceil(a) + 1) + 1) / 8 = n
    (((10.0_f64.powi(precision as i32) + 1.0).log2() + 1.0) / 8.0).ceil() as usize
}

/// Creates a parquet [`SchemaDescriptor`] from a [`ArrowSchema`].
pub fn to_parquet_schema(
    schema: &ArrowSchema,
    column_options: &[ColumnWriteOptions],
) -> PolarsResult<SchemaDescriptor> {
    let parquet_types = schema
        .iter_values()
        .zip(column_options)
        .map(|(field, options)| to_parquet_type(field, options))
        .collect::<PolarsResult<Vec<_>>>()?;
    Ok(SchemaDescriptor::new(
        PlSmallStr::from_static("root"),
        parquet_types,
    ))
}

/// Slices the [`Array`] to `Box<dyn Array>` and `Vec<Nested>`.
pub fn slice_parquet_array(
    primitive_array: &mut dyn Array,
    nested: &mut [Nested],
    mut current_offset: usize,
    mut current_length: usize,
) {
    for nested in nested.iter_mut() {
        match nested {
            Nested::LargeList(l_nested) => {
                l_nested.offsets.slice(current_offset, current_length + 1);
                if let Some(validity) = l_nested.validity.as_mut() {
                    validity.slice(current_offset, current_length)
                };

                // Update the offset/ length so that the Primitive is sliced properly.
                current_length = l_nested.offsets.range() as usize;
                current_offset = *l_nested.offsets.first() as usize;
            },
            Nested::List(l_nested) => {
                l_nested.offsets.slice(current_offset, current_length + 1);
                if let Some(validity) = l_nested.validity.as_mut() {
                    validity.slice(current_offset, current_length)
                };

                // Update the offset/ length so that the Primitive is sliced properly.
                current_length = l_nested.offsets.range() as usize;
                current_offset = *l_nested.offsets.first() as usize;
            },
            Nested::Struct(StructNested {
                validity, length, ..
            }) => {
                *length = current_length;
                if let Some(validity) = validity.as_mut() {
                    validity.slice(current_offset, current_length)
                };
            },
            Nested::Primitive(PrimitiveNested {
                validity, length, ..
            }) => {
                *length = current_length;
                if let Some(validity) = validity.as_mut() {
                    validity.slice(current_offset, current_length)
                };
                primitive_array.slice(current_offset, current_length);
            },
            Nested::FixedSizeList(FixedSizeListNested {
                validity,
                length,
                width,
                ..
            }) => {
                if let Some(validity) = validity.as_mut() {
                    validity.slice(current_offset, current_length)
                };
                *length = current_length;
                // Update the offset/ length so that the Primitive is sliced properly.
                current_length *= *width;
                current_offset *= *width;
            },
        }
    }
}

/// Get the length of [`Array`] that should be sliced.
pub fn get_max_length(nested: &[Nested]) -> usize {
    let mut length = 0;
    for nested in nested.iter() {
        match nested {
            Nested::LargeList(l_nested) => length += l_nested.offsets.range() as usize,
            Nested::List(l_nested) => length += l_nested.offsets.range() as usize,
            Nested::FixedSizeList(nested) => length += nested.length * nested.width,
            _ => {},
        }
    }
    length
}

/// Returns an iterator of [`Page`].
pub fn array_to_pages(
    primitive_array: &dyn Array,
    type_: ParquetPrimitiveType,
    nested: &[Nested],
    options: WriteOptions,
    field_options: &FieldWriteOptions,
) -> PolarsResult<DynIter<'static, PolarsResult<Page>>> {
    let mut encoding = field_options.encoding;
    if let ArrowDataType::Dictionary(key_type, _, _) = primitive_array.dtype().to_storage() {
        return match_integer_type!(key_type, |$T| {
            dictionary::array_to_pages::<$T>(
                primitive_array.as_any().downcast_ref().unwrap(),
                type_,
                &nested,
                options,
                encoding,
            )
        });
    };
    if let Encoding::RleDictionary = encoding {
        // Only take this path for primitive columns
        if matches!(nested.first(), Some(Nested::Primitive(_))) {
            if let Some(result) =
                encode_as_dictionary_optional(primitive_array, nested, type_.clone(), options)
            {
                return result;
            }
        }

        // We didn't succeed, fallback to plain
        encoding = Encoding::Plain;
    }

    let nested = nested.to_vec();

    let number_of_rows = nested[0].len();

    // note: this is not correct if the array is sliced - the estimation should happen on the
    // primitive after sliced for parquet
    let byte_size = estimated_bytes_size(primitive_array);

    const DEFAULT_PAGE_SIZE: usize = 1024 * 1024;
    let max_page_size = options.data_page_size.unwrap_or(DEFAULT_PAGE_SIZE);
    let max_page_size = max_page_size.min(2usize.pow(31) - 2usize.pow(25)); // allowed maximum page size

    // Compute row boundaries for pages
    let row_iter: Box<dyn Iterator<Item = (usize, usize)> + Send + Sync> = {
        #[cfg(feature = "content_defined_chunking")]
        if let Some(cdc_options) = options.content_defined_chunking {
            compute_cdc_row_iter(primitive_array, number_of_rows, byte_size, cdc_options)
        } else {
            compute_fixed_row_iter(number_of_rows, byte_size, max_page_size)
        }

        #[cfg(not(feature = "content_defined_chunking"))]
        compute_fixed_row_iter(number_of_rows, byte_size, max_page_size)
    };

    let primitive_array = primitive_array.to_boxed();

    let pages = row_iter.map(move |(offset, length)| {
        let mut right_array = primitive_array.clone();
        let mut right_nested = nested.clone();
        slice_parquet_array(right_array.as_mut(), &mut right_nested, offset, length);

        array_to_page(
            right_array.as_ref(),
            type_.clone(),
            &right_nested,
            options,
            encoding,
        )
    });
    Ok(DynIter::new(pages))
}

fn compute_fixed_row_iter(
    number_of_rows: usize,
    byte_size: usize,
    max_page_size: usize,
) -> Box<dyn Iterator<Item = (usize, usize)> + Send + Sync> {
    let bytes_per_row = if number_of_rows == 0 {
        0
    } else {
        ((byte_size as f64) / (number_of_rows as f64)) as usize
    };
    let rows_per_page = (max_page_size / (bytes_per_row + 1)).max(1);

    Box::new(
        (0..number_of_rows)
            .step_by(rows_per_page)
            .map(move |offset| {
                let length = if offset + rows_per_page > number_of_rows {
                    number_of_rows - offset
                } else {
                    rows_per_page
                };
                (offset, length)
            }),
    )
}

#[cfg(feature = "content_defined_chunking")]
fn compute_cdc_row_iter(
    array: &dyn Array,
    number_of_rows: usize,
    byte_size: usize,
    cdc_options: ContentDefinedChunkingOptions,
) -> Box<dyn Iterator<Item = (usize, usize)> + Send + Sync> {
    if number_of_rows == 0 {
        return Box::new(std::iter::empty());
    }

    // Get byte data for CDC analysis
    let (data, bytes_per_element) = get_cdc_byte_data(array);

    // DEBUG: Print what's happening
    eprintln!(
        "[CDC DEBUG] dtype={:?}, physical={:?}, data={}, bytes_per_elem={:?}, num_rows={}",
        array.dtype(),
        array.dtype().to_physical_type(),
        data.as_ref().map(|d| d.len()).unwrap_or(0),
        bytes_per_element,
        number_of_rows
    );

    // If we couldn't get usable byte data, fall back to fixed-size estimation
    let Some(data) = data else {
        eprintln!("[CDC DEBUG] FALLBACK - no byte data extracted!");
        let bytes_per_row = ((byte_size as f64) / (number_of_rows as f64)) as usize;
        let rows_per_page = (cdc_options.avg_size / (bytes_per_row + 1)).max(1);
        return Box::new(
            (0..number_of_rows)
                .step_by(rows_per_page)
                .map(move |offset| {
                    let length = (offset + rows_per_page).min(number_of_rows) - offset;
                    (offset, length)
                }),
        );
    };

    // Run FastCDC on the data
    let chunker = FastCDC::new(
        &data,
        cdc_options.min_size as u32,
        cdc_options.avg_size as u32,
        cdc_options.max_size as u32,
    );
    let chunks: Vec<_> = chunker.collect();

    eprintln!(
        "[CDC DEBUG] FastCDC produced {} chunks from {} bytes",
        chunks.len(),
        data.len()
    );

    if chunks.is_empty() {
        return Box::new(std::iter::once((0, number_of_rows)));
    }

    // Convert byte boundaries to row boundaries
    let row_boundaries = if let Some(elem_size) = bytes_per_element {
        // Fixed-size elements: direct mapping
        chunks
            .into_iter()
            .scan(0usize, move |current_row, chunk| {
                let start_row = *current_row;
                // Calculate how many rows this chunk covers
                let chunk_end_byte = chunk.offset + chunk.length;
                let end_row = (chunk_end_byte / elem_size).min(number_of_rows);
                let length = end_row.saturating_sub(start_row);
                *current_row = end_row;
                if length > 0 {
                    Some((start_row, length))
                } else {
                    None
                }
            })
            .filter(|(_, len)| *len > 0)
            .collect::<Vec<_>>()
    } else {
        // Variable-size elements: use chunk count to estimate row distribution
        let rows_per_chunk = (number_of_rows + chunks.len() - 1) / chunks.len();
        (0..chunks.len())
            .map(|i| {
                let start = i * rows_per_chunk;
                let end = ((i + 1) * rows_per_chunk).min(number_of_rows);
                (start, end - start)
            })
            .filter(|(_, len)| *len > 0)
            .collect::<Vec<_>>()
    };

    // Ensure we cover all rows (handle rounding edge cases)
    let mut result = row_boundaries;
    if let Some(last) = result.last() {
        let covered = last.0 + last.1;
        if covered < number_of_rows {
            result.push((covered, number_of_rows - covered));
        }
    } else {
        result.push((0, number_of_rows));
    }

    Box::new(result.into_iter())
}

/// Extract byte data from an array for CDC analysis.
/// Returns (Option<byte_data>, Option<bytes_per_element>).
/// For fixed-size types, bytes_per_element is Some.
/// For variable-size types, bytes_per_element is None.
#[cfg(feature = "content_defined_chunking")]
fn get_cdc_byte_data(array: &dyn Array) -> (Option<Vec<u8>>, Option<usize>) {
    use arrow::datatypes::PhysicalType;

    match array.dtype().to_physical_type() {
        PhysicalType::Primitive(primitive_type) => {
            use arrow::types::PrimitiveType;
            let elem_size = match primitive_type {
                PrimitiveType::Int8 | PrimitiveType::UInt8 => 1,
                PrimitiveType::Int16 | PrimitiveType::UInt16 | PrimitiveType::Float16 => 2,
                PrimitiveType::Int32 | PrimitiveType::UInt32 | PrimitiveType::Float32 => 4,
                PrimitiveType::Int64 | PrimitiveType::UInt64 | PrimitiveType::Float64 => 8,
                PrimitiveType::Int128 | PrimitiveType::UInt128 => 16,
                PrimitiveType::Int256 => 32,
                PrimitiveType::DaysMs | PrimitiveType::MonthDayMillis => 8,
                PrimitiveType::MonthDayNano => 16,
            };

            // Get the raw values buffer
            let buffers = array.as_any();

            macro_rules! extract_primitive_bytes {
                ($native_type:ty) => {{
                    if let Some(arr) = buffers.downcast_ref::<PrimitiveArray<$native_type>>() {
                        let values = arr.values();
                        let bytes: &[u8] = bytemuck::cast_slice(values.as_slice());
                        return (Some(bytes.to_vec()), Some(elem_size));
                    }
                }};
            }

            // Try common primitive types
            extract_primitive_bytes!(i8);
            extract_primitive_bytes!(i16);
            extract_primitive_bytes!(i32);
            extract_primitive_bytes!(i64);
            extract_primitive_bytes!(i128);
            extract_primitive_bytes!(u8);
            extract_primitive_bytes!(u16);
            extract_primitive_bytes!(u32);
            extract_primitive_bytes!(u64);
            extract_primitive_bytes!(u128);
            extract_primitive_bytes!(f32);
            extract_primitive_bytes!(f64);

            (None, Some(elem_size))
        },
        PhysicalType::LargeBinary | PhysicalType::Binary => {
            // For binary data, use the values buffer directly
            if let Some(arr) = array.as_any().downcast_ref::<BinaryArray<i64>>() {
                let values = arr.values();
                return (Some(values.to_vec()), None);
            }
            if let Some(arr) = array.as_any().downcast_ref::<BinaryArray<i32>>() {
                let values = arr.values();
                return (Some(values.to_vec()), None);
            }
            (None, None)
        },
        PhysicalType::LargeUtf8 | PhysicalType::Utf8 => {
            // Utf8 is stored the same as Binary
            if let Some(arr) = array.as_any().downcast_ref::<Utf8Array<i64>>() {
                let values = arr.values();
                return (Some(values.to_vec()), None);
            }
            if let Some(arr) = array.as_any().downcast_ref::<Utf8Array<i32>>() {
                let values = arr.values();
                return (Some(values.to_vec()), None);
            }
            (None, None)
        },
        PhysicalType::BinaryView | PhysicalType::Utf8View => {
            // BinaryView stores data in buffers; collect all buffer data
            if let Some(arr) = array.as_any().downcast_ref::<BinaryViewArray>() {
                let mut data = Vec::new();
                for buffer in arr.data_buffers().iter() {
                    data.extend_from_slice(buffer.as_slice());
                }
                if !data.is_empty() {
                    return (Some(data), None);
                }
            }
            (None, None)
        },
        PhysicalType::FixedSizeBinary => {
            if let Some(arr) = array.as_any().downcast_ref::<FixedSizeBinaryArray>() {
                let size = arr.size();
                let values = arr.values();
                return (Some(values.to_vec()), Some(size));
            }
            (None, None)
        },
        PhysicalType::Boolean => {
            // Boolean arrays are bit-packed; use byte representation
            if let Some(arr) = array.as_any().downcast_ref::<BooleanArray>() {
                let values = arr.values();
                let (bytes, _, _) = values.as_slice();
                return (Some(bytes.to_vec()), None);
            }
            (None, None)
        },
        // For complex types, fall back to estimation-based chunking
        _ => (None, None),
    }
}

/// Converts an [`Array`] to a [`CompressedPage`] based on options, descriptor and `encoding`.
pub fn array_to_page(
    array: &dyn Array,
    type_: ParquetPrimitiveType,
    nested: &[Nested],
    options: WriteOptions,
    encoding: Encoding,
) -> PolarsResult<Page> {
    if nested.len() == 1 {
        // special case where validity == def levels
        return array_to_page_simple(array, type_, options, encoding);
    }
    array_to_page_nested(array, type_, nested, options, encoding)
}

/// Converts an [`Array`] to a [`CompressedPage`] based on options, descriptor and `encoding`.
pub fn array_to_page_simple(
    array: &dyn Array,
    type_: ParquetPrimitiveType,
    options: WriteOptions,
    encoding: Encoding,
) -> PolarsResult<Page> {
    let dtype = array.dtype();

    if type_.field_info.repetition == Repetition::Required && array.null_count() > 0 {
        polars_bail!(InvalidOperation: "writing a missing value to required parquet column '{}'", type_.field_info.name);
    }

    match dtype {
        // Map empty struct to boolean array with same validity.
        ArrowDataType::Struct(fs) if fs.is_empty() => boolean::array_to_page(
            &BooleanArray::new(
                ArrowDataType::Boolean,
                Bitmap::new_zeroed(array.len()),
                array.validity().cloned(),
            ),
            options,
            type_,
            encoding,
        ),

        ArrowDataType::Boolean => boolean::array_to_page(
            array.as_any().downcast_ref().unwrap(),
            options,
            type_,
            encoding,
        ),
        // casts below MUST match the casts done at the metadata (field -> parquet type).
        ArrowDataType::UInt8 => {
            return primitive::array_to_page_integer::<u8, i32>(
                array.as_any().downcast_ref().unwrap(),
                options,
                type_,
                encoding,
            );
        },
        ArrowDataType::UInt16 => {
            return primitive::array_to_page_integer::<u16, i32>(
                array.as_any().downcast_ref().unwrap(),
                options,
                type_,
                encoding,
            );
        },
        ArrowDataType::UInt32 => {
            return primitive::array_to_page_integer::<u32, i32>(
                array.as_any().downcast_ref().unwrap(),
                options,
                type_,
                encoding,
            );
        },
        ArrowDataType::UInt64 => {
            return primitive::array_to_page_integer::<u64, i64>(
                array.as_any().downcast_ref().unwrap(),
                options,
                type_,
                encoding,
            );
        },
        ArrowDataType::Int8 => {
            return primitive::array_to_page_integer::<i8, i32>(
                array.as_any().downcast_ref().unwrap(),
                options,
                type_,
                encoding,
            );
        },
        ArrowDataType::Int16 => {
            return primitive::array_to_page_integer::<i16, i32>(
                array.as_any().downcast_ref().unwrap(),
                options,
                type_,
                encoding,
            );
        },
        ArrowDataType::Int32 | ArrowDataType::Date32 | ArrowDataType::Time32(_) => {
            return primitive::array_to_page_integer::<i32, i32>(
                array.as_any().downcast_ref().unwrap(),
                options,
                type_,
                encoding,
            );
        },
        ArrowDataType::Int64
        | ArrowDataType::Date64
        | ArrowDataType::Time64(_)
        | ArrowDataType::Timestamp(_, _)
        | ArrowDataType::Duration(_) => {
            return primitive::array_to_page_integer::<i64, i64>(
                array.as_any().downcast_ref().unwrap(),
                options,
                type_,
                encoding,
            );
        },
        ArrowDataType::Float16 => {
            let array: &PrimitiveArray<pf16> = array.as_any().downcast_ref().unwrap();
            let statistics = options
                .has_statistics()
                .then(|| build_statistics_float16(array, type_.clone(), &options.statistics));
            let array = FixedSizeBinaryArray::new(
                ArrowDataType::FixedSizeBinary(2),
                array.values().clone().try_transmute().unwrap(),
                array.validity().cloned(),
            );
            fixed_size_binary::array_to_page(&array, options, type_, statistics)
        },
        ArrowDataType::Float32 => primitive::array_to_page_plain::<f32, f32>(
            array.as_any().downcast_ref().unwrap(),
            options,
            type_,
        ),
        ArrowDataType::Float64 => primitive::array_to_page_plain::<f64, f64>(
            array.as_any().downcast_ref().unwrap(),
            options,
            type_,
        ),
        ArrowDataType::LargeUtf8 => {
            let array =
                polars_compute::cast::cast(array, &ArrowDataType::LargeBinary, Default::default())
                    .unwrap();
            return binary::array_to_page::<i64>(
                array.as_any().downcast_ref().unwrap(),
                options,
                type_,
                encoding,
            );
        },
        ArrowDataType::LargeBinary => {
            return binary::array_to_page::<i64>(
                array.as_any().downcast_ref().unwrap(),
                options,
                type_,
                encoding,
            );
        },
        ArrowDataType::BinaryView => {
            return binview::array_to_page(
                array.as_any().downcast_ref().unwrap(),
                options,
                type_,
                encoding,
            );
        },
        ArrowDataType::Utf8View => {
            let array =
                polars_compute::cast::cast(array, &ArrowDataType::BinaryView, Default::default())
                    .unwrap();
            return binview::array_to_page(
                array.as_any().downcast_ref().unwrap(),
                options,
                type_,
                encoding,
            );
        },
        ArrowDataType::Null => {
            let array = Int32Array::new_null(ArrowDataType::Int32, array.len());
            primitive::array_to_page_plain::<i32, i32>(&array, options, type_)
        },
        ArrowDataType::Interval(IntervalUnit::YearMonth) => {
            let array = array
                .as_any()
                .downcast_ref::<PrimitiveArray<i32>>()
                .unwrap();
            let mut values = Vec::<u8>::with_capacity(12 * array.len());
            array.values().iter().for_each(|x| {
                let bytes = &x.to_le_bytes();
                values.extend_from_slice(bytes);
                values.extend_from_slice(&[0; 8]);
            });
            let array = FixedSizeBinaryArray::new(
                ArrowDataType::FixedSizeBinary(12),
                values.into(),
                array.validity().cloned(),
            );
            let statistics = if options.has_statistics() {
                Some(fixed_size_binary::build_statistics(
                    &array,
                    type_.clone(),
                    &options.statistics,
                ))
            } else {
                None
            };
            fixed_size_binary::array_to_page(&array, options, type_, statistics)
        },
        ArrowDataType::Interval(IntervalUnit::DayTime) => {
            let array = array
                .as_any()
                .downcast_ref::<PrimitiveArray<days_ms>>()
                .unwrap();
            let mut values = Vec::<u8>::with_capacity(12 * array.len());
            array.values().iter().for_each(|x| {
                let bytes = &x.to_le_bytes();
                values.extend_from_slice(&[0; 4]); // months
                values.extend_from_slice(bytes); // days and seconds
            });
            let array = FixedSizeBinaryArray::new(
                ArrowDataType::FixedSizeBinary(12),
                values.into(),
                array.validity().cloned(),
            );
            let statistics = if options.has_statistics() {
                Some(fixed_size_binary::build_statistics(
                    &array,
                    type_.clone(),
                    &options.statistics,
                ))
            } else {
                None
            };
            fixed_size_binary::array_to_page(&array, options, type_, statistics)
        },
        ArrowDataType::FixedSizeBinary(_) => {
            let array = array.as_any().downcast_ref().unwrap();
            let statistics = if options.has_statistics() {
                Some(fixed_size_binary::build_statistics(
                    array,
                    type_.clone(),
                    &options.statistics,
                ))
            } else {
                None
            };

            fixed_size_binary::array_to_page(array, options, type_, statistics)
        },
        ArrowDataType::Decimal256(precision, _) => {
            let precision = *precision;
            let array = array
                .as_any()
                .downcast_ref::<PrimitiveArray<i256>>()
                .unwrap();
            if precision <= 9 {
                let values = array
                    .values()
                    .iter()
                    .map(|x| x.0.as_i32())
                    .collect::<Vec<_>>()
                    .into();

                let array = PrimitiveArray::<i32>::new(
                    ArrowDataType::Int32,
                    values,
                    array.validity().cloned(),
                );
                return primitive::array_to_page_integer::<i32, i32>(
                    &array, options, type_, encoding,
                );
            } else if precision <= 18 {
                let values = array
                    .values()
                    .iter()
                    .map(|x| x.0.as_i64())
                    .collect::<Vec<_>>()
                    .into();

                let array = PrimitiveArray::<i64>::new(
                    ArrowDataType::Int64,
                    values,
                    array.validity().cloned(),
                );
                return primitive::array_to_page_integer::<i64, i64>(
                    &array, options, type_, encoding,
                );
            } else if precision <= 38 {
                let size = decimal_length_from_precision(precision);
                let statistics = if options.has_statistics() {
                    let stats = fixed_size_binary::build_statistics_decimal256_with_i128(
                        array,
                        type_.clone(),
                        size,
                        &options.statistics,
                    );
                    Some(stats)
                } else {
                    None
                };

                let mut values = Vec::<u8>::with_capacity(size * array.len());
                array.values().iter().for_each(|x| {
                    let bytes = &x.0.low().to_be_bytes()[16 - size..];
                    values.extend_from_slice(bytes)
                });
                let array = FixedSizeBinaryArray::new(
                    ArrowDataType::FixedSizeBinary(size),
                    values.into(),
                    array.validity().cloned(),
                );
                fixed_size_binary::array_to_page(&array, options, type_, statistics)
            } else {
                let size = 32;
                let array = array
                    .as_any()
                    .downcast_ref::<PrimitiveArray<i256>>()
                    .unwrap();
                let statistics = if options.has_statistics() {
                    let stats = fixed_size_binary::build_statistics_decimal256(
                        array,
                        type_.clone(),
                        size,
                        &options.statistics,
                    );
                    Some(stats)
                } else {
                    None
                };
                let mut values = Vec::<u8>::with_capacity(size * array.len());
                array.values().iter().for_each(|x| {
                    let bytes = &x.to_be_bytes();
                    values.extend_from_slice(bytes)
                });
                let array = FixedSizeBinaryArray::new(
                    ArrowDataType::FixedSizeBinary(size),
                    values.into(),
                    array.validity().cloned(),
                );

                fixed_size_binary::array_to_page(&array, options, type_, statistics)
            }
        },
        ArrowDataType::Decimal(precision, _) => {
            let precision = *precision;
            let array = array
                .as_any()
                .downcast_ref::<PrimitiveArray<i128>>()
                .unwrap();
            if precision <= 9 {
                let values = array
                    .values()
                    .iter()
                    .map(|x| *x as i32)
                    .collect::<Vec<_>>()
                    .into();

                let array = PrimitiveArray::<i32>::new(
                    ArrowDataType::Int32,
                    values,
                    array.validity().cloned(),
                );
                return primitive::array_to_page_integer::<i32, i32>(
                    &array, options, type_, encoding,
                );
            } else if precision <= 18 {
                let values = array
                    .values()
                    .iter()
                    .map(|x| *x as i64)
                    .collect::<Vec<_>>()
                    .into();

                let array = PrimitiveArray::<i64>::new(
                    ArrowDataType::Int64,
                    values,
                    array.validity().cloned(),
                );
                return primitive::array_to_page_integer::<i64, i64>(
                    &array, options, type_, encoding,
                );
            } else {
                let size = decimal_length_from_precision(precision);

                let statistics = if options.has_statistics() {
                    let stats = fixed_size_binary::build_statistics_decimal(
                        array,
                        type_.clone(),
                        size,
                        &options.statistics,
                    );
                    Some(stats)
                } else {
                    None
                };

                let mut values = Vec::<u8>::with_capacity(size * array.len());
                array.values().iter().for_each(|x| {
                    let bytes = &x.to_be_bytes()[16 - size..];
                    values.extend_from_slice(bytes)
                });
                let array = FixedSizeBinaryArray::new(
                    ArrowDataType::FixedSizeBinary(size),
                    values.into(),
                    array.validity().cloned(),
                );
                fixed_size_binary::array_to_page(&array, options, type_, statistics)
            }
        },
        ArrowDataType::UInt128 => {
            let array: &PrimitiveArray<u128> = array.as_any().downcast_ref().unwrap();
            let statistics = if options.has_statistics() {
                let stats = fixed_size_binary::build_statistics_decimal(
                    array,
                    type_.clone(),
                    16,
                    &options.statistics,
                );
                Some(stats)
            } else {
                None
            };
            let array = FixedSizeBinaryArray::new(
                ArrowDataType::FixedSizeBinary(16),
                array.values().clone().try_transmute().unwrap(),
                array.validity().cloned(),
            );
            fixed_size_binary::array_to_page(&array, options, type_, statistics)
        },
        ArrowDataType::Int128 => {
            let array: &PrimitiveArray<i128> = array.as_any().downcast_ref().unwrap();
            let statistics = if options.has_statistics() {
                let stats = fixed_size_binary::build_statistics_decimal(
                    array,
                    type_.clone(),
                    16,
                    &options.statistics,
                );
                Some(stats)
            } else {
                None
            };
            let array = FixedSizeBinaryArray::new(
                ArrowDataType::FixedSizeBinary(16),
                array.values().clone().try_transmute().unwrap(),
                array.validity().cloned(),
            );
            fixed_size_binary::array_to_page(&array, options, type_, statistics)
        },
        ArrowDataType::Extension(ext) => {
            let mut boxed = array.to_boxed();
            assert!(matches!(boxed.dtype(), ArrowDataType::Extension(ext2) if ext2 == ext));
            *boxed.dtype_mut() = ext.inner.clone();
            return array_to_page_simple(boxed.as_ref(), type_, options, encoding);
        },
        other => polars_bail!(nyi = "Writing parquet pages for data type {other:?}"),
    }
    .map(Page::Data)
}

fn array_to_page_nested(
    array: &dyn Array,
    type_: ParquetPrimitiveType,
    nested: &[Nested],
    options: WriteOptions,
    _encoding: Encoding,
) -> PolarsResult<Page> {
    if type_.field_info.repetition == Repetition::Required
        && array.validity().is_some_and(|v| v.unset_bits() > 0)
    {
        polars_bail!(InvalidOperation: "writing a missing value to required parquet column '{}'", type_.field_info.name);
    }

    use ArrowDataType::*;
    match array.dtype().to_storage() {
        Null => {
            let array = Int32Array::new_null(ArrowDataType::Int32, array.len());
            primitive::nested_array_to_page::<i32, i32>(&array, options, type_, nested)
        },
        // Map empty struct to boolean array with same validity.
        Struct(fs) if fs.is_empty() => {
            let array = BooleanArray::new(
                ArrowDataType::Boolean,
                Bitmap::new_zeroed(array.len()),
                array.validity().cloned(),
            );
            boolean::nested_array_to_page(&array, options, type_, nested)
        },
        Boolean => {
            let array = array.as_any().downcast_ref().unwrap();
            boolean::nested_array_to_page(array, options, type_, nested)
        },
        LargeUtf8 => {
            let array =
                polars_compute::cast::cast(array, &LargeBinary, Default::default()).unwrap();
            let array = array.as_any().downcast_ref().unwrap();
            binary::nested_array_to_page::<i64>(array, options, type_, nested)
        },
        LargeBinary => {
            let array = array.as_any().downcast_ref().unwrap();
            binary::nested_array_to_page::<i64>(array, options, type_, nested)
        },
        BinaryView => {
            let array = array.as_any().downcast_ref().unwrap();
            binview::nested_array_to_page(array, options, type_, nested)
        },
        Utf8View => {
            let array = polars_compute::cast::cast(array, &BinaryView, Default::default()).unwrap();
            let array = array.as_any().downcast_ref().unwrap();
            binview::nested_array_to_page(array, options, type_, nested)
        },
        UInt8 => {
            let array = array.as_any().downcast_ref().unwrap();
            primitive::nested_array_to_page::<u8, i32>(array, options, type_, nested)
        },
        UInt16 => {
            let array = array.as_any().downcast_ref().unwrap();
            primitive::nested_array_to_page::<u16, i32>(array, options, type_, nested)
        },
        UInt32 => {
            let array = array.as_any().downcast_ref().unwrap();
            primitive::nested_array_to_page::<u32, i32>(array, options, type_, nested)
        },
        UInt64 => {
            let array = array.as_any().downcast_ref().unwrap();
            primitive::nested_array_to_page::<u64, i64>(array, options, type_, nested)
        },
        Int8 => {
            let array = array.as_any().downcast_ref().unwrap();
            primitive::nested_array_to_page::<i8, i32>(array, options, type_, nested)
        },
        Int16 => {
            let array = array.as_any().downcast_ref().unwrap();
            primitive::nested_array_to_page::<i16, i32>(array, options, type_, nested)
        },
        Int32 | Date32 | Time32(_) => {
            let array = array.as_any().downcast_ref().unwrap();
            primitive::nested_array_to_page::<i32, i32>(array, options, type_, nested)
        },
        Int64 | Date64 | Time64(_) | Timestamp(_, _) | Duration(_) => {
            let array = array.as_any().downcast_ref().unwrap();
            primitive::nested_array_to_page::<i64, i64>(array, options, type_, nested)
        },
        Float16 => {
            let array: &PrimitiveArray<pf16> = array.as_any().downcast_ref().unwrap();
            let statistics = options
                .has_statistics()
                .then(|| build_statistics_float16(array, type_.clone(), &options.statistics));
            let array = FixedSizeBinaryArray::new(
                ArrowDataType::FixedSizeBinary(2),
                array.values().clone().try_transmute().unwrap(),
                array.validity().cloned(),
            );
            fixed_size_binary::nested_array_to_page(&array, options, type_, nested, statistics)
        },
        Float32 => {
            let array = array.as_any().downcast_ref().unwrap();
            primitive::nested_array_to_page::<f32, f32>(array, options, type_, nested)
        },
        Float64 => {
            let array = array.as_any().downcast_ref().unwrap();
            primitive::nested_array_to_page::<f64, f64>(array, options, type_, nested)
        },
        Decimal(precision, _) => {
            let precision = *precision;
            let array = array
                .as_any()
                .downcast_ref::<PrimitiveArray<i128>>()
                .unwrap();
            if precision <= 9 {
                let values = array
                    .values()
                    .iter()
                    .map(|x| *x as i32)
                    .collect::<Vec<_>>()
                    .into();

                let array = PrimitiveArray::<i32>::new(
                    ArrowDataType::Int32,
                    values,
                    array.validity().cloned(),
                );
                primitive::nested_array_to_page::<i32, i32>(&array, options, type_, nested)
            } else if precision <= 18 {
                let values = array
                    .values()
                    .iter()
                    .map(|x| *x as i64)
                    .collect::<Vec<_>>()
                    .into();

                let array = PrimitiveArray::<i64>::new(
                    ArrowDataType::Int64,
                    values,
                    array.validity().cloned(),
                );
                primitive::nested_array_to_page::<i64, i64>(&array, options, type_, nested)
            } else {
                let size = decimal_length_from_precision(precision);

                let statistics = if options.has_statistics() {
                    let stats = fixed_size_binary::build_statistics_decimal(
                        array,
                        type_.clone(),
                        size,
                        &options.statistics,
                    );
                    Some(stats)
                } else {
                    None
                };

                let mut values = Vec::<u8>::with_capacity(size * array.len());
                array.values().iter().for_each(|x| {
                    let bytes = &x.to_be_bytes()[16 - size..];
                    values.extend_from_slice(bytes)
                });
                let array = FixedSizeBinaryArray::new(
                    ArrowDataType::FixedSizeBinary(size),
                    values.into(),
                    array.validity().cloned(),
                );
                fixed_size_binary::nested_array_to_page(&array, options, type_, nested, statistics)
            }
        },
        Decimal256(precision, _) => {
            let precision = *precision;
            let array = array
                .as_any()
                .downcast_ref::<PrimitiveArray<i256>>()
                .unwrap();
            if precision <= 9 {
                let values = array
                    .values()
                    .iter()
                    .map(|x| x.0.as_i32())
                    .collect::<Vec<_>>()
                    .into();

                let array = PrimitiveArray::<i32>::new(
                    ArrowDataType::Int32,
                    values,
                    array.validity().cloned(),
                );
                primitive::nested_array_to_page::<i32, i32>(&array, options, type_, nested)
            } else if precision <= 18 {
                let values = array
                    .values()
                    .iter()
                    .map(|x| x.0.as_i64())
                    .collect::<Vec<_>>()
                    .into();

                let array = PrimitiveArray::<i64>::new(
                    ArrowDataType::Int64,
                    values,
                    array.validity().cloned(),
                );
                primitive::nested_array_to_page::<i64, i64>(&array, options, type_, nested)
            } else if precision <= 38 {
                let size = decimal_length_from_precision(precision);
                let statistics = if options.has_statistics() {
                    let stats = fixed_size_binary::build_statistics_decimal256_with_i128(
                        array,
                        type_.clone(),
                        size,
                        &options.statistics,
                    );
                    Some(stats)
                } else {
                    None
                };

                let mut values = Vec::<u8>::with_capacity(size * array.len());
                array.values().iter().for_each(|x| {
                    let bytes = &x.0.low().to_be_bytes()[16 - size..];
                    values.extend_from_slice(bytes)
                });
                let array = FixedSizeBinaryArray::new(
                    ArrowDataType::FixedSizeBinary(size),
                    values.into(),
                    array.validity().cloned(),
                );
                fixed_size_binary::nested_array_to_page(&array, options, type_, nested, statistics)
            } else {
                let size = 32;
                let array = array
                    .as_any()
                    .downcast_ref::<PrimitiveArray<i256>>()
                    .unwrap();
                let statistics = if options.has_statistics() {
                    let stats = fixed_size_binary::build_statistics_decimal256(
                        array,
                        type_.clone(),
                        size,
                        &options.statistics,
                    );
                    Some(stats)
                } else {
                    None
                };
                let mut values = Vec::<u8>::with_capacity(size * array.len());
                array.values().iter().for_each(|x| {
                    let bytes = &x.to_be_bytes();
                    values.extend_from_slice(bytes)
                });
                let array = FixedSizeBinaryArray::new(
                    ArrowDataType::FixedSizeBinary(size),
                    values.into(),
                    array.validity().cloned(),
                );

                fixed_size_binary::nested_array_to_page(&array, options, type_, nested, statistics)
            }
        },
        Int128 => {
            let array: &PrimitiveArray<i128> = array.as_any().downcast_ref().unwrap();
            // Can't write min/max statistics for signed 128-bit integer, see #25965.
            let mut no_mm_options = options;
            no_mm_options.statistics.min_value = false;
            no_mm_options.statistics.max_value = false;
            let statistics = if no_mm_options.has_statistics() {
                let stats = fixed_size_binary::build_statistics_decimal(
                    array,
                    type_.clone(),
                    16,
                    &no_mm_options.statistics,
                );
                Some(stats)
            } else {
                None
            };
            let array = FixedSizeBinaryArray::new(
                ArrowDataType::FixedSizeBinary(16),
                array.values().clone().try_transmute().unwrap(),
                array.validity().cloned(),
            );
            fixed_size_binary::nested_array_to_page(
                &array,
                no_mm_options,
                type_,
                nested,
                statistics,
            )
        },
        UInt128 => {
            let array: &PrimitiveArray<u128> = array.as_any().downcast_ref().unwrap();
            let statistics = if options.has_statistics() {
                let stats = fixed_size_binary::build_statistics_decimal(
                    array,
                    type_.clone(),
                    16,
                    &options.statistics,
                );
                Some(stats)
            } else {
                None
            };
            let array = FixedSizeBinaryArray::new(
                ArrowDataType::FixedSizeBinary(16),
                array.values().clone().try_transmute().unwrap(),
                array.validity().cloned(),
            );
            fixed_size_binary::nested_array_to_page(&array, options, type_, nested, statistics)
        },
        other => polars_bail!(nyi = "Writing nested parquet pages for data type {other:?}"),
    }
    .map(Page::Data)
}

fn transverse_recursive<T, F: Fn(&ArrowDataType) -> T + Clone>(
    dtype: &ArrowDataType,
    map: F,
    encodings: &mut Vec<T>,
) {
    use arrow::datatypes::PhysicalType::*;
    match dtype.to_physical_type() {
        Null | Boolean | Primitive(_) | Binary | FixedSizeBinary | LargeBinary | Utf8
        | Dictionary(_) | LargeUtf8 | BinaryView | Utf8View => encodings.push(map(dtype)),
        List | FixedSizeList | LargeList => {
            let a = dtype.to_storage();
            if let ArrowDataType::List(inner) = a {
                transverse_recursive(&inner.dtype, map, encodings)
            } else if let ArrowDataType::LargeList(inner) = a {
                transverse_recursive(&inner.dtype, map, encodings)
            } else if let ArrowDataType::FixedSizeList(inner, _) = a {
                transverse_recursive(&inner.dtype, map, encodings)
            } else {
                unreachable!()
            }
        },
        Struct => {
            if let ArrowDataType::Struct(fields) = dtype.to_storage() {
                for field in fields {
                    transverse_recursive(&field.dtype, map.clone(), encodings)
                }
            } else {
                unreachable!()
            }
        },
        Map => {
            if let ArrowDataType::Map(field, _) = dtype.to_storage() {
                if let ArrowDataType::Struct(fields) = field.dtype.to_storage() {
                    for field in fields {
                        transverse_recursive(&field.dtype, map.clone(), encodings)
                    }
                } else {
                    unreachable!()
                }
            } else {
                unreachable!()
            }
        },
        Union => todo!(),
    }
}

/// Transverses the `dtype` up to its (parquet) columns and returns a vector of
/// items based on `map`.
///
/// This is used to assign an [`Encoding`] to every parquet column based on the columns' type (see example)
pub fn transverse<T, F: Fn(&ArrowDataType) -> T + Clone>(dtype: &ArrowDataType, map: F) -> Vec<T> {
    let mut encodings = vec![];
    transverse_recursive(dtype, map, &mut encodings);
    encodings
}

#[cfg(all(test, feature = "content_defined_chunking"))]
mod cdc_tests {
    use super::*;

    #[test]
    fn test_cdc_options_default() {
        let opts = ContentDefinedChunkingOptions::default();
        assert_eq!(opts.min_size, 256 * 1024);
        assert_eq!(opts.avg_size, 512 * 1024);
        assert_eq!(opts.max_size, 1024 * 1024);
    }

    #[test]
    fn test_fixed_row_iter() {
        let iter: Vec<_> = compute_fixed_row_iter(100, 1000, 100).collect();
        assert!(!iter.is_empty());
        // Verify all rows covered
        let total: usize = iter.iter().map(|(_, len)| len).sum();
        assert_eq!(total, 100);
    }

    #[test]
    fn test_cdc_primitive_array() {
        // Need enough data to exceed min chunk size
        // 50000 i64 values = 400KB, enough for CDC with default settings
        let values: Vec<i64> = (0..50000).collect();
        let array = PrimitiveArray::from_vec(values);

        let cdc_opts = ContentDefinedChunkingOptions {
            min_size: 64 * 1024,   // 64 KB
            avg_size: 128 * 1024,  // 128 KB  
            max_size: 256 * 1024,  // 256 KB
        };

        let iter: Vec<_> =
            compute_cdc_row_iter(&array, array.len(), array.len() * 8, cdc_opts).collect();

        // Verify all rows covered
        let total: usize = iter.iter().map(|(_, len)| len).sum();
        assert_eq!(total, 50000);

        // Should have multiple chunks with 400KB data and 128KB avg
        assert!(iter.len() >= 2, "Expected multiple chunks, got {}", iter.len());
    }
}
