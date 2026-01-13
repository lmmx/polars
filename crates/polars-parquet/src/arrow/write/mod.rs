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
#[cfg(feature = "content_defined_chunking")]
mod cdc;
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
pub use cdc::{CdcPageData, EncodedColumnData, PageValueRange, ValueBoundaryTracker};
pub use nested::{num_values, write_rep_and_def};
pub use pages::{to_leaves, to_nested, to_parquet_leaves};
use polars_utils::float16::pf16;
use polars_utils::pl_str::PlSmallStr;
#[cfg(feature = "content_defined_chunking")]
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
    
    // Handle dictionary types first (unchanged)
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
        if matches!(nested.first(), Some(Nested::Primitive(_))) {
            if let Some(result) =
                encode_as_dictionary_optional(primitive_array, nested, type_.clone(), options)
            {
                return result;
            }
        }
        encoding = Encoding::Plain;
    }

    let nested = nested.to_vec();
    let number_of_rows = nested[0].len();
    let byte_size = estimated_bytes_size(primitive_array);

    const DEFAULT_PAGE_SIZE: usize = 1024 * 1024;
    let max_page_size = options.data_page_size.unwrap_or(DEFAULT_PAGE_SIZE);
    let max_page_size = max_page_size.min(2usize.pow(31) - 2usize.pow(25));

    // Route to CDC or fixed-size chunking
    #[cfg(feature = "content_defined_chunking")]
    if let Some(cdc_options) = options.content_defined_chunking {
        return array_to_pages_with_cdc(
            primitive_array,
            type_,
            &nested,
            options,
            encoding,
            cdc_options,
        );
    }

    // Fixed-size chunking (existing approach)
    let row_iter = compute_fixed_row_iter(number_of_rows, byte_size, max_page_size);
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

/// CDC-based page splitting: determine row boundaries via CDC, then use existing encoding.
/// This approach runs CDC on a preview of encoded bytes to find good split points,
/// then uses the standard slice-and-encode path to create valid pages.
#[cfg(feature = "content_defined_chunking")]
fn array_to_pages_with_cdc(
    primitive_array: &dyn Array,
    type_: ParquetPrimitiveType,
    nested: &[Nested],
    options: WriteOptions,
    encoding: Encoding,
    cdc_options: ContentDefinedChunkingOptions,
) -> PolarsResult<DynIter<'static, PolarsResult<Page>>> {
    let nested = nested.to_vec();
    let number_of_rows = nested[0].len();

    if number_of_rows == 0 {
        return Ok(DynIter::new(std::iter::empty()));
    }

    // Get row boundaries from CDC analysis
    let row_ranges = compute_cdc_row_ranges(primitive_array, number_of_rows, cdc_options);

    let primitive_array = primitive_array.to_boxed();

    // Use the same slice-then-encode approach as the non-CDC path
    let pages = row_ranges.into_iter().map(move |(offset, length)| {
        let mut sliced_array = primitive_array.clone();
        let mut sliced_nested = nested.clone();
        slice_parquet_array(sliced_array.as_mut(), &mut sliced_nested, offset, length);

        array_to_page(
            sliced_array.as_ref(),
            type_.clone(),
            &sliced_nested,
            options,
            encoding,
        )
    });

    Ok(DynIter::new(pages))
}

/// Compute row ranges using CDC on encoded byte representation.
/// For types where we can predict the encoding, we analyze those bytes.
/// For others, we fall back to fixed-size chunking.
#[cfg(feature = "content_defined_chunking")]
fn compute_cdc_row_ranges(
    array: &dyn Array,
    number_of_rows: usize,
    cdc_options: ContentDefinedChunkingOptions,
) -> Vec<(usize, usize)> {
    use arrow::datatypes::PhysicalType;
    use fastcdc::v2020::FastCDC;

    if number_of_rows == 0 {
        return vec![];
    }

    // Try to get byte data suitable for CDC
    let (bytes, bytes_per_row) = get_cdc_byte_data_for_encoding(array);

    let Some(bytes) = bytes else {
        // Fall back to single page if we can't get usable bytes
        return vec![(0, number_of_rows)];
    };

    if bytes.len() < cdc_options.min_size {
        // Data too small for CDC, single page
        return vec![(0, number_of_rows)];
    }

    // Run FastCDC
    let chunker = FastCDC::new(
        &bytes,
        cdc_options.min_size as u32,
        cdc_options.avg_size as u32,
        cdc_options.max_size as u32,
    );

    let chunks: Vec<_> = chunker.collect();

    if chunks.is_empty() {
        return vec![(0, number_of_rows)];
    }

    // Convert byte boundaries to row boundaries
    let mut row_ranges = Vec::with_capacity(chunks.len());
    let mut current_row = 0usize;

    for chunk in chunks {
        let chunk_end_byte = chunk.offset + chunk.length;
        
        // Calculate end row for this chunk
        let end_row = if let Some(bpr) = bytes_per_row {
            // Fixed-size elements: direct calculation
            (chunk_end_byte / bpr).min(number_of_rows)
        } else {
            // Variable-size: estimate based on chunk position
            ((chunk_end_byte as f64 / bytes.len() as f64) * number_of_rows as f64).ceil() as usize
        };
        
        let end_row = end_row.min(number_of_rows);
        let length = end_row.saturating_sub(current_row);
        
        if length > 0 {
            row_ranges.push((current_row, length));
            current_row = end_row;
        }
    }

    // Ensure we cover all rows
    if current_row < number_of_rows {
        row_ranges.push((current_row, number_of_rows - current_row));
    }

    // Merge any tiny trailing chunks
    if row_ranges.len() > 1 {
        if let Some(last) = row_ranges.last() {
            if last.1 < 100 {
                let last = row_ranges.pop().unwrap();
                if let Some(prev) = row_ranges.last_mut() {
                    prev.1 += last.1;
                } else {
                    row_ranges.push(last);
                }
            }
        }
    }

    row_ranges
}

/// Extract byte data for CDC analysis that approximates what will be encoded.
/// Returns (bytes, bytes_per_row) where bytes_per_row is Some for fixed-size types.
#[cfg(feature = "content_defined_chunking")]
fn get_cdc_byte_data_for_encoding(array: &dyn Array) -> (Option<Vec<u8>>, Option<usize>) {
    use arrow::datatypes::PhysicalType;

    match array.dtype().to_physical_type() {
        PhysicalType::Primitive(ptype) => {
            use arrow::types::PrimitiveType;
            
            // For primitives, the encoded size depends on the Parquet type mapping
            let parquet_elem_size = match ptype {
                // These get promoted to i32 in Parquet
                PrimitiveType::Int8 | PrimitiveType::UInt8 |
                PrimitiveType::Int16 | PrimitiveType::UInt16 |
                PrimitiveType::Int32 | PrimitiveType::UInt32 => 4,
                // These stay as i64
                PrimitiveType::Int64 | PrimitiveType::UInt64 => 8,
                // Floats stay same size
                PrimitiveType::Float32 => 4,
                PrimitiveType::Float64 => 8,
                // Others
                PrimitiveType::Int128 | PrimitiveType::UInt128 => 16,
                _ => return (None, None),
            };

            // Extract the actual values for CDC
            macro_rules! extract_and_widen {
                ($arr_type:ty, $parquet_type:ty) => {{
                    if let Some(arr) = array.as_any().downcast_ref::<PrimitiveArray<$arr_type>>() {
                        let mut buffer = Vec::with_capacity(arr.len() * std::mem::size_of::<$parquet_type>());
                        for val in arr.values().iter() {
                            let widened = *val as $parquet_type;
                            buffer.extend_from_slice(&widened.to_le_bytes());
                        }
                        return (Some(buffer), Some(parquet_elem_size));
                    }
                }};
            }

            // Handle type widening to match Parquet encoding
            extract_and_widen!(i8, i32);
            extract_and_widen!(i16, i32);
            extract_and_widen!(i32, i32);
            extract_and_widen!(u8, i32);
            extract_and_widen!(u16, i32);
            extract_and_widen!(u32, i32);
            extract_and_widen!(i64, i64);
            extract_and_widen!(u64, i64);
            
            // Floats: just copy bytes directly
            if let Some(arr) = array.as_any().downcast_ref::<PrimitiveArray<f32>>() {
                let bytes: &[u8] = bytemuck::cast_slice(arr.values().as_slice());
                return (Some(bytes.to_vec()), Some(4));
            }
            if let Some(arr) = array.as_any().downcast_ref::<PrimitiveArray<f64>>() {
                let bytes: &[u8] = bytemuck::cast_slice(arr.values().as_slice());
                return (Some(bytes.to_vec()), Some(8));
            }

            (None, None)
        }
        
        PhysicalType::BinaryView | PhysicalType::Utf8View => {
            // For view types, encode as length-prefixed strings (matching Parquet PLAIN encoding)
            // This is the KEY fix: CDC sees the actual string content, not the views
            let mut buffer = Vec::new();
            
            if let Some(arr) = array.as_any().downcast_ref::<BinaryViewArray>() {
                for val in arr.iter() {
                    if let Some(bytes) = val {
                        buffer.extend_from_slice(&(bytes.len() as u32).to_le_bytes());
                        buffer.extend_from_slice(bytes);
                    }
                }
                return (Some(buffer), None); // Variable size
            }
            if let Some(arr) = array.as_any().downcast_ref::<Utf8ViewArray>() {
                for val in arr.iter() {
                    if let Some(s) = val {
                        let bytes = s.as_bytes();
                        buffer.extend_from_slice(&(bytes.len() as u32).to_le_bytes());
                        buffer.extend_from_slice(bytes);
                    }
                }
                return (Some(buffer), None); // Variable size
            }
            
            (None, None)
        }
        
        PhysicalType::LargeBinary | PhysicalType::LargeUtf8 => {
            // Similar to view types
            let mut buffer = Vec::new();
            
            if let Some(arr) = array.as_any().downcast_ref::<BinaryArray<i64>>() {
                for val in arr.iter() {
                    if let Some(bytes) = val {
                        buffer.extend_from_slice(&(bytes.len() as u32).to_le_bytes());
                        buffer.extend_from_slice(bytes);
                    }
                }
                return (Some(buffer), None);
            }
            if let Some(arr) = array.as_any().downcast_ref::<Utf8Array<i64>>() {
                for val in arr.iter() {
                    if let Some(s) = val {
                        let bytes = s.as_bytes();
                        buffer.extend_from_slice(&(bytes.len() as u32).to_le_bytes());
                        buffer.extend_from_slice(bytes);
                    }
                }
                return (Some(buffer), None);
            }
            
            (None, None)
        }
        
        _ => (None, None),
    }
}

/// Encode an array to bytes for CDC processing.
/// Returns EncodedColumnData with the encoded bytes and value boundary tracking.
#[cfg(feature = "content_defined_chunking")]
fn encode_array_for_cdc(
    array: &dyn Array,
    type_: ParquetPrimitiveType,
    nested: &[Nested],
    options: WriteOptions,
    encoding: Encoding,
) -> PolarsResult<cdc::EncodedColumnData> {
    let dtype = array.dtype();
    let num_values = array.len();

    // For simple (non-nested) case with primitive types
    if nested.len() == 1 {
        match dtype.to_physical_type() {
            PhysicalType::Primitive(ptype) => {
                let elem_size = primitive_element_size(ptype);
                let (data, tracker) = encode_primitive_for_cdc(array, elem_size)?;

                // Encode definition levels if nullable
                let def_levels = if array.validity().is_some() {
                    Some(encode_validity_as_def_levels(array.validity(), num_values))
                } else {
                    None
                };

                return Ok(EncodedColumnData {
                    data,
                    num_values,
                    encoding,
                    def_levels,
                    rep_levels: None,
                    value_boundaries: tracker,
                });
            }
            PhysicalType::BinaryView | PhysicalType::Utf8View => {
                let (data, tracker) = encode_binview_for_cdc(array)?;

                let def_levels = if array.validity().is_some() {
                    Some(encode_validity_as_def_levels(array.validity(), num_values))
                } else {
                    None
                };

                return Ok(EncodedColumnData {
                    data,
                    num_values,
                    encoding,
                    def_levels,
                    rep_levels: None,
                    value_boundaries: tracker,
                });
            }
            PhysicalType::LargeBinary | PhysicalType::LargeUtf8 => {
                let (data, tracker) = encode_binary_for_cdc(array)?;

                let def_levels = if array.validity().is_some() {
                    Some(encode_validity_as_def_levels(array.validity(), num_values))
                } else {
                    None
                };

                return Ok(EncodedColumnData {
                    data,
                    num_values,
                    encoding,
                    def_levels,
                    rep_levels: None,
                    value_boundaries: tracker,
                });
            }
            _ => {}
        }
    }

    // Fallback: encode via existing path and wrap result
    // This handles nested types and edge cases
    encode_via_existing_path(array, type_, nested, options, encoding)
}

#[cfg(feature = "content_defined_chunking")]
fn primitive_element_size(ptype: arrow::types::PrimitiveType) -> usize {
    use arrow::types::PrimitiveType;
    match ptype {
        PrimitiveType::Int8 | PrimitiveType::UInt8 => 1,
        PrimitiveType::Int16 | PrimitiveType::UInt16 | PrimitiveType::Float16 => 2,
        PrimitiveType::Int32 | PrimitiveType::UInt32 | PrimitiveType::Float32 => 4,
        PrimitiveType::Int64 | PrimitiveType::UInt64 | PrimitiveType::Float64 => 8,
        PrimitiveType::Int128 | PrimitiveType::UInt128 => 16,
        PrimitiveType::Int256 => 32,
        PrimitiveType::DaysMs | PrimitiveType::MonthDayMillis => 8,
        PrimitiveType::MonthDayNano => 16,
    }
}

/// Encode primitive array to bytes with value boundary tracking.
#[cfg(feature = "content_defined_chunking")]
fn encode_primitive_for_cdc(
    array: &dyn Array,
    elem_size: usize,
) -> PolarsResult<(Vec<u8>, cdc::ValueBoundaryTracker)> {
    // Get raw bytes from the primitive array's values buffer
    macro_rules! encode_primitive {
        ($native_type:ty) => {{
            if let Some(arr) = array.as_any().downcast_ref::<PrimitiveArray<$native_type>>() {
                let values = arr.values();
                let bytes: &[u8] = bytemuck::cast_slice(values.as_slice());
                let tracker = cdc::ValueBoundaryTracker::for_fixed_size(arr.len(), elem_size);
                return Ok((bytes.to_vec(), tracker));
            }
        }};
    }

    encode_primitive!(i8);
    encode_primitive!(i16);
    encode_primitive!(i32);
    encode_primitive!(i64);
    encode_primitive!(i128);
    encode_primitive!(u8);
    encode_primitive!(u16);
    encode_primitive!(u32);
    encode_primitive!(u64);
    encode_primitive!(u128);
    encode_primitive!(f32);
    encode_primitive!(f64);

    // Fallback for other primitive types
    let num_values = array.len();
    let tracker = cdc::ValueBoundaryTracker::for_fixed_size(num_values, elem_size);
    Ok((Vec::new(), tracker))
}

/// Encode BinaryView/Utf8View to length-prefixed bytes for CDC.
/// This produces the ACTUAL bytes that Parquet will encode, not the view structs.
#[cfg(feature = "content_defined_chunking")]
fn encode_binview_for_cdc(
    array: &dyn Array,
) -> PolarsResult<(Vec<u8>, cdc::ValueBoundaryTracker)> {
    let mut buffer = Vec::new();
    let mut tracker = cdc::ValueBoundaryTracker::new();

    if let Some(arr) = array.as_any().downcast_ref::<Utf8ViewArray>() {
        for value in arr.iter() {
            if let Some(bytes) = value {
                // Length-prefixed encoding (matching Parquet PLAIN encoding for binary)
                buffer.extend_from_slice(&(bytes.len() as u32).to_le_bytes());
                buffer.extend_from_slice(bytes.as_bytes());
            }
            // For nulls, we don't write anything - handled by def_levels
            tracker.record(buffer.len());
        }
    } else if let Some(arr) = array.as_any().downcast_ref::<BinaryViewArray>() {
        for value in arr.iter() {
            if let Some(bytes) = value {
                buffer.extend_from_slice(&(bytes.len() as u32).to_le_bytes());
                buffer.extend_from_slice(bytes);
            }
            tracker.record(buffer.len());
        }
    }

    Ok((buffer, tracker))
}

/// Encode Binary/LargeBinary to length-prefixed bytes for CDC.
#[cfg(feature = "content_defined_chunking")]
fn encode_binary_for_cdc(
    array: &dyn Array,
) -> PolarsResult<(Vec<u8>, cdc::ValueBoundaryTracker)> {
    let mut buffer = Vec::new();
    let mut tracker = cdc::ValueBoundaryTracker::new();

    if let Some(arr) = array.as_any().downcast_ref::<BinaryArray<i64>>() {
        for value in arr.iter() {
            if let Some(bytes) = value {
                buffer.extend_from_slice(&(bytes.len() as u32).to_le_bytes());
                buffer.extend_from_slice(bytes);
            }
            tracker.record(buffer.len());
        }
    } else if let Some(arr) = array.as_any().downcast_ref::<BinaryArray<i32>>() {
        for value in arr.iter() {
            if let Some(bytes) = value {
                buffer.extend_from_slice(&(bytes.len() as u32).to_le_bytes());
                buffer.extend_from_slice(bytes);
            }
            tracker.record(buffer.len());
        }
    } else if let Some(arr) = array.as_any().downcast_ref::<Utf8Array<i64>>() {
        for value in arr.iter() {
            if let Some(s) = value {
                let bytes = s.as_bytes();
                buffer.extend_from_slice(&(bytes.len() as u32).to_le_bytes());
                buffer.extend_from_slice(bytes);
            }
            tracker.record(buffer.len());
        }
    } else if let Some(arr) = array.as_any().downcast_ref::<Utf8Array<i32>>() {
        for value in arr.iter() {
            if let Some(s) = value {
                let bytes = s.as_bytes();
                buffer.extend_from_slice(&(bytes.len() as u32).to_le_bytes());
                buffer.extend_from_slice(bytes);
            }
            tracker.record(buffer.len());
        }
    }

    Ok((buffer, tracker))
}

/// Encode validity bitmap as RLE-encoded definition levels.
#[cfg(feature = "content_defined_chunking")]
fn encode_validity_as_def_levels(
    validity: Option<&Bitmap>,
    num_values: usize,
) -> Vec<u8> {
    // Simple approach: create def levels buffer
    // 1 = defined, 0 = null
    let mut def_levels = Vec::with_capacity(num_values);

    if let Some(bitmap) = validity {
        for i in 0..num_values {
            def_levels.push(if bitmap.get_bit(i) { 1u8 } else { 0u8 });
        }
    } else {
        def_levels.extend(std::iter::repeat(1u8).take(num_values));
    }

    // RLE encode the definition levels
    let mut buffer = Vec::new();
    // Write using the existing utilities
    // This is a simplified version - full implementation would use proper RLE
    buffer.extend_from_slice(&def_levels);
    buffer
}

/// Fallback: encode via existing single-page path and wrap.
#[cfg(feature = "content_defined_chunking")]
fn encode_via_existing_path(
    array: &dyn Array,
    type_: ParquetPrimitiveType,
    nested: &[Nested],
    options: WriteOptions,
    encoding: Encoding,
) -> PolarsResult<cdc::EncodedColumnData> {
    // Create a single page using existing logic
    let page = array_to_page(array, type_, nested, options, encoding)?;

    match page {
        Page::Data(data_page) => {
            let num_values = data_page.num_values();
            let buffer = data_page.buffer().to_vec();

            Ok(cdc::EncodedColumnData {
                data: buffer,
                num_values,
                encoding,
                def_levels: None, // Already included in buffer
                rep_levels: None,
                value_boundaries: cdc::ValueBoundaryTracker::for_fixed_size(num_values, 1),
            })
        }
        Page::Dict(_) => {
            polars_bail!(InvalidOperation: "CDC not supported for dictionary pages")
        }
    }
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
