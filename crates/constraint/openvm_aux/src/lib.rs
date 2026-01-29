use std::collections::HashMap;

#[derive(Debug, Clone, Default)]
pub struct ConstraintExtInfo {
    /// Column to name mapping. column i will get map to the string "f_i" where f is the field
    /// in the column struct that contains column i
    pub col_to_name: HashMap<usize, String>,
    /// Ranges of columns marked as inputs.
    /// Each tuple contains (`start_index`, `end_index`, `field_name`) where:
    /// - `start_index` is the first column index (inclusive)
    /// - `end_index` is the last column index (exclusive)
    /// - `field_name` is the name of the field
    pub input_ranges: Vec<(usize, usize, String)>,

    /// Ranges of columns marked as outputs.
    /// Each tuple contains (`start_index`, `end_index`, `field_name`) where:
    /// - `start_index` is the first column index (inclusive)
    /// - `end_index` is the last column index (exclusive)
    /// - `field_name` is the name of the field
    pub output_ranges: Vec<(usize, usize, String)>,

    /// Indices of columns marked as selectors.
    /// Each tuple contains (`column_index`, `field_name`) where:
    /// - `column_index` is the index of the selector column
    /// - `field_name` is the name of the field
    pub selector_indices: Vec<(usize, String)>,
}

impl ConstraintExtInfo {
    //In the combined AIR, the adapter AIR columns are arranged first, followed by the core AIR columns. The proc macro gives the 0-offset column indices. For correctness, we need to offset them by the core width.
    //TODO: improve this; This shouldn't have to be invoked manually.
    pub fn adjust_for_offset(self, offset: usize) -> Self {
        Self {
            col_to_name: self.col_to_name.into_iter().map(|(k, v)| (k + offset, v)).collect(),
            input_ranges: self.input_ranges.into_iter().map(|(start, end, name)| (start + offset, end + offset, name)).collect(),
            output_ranges: self.output_ranges.into_iter().map(|(start, end, name)| (start + offset, end + offset, name)).collect(),
            selector_indices: self.selector_indices.into_iter().map(|(col, name)| (col + offset, name)).collect(),
        }
    }
}


// Static Bus Indices used by the OpenVM test harnesses. We should be able to use these to get away from maintaining a bus registry ourselves, and having to tweak a bunch of backend traits.
// Taken from https://github.com/openvm-org/openvm/blob/main/crates/vm/src/arch/testing/mod.rs#L24-L33
//TODO: some buses appear to be missing, for instance the Variable Range Checker Bus. Investigate how the test harness is identifying it (if at all).
pub type BusIndex = u16;
pub const EXECUTION_BUS: BusIndex = 0;
pub const MEMORY_BUS: BusIndex = 1;
pub const POSEIDON2_DIRECT_BUS: BusIndex = 6;
pub const READ_INSTRUCTION_BUS: BusIndex = 8;
pub const BITWISE_OP_LOOKUP_BUS: BusIndex = 9;
pub const BYTE_XOR_BUS: BusIndex = 10;
pub const RANGE_TUPLE_CHECKER_BUS: BusIndex = 11;
pub const MEMORY_MERKLE_BUS: BusIndex = 12;
pub const RANGE_CHECKER_BUS: BusIndex = 4;