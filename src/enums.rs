/// The three standard medical viewing axes of a volume.
///
/// The volume is stored as a (depth, height, width) array; each orientation
/// selects which of these dimensions the slice index addresses.
#[derive(Clone, Copy)]
pub enum Orientation {
    /// Slice along the first dimension of the volume (head-to-foot view).
    /// Not interpolated, as it matches the acquisition plane.
    Axial = 0,
    /// Front-to-back view through the second dimension.
    Coronal = 1,
    /// Side view through the third dimension.
    Sagittal = 2,
}

/// Resampling applied when extracting a slice.
#[derive(Clone, Copy, Default)]
pub enum Interpolation {
    /// Bilinear resampling to isotropic dimensions based on the voxel spacing.
    Linear,
    /// No resampling: return the raw slice.
    #[default]
    None,
}

/// Attribute used to establish the ordering of slices within the volume.
#[derive(Default)]
pub enum SortBy {
    /// Sort by the Image Position Patient attribute, descending along the
    /// slice normal.
    #[default]
    ImagePositionPatient,
    /// Sort by the Table Position attribute.
    TablePosition,
    /// Sort by the Instance Number attribute.
    InstanceNumber,
    /// Keep the order in which the files or objects were provided.
    None,
}
