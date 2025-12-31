#[derive(Clone, Copy)]
pub enum Orientation {
    Axial = 0,
    Coronal = 1,
    Sagittal = 2,
}

#[derive(Default)]
pub enum Interpolation {
    Linear,
    #[default]
    None,
}

#[derive(Default)]
pub enum SortBy {
    #[default]
    ImagePositionPatient,
    TablePosition,
    InstanceNumber,
    None,
}
