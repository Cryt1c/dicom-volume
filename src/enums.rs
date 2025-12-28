#[derive(Clone, Copy)]
pub enum Orientation {
    Axial,
    Coronal,
    Sagittal,
}

#[derive(Default)]
pub enum Interpolation {
    Bilinear(Processor),
    // TODO:
    // Trilinear(Processor),
    // Cubic(Processor),
    #[default]
    None,
}

pub enum Processor {
    CPU,
    // TODO:
    // GPU,
}

#[derive(Default)]
pub enum SortBy {
    #[default]
    ImagePositionPatient,
    TablePosition,
    InstanceNumber,
    None,
}
