use ndarray::ArrayView2;

pub(crate) struct Interpolator;

impl Interpolator {
    pub(crate) fn get_isotropic_dimensions(
        spacing: (f32, f32, f32),
        original_dim: (usize, usize, usize),
    ) -> (u32, u32, u32) {
        let (x_spacing, y_spacing, z_spacing) = spacing;
        let min_spacing = x_spacing.min(y_spacing).min(z_spacing);
        let inv_min_spacing = 1.0 / min_spacing; // Multiply instead of divide

        // original_dim is (depth, height, width) corresponding to (z, y, x)
        let new_x = (original_dim.2 as f32 * x_spacing * inv_min_spacing) as u32;
        let new_y = (original_dim.1 as f32 * y_spacing * inv_min_spacing) as u32;
        let new_z = (original_dim.0 as f32 * z_spacing * inv_min_spacing) as u32;

        (new_z, new_y, new_x)
    }

    #[inline]
    pub(crate) fn bilinear_interpolate(slice: &ArrayView2<f32>, y: f32, x: f32) -> f32 {
        let (height, width) = slice.dim();

        let y0 = y.floor() as usize;
        let x0 = x.floor() as usize;
        let y1 = (y0 + 1).min(height - 1);
        let x1 = (x0 + 1).min(width - 1);

        let dy = y - y0 as f32;
        let dx = x - x0 as f32;
        let one_minus_dx = 1.0 - dx;
        let one_minus_dy = 1.0 - dy;

        let v00 = slice[[y0, x0]];
        let v01 = slice[[y0, x1]];
        let v10 = slice[[y1, x0]];
        let v11 = slice[[y1, x1]];

        let v0 = v00.mul_add(one_minus_dx, v01 * dx);
        let v1 = v10.mul_add(one_minus_dx, v11 * dx);

        v0.mul_add(one_minus_dy, v1 * dy)
    }
}
