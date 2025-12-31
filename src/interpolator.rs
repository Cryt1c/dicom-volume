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
    pub(crate) fn bilinear_interpolate(slice: &ArrayView2<u16>, y: f32, x: f32) -> u16 {
        let (height, width) = slice.dim();

        let y0 = y.floor() as usize;
        let x0 = x.floor() as usize;
        let y1 = (y0 + 1).min(height - 1);
        let x1 = (x0 + 1).min(width - 1);

        let dy = y - y0 as f32;
        let dx = x - x0 as f32;
        let one_minus_dx = 1.0 - dx;
        let one_minus_dy = 1.0 - dy;

        let v00 = slice[[y0, x0]] as f32;
        let v01 = slice[[y0, x1]] as f32;
        let v10 = slice[[y1, x0]] as f32;
        let v11 = slice[[y1, x1]] as f32;

        let v0 = v00.mul_add(one_minus_dx, v01 as f32 * dx);
        let v1 = v10.mul_add(one_minus_dx, v11 as f32 * dx);

        v0.mul_add(one_minus_dy, v1 * dy) as u16
    }
}

#[cfg(test)]
mod tests {
    use ndarray::Array2;

    use super::*;

    #[test]
    fn test_get_isotropic_dimensions_uniform_spacing() {
        let spacing = (1.0, 1.0, 1.0);
        let original_dim = (100, 200, 300);

        let result = Interpolator::get_isotropic_dimensions(spacing, original_dim);

        assert_eq!(result, (100, 200, 300));
    }

    #[test]
    fn test_get_isotropic_dimensions_z_larger_spacing() {
        let spacing = (1.0, 1.0, 2.0);
        let original_dim = (100, 200, 300);

        let result = Interpolator::get_isotropic_dimensions(spacing, original_dim);

        // min_spacing = 1.0
        // new_z = 100 * 2.0 / 1.0 = 200
        // new_y = 200 * 1.0 / 1.0 = 200
        // new_x = 300 * 1.0 / 1.0 = 300
        assert_eq!(result, (200, 200, 300));
    }

    #[test]
    fn test_get_isotropic_dimensions_x_smaller_spacing() {
        let spacing = (0.5, 1.0, 1.0);
        let original_dim = (100, 200, 300);

        let result = Interpolator::get_isotropic_dimensions(spacing, original_dim);

        // min_spacing = 0.5
        // new_z = 100 * 1.0 / 0.5 = 200
        // new_y = 200 * 1.0 / 0.5 = 400
        // new_x = 300 * 0.5 / 0.5 = 300
        assert_eq!(result, (200, 400, 300));
    }

    #[test]
    fn test_get_isotropic_dimensions_mixed_spacing() {
        let spacing = (2.0, 1.0, 3.0);
        let original_dim = (60, 120, 180);

        let result = Interpolator::get_isotropic_dimensions(spacing, original_dim);

        // min_spacing = 1.0
        // new_z = 60 * 3.0 / 1.0 = 180
        // new_y = 120 * 1.0 / 1.0 = 120
        // new_x = 180 * 2.0 / 1.0 = 360
        assert_eq!(result, (180, 120, 360));
    }

    #[test]
    fn test_bilinear_interpolate_at_corner() {
        let data =
            Array2::from_shape_vec((3, 3), vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0])
                .unwrap();
        let view = data.view();

        let result = Interpolator::bilinear_interpolate(&view, 0.0, 0.0);

        assert_eq!(result, 1.0);
    }

    #[test]
    fn test_bilinear_interpolate_at_exact_point() {
        let data =
            Array2::from_shape_vec((3, 3), vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0])
                .unwrap();
        let view = data.view();

        let result = Interpolator::bilinear_interpolate(&view, 1.0, 1.0);

        assert_eq!(result, 5.0);
    }

    #[test]
    fn test_bilinear_interpolate_midpoint() {
        let data = Array2::from_shape_vec((2, 2), vec![0.0, 2.0, 4.0, 6.0]).unwrap();
        let view = data.view();

        let result = Interpolator::bilinear_interpolate(&view, 0.5, 0.5);

        // Average of all four corners
        assert_eq!(result, 3.0);
    }

    #[test]
    fn test_bilinear_interpolate_horizontal_midpoint() {
        let data = Array2::from_shape_vec((2, 2), vec![0.0, 4.0, 0.0, 4.0]).unwrap();
        let view = data.view();

        let result = Interpolator::bilinear_interpolate(&view, 0.0, 0.5);

        // Midpoint between 0.0 and 4.0
        assert_eq!(result, 2.0);
    }

    #[test]
    fn test_bilinear_interpolate_vertical_midpoint() {
        let data = Array2::from_shape_vec((2, 2), vec![0.0, 0.0, 4.0, 4.0]).unwrap();
        let view = data.view();

        let result = Interpolator::bilinear_interpolate(&view, 0.5, 0.0);

        // Midpoint between 0.0 and 4.0
        assert_eq!(result, 2.0);
    }

    #[test]
    fn test_bilinear_interpolate_quarter_point() {
        let data = Array2::from_shape_vec((2, 2), vec![0.0, 4.0, 8.0, 12.0]).unwrap();
        let view = data.view();

        let result = Interpolator::bilinear_interpolate(&view, 0.25, 0.25);

        // v0 = 0.0 * 0.75 + 4.0 * 0.25 = 1.0
        // v1 = 8.0 * 0.75 + 12.0 * 0.25 = 9.0
        // result = 1.0 * 0.75 + 9.0 * 0.25 = 3.0
        assert_eq!(result, 3.0);
    }

    #[test]
    fn test_bilinear_interpolate_boundary_clamping() {
        let data =
            Array2::from_shape_vec((3, 3), vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0])
                .unwrap();
        let view = data.view();

        // Test at the bottom-right corner (should clamp to last valid indices)
        let result = Interpolator::bilinear_interpolate(&view, 2.5, 2.5);

        // Should clamp to (2, 2) which is 9.0
        assert_eq!(result, 9.0);
    }

    #[test]
    fn test_bilinear_interpolate_single_pixel() {
        let data = Array2::from_shape_vec((1, 1), vec![42.0]).unwrap();
        let view = data.view();

        let result = Interpolator::bilinear_interpolate(&view, 0.0, 0.0);

        assert_eq!(result, 42.0);
    }

    #[test]
    fn test_bilinear_interpolate_fractional_coordinates() {
        let data =
            Array2::from_shape_vec((3, 3), vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0])
                .unwrap();
        let view = data.view();

        let result = Interpolator::bilinear_interpolate(&view, 0.75, 1.25);

        // y0=0, y1=1, x0=1, x1=2
        // dy=0.75, dx=0.25
        // v00=2.0, v01=3.0, v10=5.0, v11=6.0
        // v0 = 2.0 * 0.75 + 3.0 * 0.25 = 2.25
        // v1 = 5.0 * 0.75 + 6.0 * 0.25 = 5.25
        // result = 2.25 * 0.25 + 5.25 * 0.75 = 4.5
        assert!((result - 4.5).abs() < 1e-6);
    }
}
