use crate::arena::PackedVertex;

/// Generate a unit cube (12 triangles, 24 unique vertices for flat normals, 36 indices).
pub fn generate_cube() -> (Vec<PackedVertex>, Vec<u32>) {
    // Six faces, each 4 vertices (unique per face for hard normals).
    let face_data: &[([f32; 3], [f32; 3])] = &[
        // pos offset, normal
        ([0.0, 0.0, 0.5], [0.0, 0.0, 1.0]),   // +Z
        ([0.0, 0.0, -0.5], [0.0, 0.0, -1.0]), // -Z
        ([0.5, 0.0, 0.0], [1.0, 0.0, 0.0]),   // +X
        ([-0.5, 0.0, 0.0], [-1.0, 0.0, 0.0]), // -X
        ([0.0, 0.5, 0.0], [0.0, 1.0, 0.0]),   // +Y
        ([0.0, -0.5, 0.0], [0.0, -1.0, 0.0]), // -Y
    ];

    // Local face quad corners (in tangent space, scaled to ±0.5)
    let quad_corners: &[[f32; 2]] = &[[-0.5, -0.5], [0.5, -0.5], [0.5, 0.5], [-0.5, 0.5]];
    let quad_uvs: &[[f32; 2]] = &[[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]];
    let quad_indices: &[u32] = &[0, 1, 2, 0, 2, 3];

    let mut vertices = Vec::with_capacity(24);
    let mut indices = Vec::with_capacity(36);

    for (face_idx, (center, normal)) in face_data.iter().enumerate() {
        let base = (face_idx * 4) as u32;
        for qi in quad_indices {
            indices.push(base + qi);
        }

        // Build two tangent axes from the normal
        let n = glam::Vec3::from(*normal);
        let (t, b) = tangent_frame(n);

        for (corner, uv) in quad_corners.iter().zip(quad_uvs.iter()) {
            let local = t * corner[0] + b * corner[1];
            let pos = glam::Vec3::from(*center) + local;
            vertices.push(PackedVertex::new([pos.x, pos.y, pos.z], *normal, *uv));
        }
    }

    (vertices, indices)
}

/// Generate a UV sphere with `stacks` latitude bands and `slices` longitude segments.
pub fn generate_sphere(stacks: u32, slices: u32) -> (Vec<PackedVertex>, Vec<u32>) {
    use std::f32::consts::PI;

    let mut vertices = Vec::new();
    let mut indices = Vec::new();

    for stack in 0..=stacks {
        let phi = PI * stack as f32 / stacks as f32; // 0..π
        let sin_phi = phi.sin();
        let cos_phi = phi.cos();

        for slice in 0..=slices {
            let theta = 2.0 * PI * slice as f32 / slices as f32; // 0..2π
            let x = sin_phi * theta.cos();
            let y = cos_phi;
            let z = sin_phi * theta.sin();

            let u = slice as f32 / slices as f32;
            let v = stack as f32 / stacks as f32;

            vertices.push(PackedVertex::new(
                [x * 0.5, y * 0.5, z * 0.5], // radius 0.5
                [x, y, z],                   // normalized position = outward normal
                [u, v],
            ));
        }
    }

    for stack in 0..stacks {
        for slice in 0..slices {
            let row_len = slices + 1;
            let tl = stack * row_len + slice;
            let tr = stack * row_len + slice + 1;
            let bl = (stack + 1) * row_len + slice;
            let br = (stack + 1) * row_len + slice + 1;

            indices.push(tl);
            indices.push(bl);
            indices.push(tr);
            indices.push(tr);
            indices.push(bl);
            indices.push(br);
        }
    }

    (vertices, indices)
}

fn tangent_frame(n: glam::Vec3) -> (glam::Vec3, glam::Vec3) {
    let up = if n.y.abs() < 0.9 {
        glam::Vec3::Y
    } else {
        glam::Vec3::X
    };
    let t = up.cross(n).normalize();
    let b = n.cross(t);
    (t, b)
}
