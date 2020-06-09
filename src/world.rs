use std::f32::consts::PI;
use std::ops::{Index, IndexMut, Mul};

use sdl2::{pixels::Color, rect::Point, render::Canvas, video::Window};

const FOV: f32 = 90.0;
const FAR: f32 = 1000.0;
const NEAR: f32 = 0.1;

pub(crate) struct World {
    mesh_cube: Mesh,
    mat_proj: Mat4x4,
}

impl World {
    pub(crate) fn new() -> Self {
        let mesh_cube = Mesh::new(vec![
            // SOUTH
            Triangle::from_points(0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 1.0, 1.0, 0.0),
            Triangle::from_points(0.0, 0.0, 0.0, 1.0, 1.0, 0.0, 1.0, 0.0, 0.0),
            // EAST
            Triangle::from_points(1.0, 0.0, 0.0, 1.0, 1.0, 0.0, 1.0, 1.0, 1.0),
            Triangle::from_points(1.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 0.0, 1.0),
            // NORTH
            Triangle::from_points(1.0, 0.0, 1.0, 1.0, 1.0, 1.0, 0.0, 1.0, 1.0),
            Triangle::from_points(1.0, 0.0, 1.0, 0.0, 1.0, 1.0, 0.0, 0.0, 1.0),
            // WEST
            Triangle::from_points(0.0, 0.0, 1.0, 0.0, 1.0, 1.0, 0.0, 1.0, 0.0),
            Triangle::from_points(0.0, 0.0, 1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0),
            // TOP
            Triangle::from_points(0.0, 1.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0),
            Triangle::from_points(0.0, 1.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0),
            // BOTTOM
            Triangle::from_points(1.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0),
            Triangle::from_points(1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0),
        ]);
        let mut mat_proj = Mat4x4::default();
        mat_proj[2][2] = FAR / (FAR - NEAR);
        mat_proj[3][2] = (-FAR * NEAR) / (FAR - NEAR);
        mat_proj[2][3] = 1.0;
        mat_proj[3][3] = 0.0;
        World {
            mesh_cube,
            mat_proj,
        }
    }

    pub(crate) fn do_tick(
        &mut self,
        canvas: &mut Canvas<Window>,
        elapsed_time: f32,
    ) -> Result<(), String> {
        let window = canvas.window();
        let (window_width, window_height) = window.size();
        let aspect_ratio = window_height as f32 / window_width as f32;
        let fov_rad = 1.0 / (FOV * 0.5 / 180.0 * PI).tan();
        self.mat_proj[0][0] = aspect_ratio * fov_rad;
        self.mat_proj[1][1] = fov_rad;

        let mut mat_rot_z = Mat4x4::default();
        let mut mat_rot_x = Mat4x4::default();

        let theta = 1.0 * elapsed_time;
        mat_rot_z[0][0] = theta.cos();
        mat_rot_z[0][1] = theta.sin();
        mat_rot_z[1][0] = -theta.sin();
        mat_rot_z[1][1] = theta.cos();
        mat_rot_z[2][2] = 1.0;
        mat_rot_z[3][3] = 1.0;

        mat_rot_x[0][0] = 1.0;
        mat_rot_x[1][1] = (theta * 0.5).cos();
        mat_rot_x[1][2] = (theta * 0.5).sin();
        mat_rot_x[2][1] = -(theta * 0.5).sin();
        mat_rot_x[2][2] = (theta * 0.5).cos();
        mat_rot_x[3][3] = 1.0;

        for tri in self.mesh_cube.0.iter() {
            let tri_rotated_z = tri * &mat_rot_z;
            let mut tri_rotated_zx = &tri_rotated_z * &mat_rot_x;

            tri_rotated_zx[0].z += 3.0;
            tri_rotated_zx[1].z += 3.0;
            tri_rotated_zx[2].z += 3.0;

            let mut tri_projected = &tri_rotated_zx * &self.mat_proj;

            tri_projected[0].x += 1.0;
            tri_projected[1].x += 1.0;
            tri_projected[2].x += 1.0;
            tri_projected[0].y += 1.0;
            tri_projected[1].y += 1.0;
            tri_projected[2].y += 1.0;
            tri_projected[0].x *= 0.5 * window_width as f32;
            tri_projected[1].x *= 0.5 * window_width as f32;
            tri_projected[2].x *= 0.5 * window_width as f32;
            tri_projected[0].y *= 0.5 * window_height as f32;
            tri_projected[1].y *= 0.5 * window_height as f32;
            tri_projected[2].y *= 0.5 * window_height as f32;

            canvas.set_draw_color(Color::RGB(255, 255, 255));
            tri_projected.draw(canvas)?;
        }

        Ok(())
    }
}

#[derive(Debug, Clone, Copy, Default)]
struct Vec3D {
    x: f32,
    y: f32,
    z: f32,
}

impl Vec3D {
    fn new(x: f32, y: f32, z: f32) -> Self {
        Vec3D { x, y, z }
    }
}

impl Mul<&Mat4x4> for Vec3D {
    type Output = Vec3D;

    #[allow(clippy::suspicious_arithmetic_impl)]
    fn mul(self, rhs: &Mat4x4) -> Self::Output {
        let mut x = self.x * rhs[0][0] + self.y * rhs[1][0] + self.z * rhs[2][0] + rhs[3][0];
        let mut y = self.x * rhs[0][1] + self.y * rhs[1][1] + self.z * rhs[2][1] + rhs[3][1];
        let mut z = self.x * rhs[0][2] + self.y * rhs[1][2] + self.z * rhs[2][2] + rhs[3][2];
        let w = self.x * rhs[0][3] + self.y * rhs[1][3] + self.z * rhs[2][3] + rhs[3][3];

        if w != 0.0 {
            x /= w;
            y /= w;
            z /= w;
        }

        Vec3D::new(x, y, z)
    }
}

#[derive(Debug)]
struct Triangle([Vec3D; 3]);

impl Triangle {
    fn new(a: Vec3D, b: Vec3D, c: Vec3D) -> Self {
        Triangle([a, b, c])
    }

    #[allow(clippy::too_many_arguments)]
    fn from_points(
        ax: f32,
        ay: f32,
        az: f32,
        bx: f32,
        by: f32,
        bz: f32,
        cx: f32,
        cy: f32,
        cz: f32,
    ) -> Self {
        Triangle([
            Vec3D::new(ax, ay, az),
            Vec3D::new(bx, by, bz),
            Vec3D::new(cx, cy, cz),
        ])
    }

    fn draw(&self, canvas: &mut Canvas<Window>) -> Result<(), String> {
        let points = [
            Point::new(self[0].x as i32, self[0].y as i32),
            Point::new(self[1].x as i32, self[1].y as i32),
            Point::new(self[2].x as i32, self[2].y as i32),
            Point::new(self[0].x as i32, self[0].y as i32),
        ];
        canvas.draw_lines(&points[..])
    }
}

impl Index<usize> for Triangle {
    type Output = Vec3D;

    fn index(&self, index: usize) -> &Self::Output {
        &self.0[index]
    }
}

impl IndexMut<usize> for Triangle {
    fn index_mut(&mut self, index: usize) -> &mut Self::Output {
        &mut self.0[index]
    }
}

impl Mul<&Mat4x4> for &Triangle {
    type Output = Triangle;

    fn mul(self, rhs: &Mat4x4) -> Self::Output {
        Triangle::new(self[0] * rhs, self[1] * rhs, self[2] * rhs)
    }
}

#[derive(Debug)]
struct Mesh(Vec<Triangle>);

impl Mesh {
    fn new(triangles: Vec<Triangle>) -> Self {
        Mesh(triangles)
    }
}

#[derive(Debug, Default)]
struct Mat4x4([[f32; 4]; 4]);

impl Index<usize> for Mat4x4 {
    type Output = [f32; 4];

    fn index(&self, index: usize) -> &Self::Output {
        &self.0[index]
    }
}

impl IndexMut<usize> for Mat4x4 {
    fn index_mut(&mut self, index: usize) -> &mut Self::Output {
        &mut self.0[index]
    }
}
