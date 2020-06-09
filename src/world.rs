use std::f32::consts::PI;
use std::fs::File;
use std::io::{BufRead, BufReader, Error as IOError};
use std::ops::{Index, IndexMut, Mul};

use sdl2::{pixels::Color, rect::Point, render::Canvas, video::Window};

use snafu::Snafu;

const FOV: f32 = 90.0;
const FAR: f32 = 1000.0;
const NEAR: f32 = 0.1;

#[derive(Debug, Snafu)]
pub(crate) enum WorldError {
    #[snafu(display("IO Error: {}", e))]
    IO { e: IOError },
    #[snafu(display("SDL Error: {}", msg))]
    SDL { msg: String },
    #[snafu(display("Cannot parse object file: {}", msg))]
    Parsing { msg: String },
}

impl WorldError {
    fn sdl_error(msg: impl Into<String>) -> Self {
        WorldError::SDL { msg: msg.into() }
    }

    fn parsing(msg: impl Into<String>) -> Self {
        WorldError::Parsing { msg: msg.into() }
    }
}

impl From<IOError> for WorldError {
    fn from(from: IOError) -> Self {
        WorldError::IO { e: from }
    }
}

pub(crate) struct World {
    mesh_cube: Mesh,
    mat_proj: Mat4x4,
}

impl World {
    pub(crate) fn new() -> Result<Self, WorldError> {
        let mesh_cube = Mesh::from_object_file("VideoShip.obj")?;
        let mut mat_proj = Mat4x4::default();
        mat_proj[2][2] = FAR / (FAR - NEAR);
        mat_proj[3][2] = (-FAR * NEAR) / (FAR - NEAR);
        mat_proj[2][3] = 1.0;
        mat_proj[3][3] = 0.0;
        Ok(World {
            mesh_cube,
            mat_proj,
        })
    }

    pub(crate) fn do_tick(
        &mut self,
        canvas: &mut Canvas<Window>,
        elapsed_time: f32,
    ) -> Result<(), WorldError> {
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
            let tri_rotated_z = *tri * &mat_rot_z;
            let mut tri_rotated_zx = tri_rotated_z * &mat_rot_x;

            tri_rotated_zx[0].z += 3.0;
            tri_rotated_zx[1].z += 3.0;
            tri_rotated_zx[2].z += 3.0;

            let mut tri_projected = tri_rotated_zx * &self.mat_proj;

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
            tri_projected.draw(canvas).map_err(WorldError::sdl_error)?;
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

#[derive(Debug, Clone, Copy, Default)]
struct Triangle {
    p: [Vec3D; 3],
    col: f32,
}

impl Triangle {
    fn new(a: Vec3D, b: Vec3D, c: Vec3D) -> Self {
        Triangle {
            p: [a, b, c],
            ..Default::default()
        }
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
        Triangle {
            p: [
                Vec3D::new(ax, ay, az),
                Vec3D::new(bx, by, bz),
                Vec3D::new(cx, cy, cz),
            ],
            ..Default::default()
        }
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
        &self.p[index]
    }
}

impl IndexMut<usize> for Triangle {
    fn index_mut(&mut self, index: usize) -> &mut Self::Output {
        &mut self.p[index]
    }
}

impl Mul<&Mat4x4> for Triangle {
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

    fn from_object_file(filename: &str) -> Result<Self, WorldError> {
        let file = File::open(filename)?;
        let mut verts = Vec::new();
        let mut triangles = Vec::new();
        for line in BufReader::new(file).lines() {
            let line = line?;
            let parse_float = |s: &str| {
                s.parse()
                    .map_err(|e| WorldError::parsing(format!("Cannot parse {}, {}", line, e)))
            };
            let parse_int = |s: &str| {
                s.parse::<usize>()
                    .map_err(|e| WorldError::parsing(format!("Cannot parse {}, {}", line, e)))
            };
            let mut split_line = line.split(' ');
            match (
                split_line.next(),
                split_line.next(),
                split_line.next(),
                split_line.next(),
            ) {
                (Some("#"), _, _, _) => (),
                (Some("v"), Some(v1), Some(v2), Some(v3)) => verts.push(Vec3D::new(
                    parse_float(v1)?,
                    parse_float(v2)?,
                    parse_float(v3)?,
                )),
                (Some("s"), _, _, _) => (),
                (Some("f"), Some(f1), Some(f2), Some(f3)) => triangles.push(Triangle::new(
                    verts[parse_int(f1)? - 1],
                    verts[parse_int(f2)? - 1],
                    verts[parse_int(f3)? - 1],
                )),
                _ => {
                    return Err(WorldError::parsing(format!(
                        "Object file not in correct format: {}",
                        line
                    )))
                }
            }
        }

        Ok(Mesh(triangles))
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
