use std::cmp::Ordering;
use std::collections::VecDeque;
use std::f32::consts::PI;
use std::fs::File;
use std::io::{BufRead, BufReader, Error as IOError};
use std::ops::{Add, AddAssign, Div, DivAssign, Index, IndexMut, Mul, MulAssign, Sub, SubAssign};

use sdl2::{keyboard::Keycode, pixels::Color, rect::Point, render::Canvas, video::Window};

use smallvec::SmallVec;

use snafu::Snafu;

use super::sdl_ext::{fill_polygon, Pt};

const FAR: f32 = 1000.0;
const NEAR: f32 = 0.1;

static UP: Vec3D = Vec3D::new(0.0, 1.0, 0.0);
static TARGET: Vec3D = Vec3D::new(0.0, 0.0, 1.0);

static PLANE_P: Vec3D = Vec3D::new(0.0, 0.0, 0.1);
static PLANE_N: Vec3D = Vec3D::new(0.0, 0.0, 1.0);

static OFFSET_VIEW: Vec3D = Vec3D::new(1.0, 1.0, 0.0);

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

#[derive(Debug)]
pub(crate) struct World {
    mesh_cube: Mesh,

    triangles_to_raster: Vec<Triangle>,
    triangles_to_clip: VecDeque<Triangle>,

    camera: Vec3D,
    look_dir: Vec3D,

    yaw: f32,
    yaw_diff: f32,

    direction_of_travel: Vec3D,

    theta: f32,
}

impl World {
    pub(crate) fn new() -> Result<Self, WorldError> {
        let mesh_cube = Mesh::from_object_file("mountains.obj")?;

        Ok(World {
            mesh_cube,
            triangles_to_raster: Vec::new(),
            triangles_to_clip: VecDeque::new(),
            camera: Vec3D::default(),
            look_dir: Vec3D::default(),
            yaw: 0.0,
            yaw_diff: 0.0,
            direction_of_travel: Vec3D::default(),
            theta: 0.0,
        })
    }

    pub(crate) fn handle_key(&mut self, keycode: Keycode, elapsed_time: f32) {
        match keycode {
            Keycode::Up => {
                let up = UP * (32.0 * elapsed_time);
                self.direction_of_travel += up;
            }
            Keycode::Down => {
                let up = UP * (32.0 * elapsed_time);
                self.direction_of_travel -= up;
            }
            Keycode::Left => {
                let left = (self.look_dir * Mat4x4::rotation_y(PI / 2.0)).normalise()
                    * (32.0 * elapsed_time);
                self.direction_of_travel -= left;
            }
            Keycode::Right => {
                let left = (self.look_dir * Mat4x4::rotation_y(PI / 2.0)).normalise()
                    * (32.0 * elapsed_time);
                self.direction_of_travel += left;
            }
            Keycode::W => {
                let forward = self.look_dir * (32.0 * elapsed_time);
                self.direction_of_travel += forward;
            }
            Keycode::S => {
                let forward = self.look_dir * (32.0 * elapsed_time);
                self.direction_of_travel -= forward;
            }
            Keycode::A => self.yaw_diff -= 2.0 * elapsed_time,
            Keycode::D => self.yaw_diff += 2.0 * elapsed_time,
            _ => (),
        }
    }

    pub(crate) fn do_tick(
        &mut self,
        canvas: &mut Canvas<Window>,
        _elapsed_time: f32,
    ) -> Result<(), WorldError> {
        let window = canvas.window();
        let (window_width, window_height) = window.size();
        let aspect_ratio = window_height as f32 / window_width as f32;

        self.camera += self.direction_of_travel;
        self.direction_of_travel *= 0.9;

        self.yaw += self.yaw_diff;
        self.yaw_diff *= 0.9;

        let mat_proj = Mat4x4::projection(60.0, aspect_ratio, NEAR, FAR);

        let mat_rot_z = Mat4x4::rotation_z(self.theta / 2.0);
        let mat_rot_x = Mat4x4::rotation_x(self.theta);

        let mat_trans = Mat4x4::translation(0.0, 0.0, 5.0);

        let mat_world = mat_rot_z * mat_rot_x;
        let mat_world = mat_world * mat_trans;

        let mat_camera_rot = Mat4x4::rotation_y(self.yaw);
        self.look_dir = TARGET * mat_camera_rot;
        let target = self.camera + self.look_dir;
        let mat_camera = self.camera.point_at(&target, &UP);

        let mat_view = mat_camera.quick_inverse();

        self.triangles_to_raster.clear();

        for tri in self.mesh_cube.0.iter() {
            let mut tri_transformed =
                Triangle::new(tri[0] * mat_world, tri[1] * mat_world, tri[2] * mat_world);

            let line_1 = tri_transformed[1] - tri_transformed[0];
            let line_2 = tri_transformed[2] - tri_transformed[0];

            let normal = line_1.cross_product(&line_2).normalise();
            let camera_ray = tri_transformed[0] - self.camera;

            if normal.dot_product(&camera_ray) < 0.0 {
                let light_direction = Vec3D::new(0.0, 1.0, -1.0).normalise();
                let mut dp = light_direction.dot_product(&normal);
                if dp < 0.1 {
                    dp = 0.1
                }
                tri_transformed.col = (255.0 * dp) as u8;
                tri_transformed[0] *= mat_view;
                tri_transformed[1] *= mat_view;
                tri_transformed[2] *= mat_view;

                let clipped_triangles = tri_transformed.clip_against_plane(&PLANE_P, &PLANE_N);

                for mut clipped_tri in clipped_triangles {
                    clipped_tri[0] *= mat_proj;
                    clipped_tri[1] *= mat_proj;
                    clipped_tri[2] *= mat_proj;

                    clipped_tri[0] = clipped_tri[0] / clipped_tri[0].w;
                    clipped_tri[1] = clipped_tri[1] / clipped_tri[1].w;
                    clipped_tri[2] = clipped_tri[2] / clipped_tri[2].w;

                    clipped_tri[0].x *= -1.0;
                    clipped_tri[1].x *= -1.0;
                    clipped_tri[2].x *= -1.0;
                    clipped_tri[0].y *= -1.0;
                    clipped_tri[1].y *= -1.0;
                    clipped_tri[2].y *= -1.0;

                    clipped_tri[0] += OFFSET_VIEW;
                    clipped_tri[1] += OFFSET_VIEW;
                    clipped_tri[2] += OFFSET_VIEW;
                    clipped_tri[0].x *= 0.5 * window_width as f32;
                    clipped_tri[0].y *= 0.5 * window_height as f32;
                    clipped_tri[1].x *= 0.5 * window_width as f32;
                    clipped_tri[1].y *= 0.5 * window_height as f32;
                    clipped_tri[2].x *= 0.5 * window_width as f32;
                    clipped_tri[2].y *= 0.5 * window_height as f32;

                    self.triangles_to_raster.push(clipped_tri);
                }
            }
        }

        self.triangles_to_raster.sort_by(|t1, t2| {
            let z1 = (t1[0].z + t1[1].z + t1[2].z) / 3.0;
            let z2 = (t2[0].z + t2[1].z + t2[2].z) / 3.0;
            z2.partial_cmp(&z1).unwrap_or(Ordering::Equal)
        });

        for tri in self.triangles_to_raster.drain(..) {
            self.triangles_to_clip.push_back(tri);
            let mut new_triangles = 1;
            for p in 0..4 {
                while new_triangles > 0 {
                    let test = self
                        .triangles_to_clip
                        .pop_front()
                        .expect("Already tested for...");
                    new_triangles -= 1;
                    let (plane_p, plane_n) = match p {
                        0 => (Vec3D::new(0.0, 0.0, 0.0), Vec3D::new(0.0, 1.0, 0.0)),
                        1 => (
                            Vec3D::new(0.0, window_height as f32 - 1.0, 0.0),
                            Vec3D::new(0.0, -1.0, 0.0),
                        ),
                        2 => (Vec3D::new(0.0, 0.0, 0.0), Vec3D::new(1.0, 0.0, 0.0)),
                        3 => (
                            Vec3D::new(window_width as f32 - 1.0, 0.0, 0.0),
                            Vec3D::new(-1.0, 0.0, 0.0),
                        ),
                        _x => unreachable!(),
                    };

                    self.triangles_to_clip
                        .extend(test.clip_against_plane(&plane_p, &plane_n));
                }
                new_triangles = self.triangles_to_clip.len();
            }

            for tri in self.triangles_to_clip.drain(..) {
                tri.fill(canvas)?;
            }
        }

        Ok(())
    }
}

#[derive(Debug, Clone, Copy)]
struct Vec3D {
    x: f32,
    y: f32,
    z: f32,
    w: f32,
}

impl Vec3D {
    const fn new(x: f32, y: f32, z: f32) -> Self {
        Vec3D { x, y, z, w: 0.0 }
    }

    fn len(&self) -> f32 {
        self.dot_product(self).sqrt()
    }

    fn normalise(&self) -> Vec3D {
        let l = self.len();
        Vec3D::new(self.x / l, self.y / l, self.z / l)
    }

    fn dot_product(&self, other: &Vec3D) -> f32 {
        self.x * other.x + self.y * other.y + self.z * other.z
    }

    fn cross_product(&self, other: &Vec3D) -> Vec3D {
        Vec3D::new(
            self.y * other.z - self.z * other.y,
            self.z * other.x - self.x * other.z,
            self.x * other.y - self.y * other.x,
        )
    }

    fn point_at(&self, target: &Vec3D, up: &Vec3D) -> Mat4x4 {
        let new_forward = (*target - *self).normalise();

        let a = new_forward * up.dot_product(&new_forward);
        let new_up = *up - a;

        let new_right = new_up.cross_product(&new_forward);
        let mut m = Mat4x4::default();
        m[0][0] = new_right.x;
        m[0][1] = new_right.y;
        m[0][2] = new_right.z;
        m[0][3] = 0.0;
        m[1][0] = new_up.x;
        m[1][1] = new_up.y;
        m[1][2] = new_up.z;
        m[1][3] = 0.0;
        m[2][0] = new_forward.x;
        m[2][1] = new_forward.y;
        m[2][2] = new_forward.z;
        m[2][3] = 0.0;
        m[3][0] = self.x;
        m[3][1] = self.y;
        m[3][2] = self.z;
        m[3][3] = 1.0;
        m
    }
}

fn intersect_plane(
    plane_p: &Vec3D,
    plane_n: &Vec3D,
    line_start: &Vec3D,
    line_end: &Vec3D,
) -> Vec3D {
    let plane_n = plane_n.normalise();
    let plane_d = -plane_n.dot_product(plane_p);
    let ad = line_start.dot_product(&plane_n);
    let bd = line_end.dot_product(&plane_n);
    let t = (-plane_d - ad) / (bd - ad);
    let line_start_to_end = *line_end - *line_start;
    let line_to_intersect = line_start_to_end * t;
    *line_start + line_to_intersect
}

impl Default for Vec3D {
    fn default() -> Self {
        Vec3D {
            x: 0.0,
            y: 0.0,
            z: 0.0,
            w: 1.0,
        }
    }
}

impl Mul<Mat4x4> for Vec3D {
    type Output = Vec3D;

    fn mul(self, rhs: Mat4x4) -> Self::Output {
        let x = self.x * rhs[0][0] + self.y * rhs[1][0] + self.z * rhs[2][0] + rhs[3][0];
        let y = self.x * rhs[0][1] + self.y * rhs[1][1] + self.z * rhs[2][1] + rhs[3][1];
        let z = self.x * rhs[0][2] + self.y * rhs[1][2] + self.z * rhs[2][2] + rhs[3][2];
        let w = self.x * rhs[0][3] + self.y * rhs[1][3] + self.z * rhs[2][3] + rhs[3][3];

        Vec3D { x, y, z, w }
    }
}

impl MulAssign<Mat4x4> for Vec3D {
    fn mul_assign(&mut self, rhs: Mat4x4) {
        let x = self.x * rhs[0][0] + self.y * rhs[1][0] + self.z * rhs[2][0] + rhs[3][0];
        let y = self.x * rhs[0][1] + self.y * rhs[1][1] + self.z * rhs[2][1] + rhs[3][1];
        let z = self.x * rhs[0][2] + self.y * rhs[1][2] + self.z * rhs[2][2] + rhs[3][2];
        let w = self.x * rhs[0][3] + self.y * rhs[1][3] + self.z * rhs[2][3] + rhs[3][3];

        self.x = x;
        self.y = y;
        self.z = z;
        self.w = w;
    }
}

impl Add<Vec3D> for Vec3D {
    type Output = Vec3D;

    fn add(self, rhs: Vec3D) -> Self::Output {
        Vec3D::new(self.x + rhs.x, self.y + rhs.y, self.z + rhs.z)
    }
}

impl AddAssign<Vec3D> for Vec3D {
    fn add_assign(&mut self, rhs: Vec3D) {
        self.x += rhs.x;
        self.y += rhs.y;
        self.z += rhs.z;
    }
}

impl Sub<Vec3D> for Vec3D {
    type Output = Vec3D;

    fn sub(self, rhs: Vec3D) -> Self::Output {
        Vec3D::new(self.x - rhs.x, self.y - rhs.y, self.z - rhs.z)
    }
}

impl SubAssign<Vec3D> for Vec3D {
    fn sub_assign(&mut self, rhs: Vec3D) {
        self.x -= rhs.x;
        self.y -= rhs.y;
        self.z -= rhs.z;
    }
}

impl Mul<f32> for Vec3D {
    type Output = Vec3D;

    fn mul(self, rhs: f32) -> Self::Output {
        Vec3D::new(self.x * rhs, self.y * rhs, self.z * rhs)
    }
}

impl MulAssign<f32> for Vec3D {
    fn mul_assign(&mut self, rhs: f32) {
        self.x *= rhs;
        self.y *= rhs;
        self.z *= rhs;
        self.w = 0.0;
    }
}

impl Div<f32> for Vec3D {
    type Output = Vec3D;

    fn div(self, rhs: f32) -> Self::Output {
        Vec3D::new(self.x / rhs, self.y / rhs, self.z / rhs)
    }
}

impl DivAssign<f32> for Vec3D {
    fn div_assign(&mut self, rhs: f32) {
        self.x /= rhs;
        self.y /= rhs;
        self.z /= rhs;
    }
}

#[derive(Debug, Clone, Copy, Default)]
struct Triangle {
    p: [Vec3D; 3],
    col: u8,
}

impl Triangle {
    fn new(a: Vec3D, b: Vec3D, c: Vec3D) -> Self {
        Triangle {
            p: [a, b, c],
            ..Default::default()
        }
    }

    #[allow(dead_code)]
    fn draw(&self, canvas: &mut Canvas<Window>) -> Result<(), WorldError> {
        canvas.set_draw_color(Color::RGB(255, self.col, self.col));
        let points = [
            Point::new(self[0].x as i32, self[0].y as i32),
            Point::new(self[1].x as i32, self[1].y as i32),
            Point::new(self[2].x as i32, self[2].y as i32),
            Point::new(self[0].x as i32, self[0].y as i32),
        ];
        canvas
            .draw_lines(&points[..])
            .map_err(|e| WorldError::sdl_error(format!("Cannot draw lines: {}", e)))
    }

    fn fill(&self, canvas: &mut Canvas<Window>) -> Result<(), WorldError> {
        canvas.set_draw_color(Color::RGB(self.col, self.col, self.col));

        let points = [
            Pt::new(self[0].x as i32, self[0].y as i32),
            Pt::new(self[1].x as i32, self[1].y as i32),
            Pt::new(self[2].x as i32, self[2].y as i32),
            Pt::new(self[0].x as i32, self[0].y as i32),
        ];
        fill_polygon(canvas, &points[..])
            .map_err(|e| WorldError::sdl_error(format!("Cannot fill polygon: {}", e)))
    }

    fn clip_against_plane(&self, plane_p: &Vec3D, plane_n: &Vec3D) -> SmallVec<[Triangle; 2]> {
        let plane_n = plane_n.normalise();
        let dist = |p: &Vec3D| {
            plane_n.x * p.x + plane_n.y * p.y + plane_n.z * p.z - plane_n.dot_product(plane_p)
        };
        let mut inside_points = SmallVec::<[&Vec3D; 3]>::new();
        let mut outside_points = SmallVec::<[&Vec3D; 3]>::new();

        for idx in 0..3 {
            let d = dist(&self[idx]);
            if d >= 0.0 {
                inside_points.push(&self[idx])
            } else {
                outside_points.push(&self[idx])
            }
        }

        let inside_points_count = inside_points.len();
        let outside_points_count = outside_points.len();

        let mut result = SmallVec::new();
        if inside_points_count == 0 {
            return result;
        }

        if inside_points_count == 3 {
            result.push(*self);
            return result;
        }

        if inside_points_count == 1 && outside_points_count == 2 {
            let mut tri = *self;
            tri[0] = *inside_points[0];
            tri[1] = intersect_plane(plane_p, &plane_n, &inside_points[0], &outside_points[0]);
            tri[2] = intersect_plane(plane_p, &plane_n, &inside_points[0], &outside_points[1]);

            result.push(tri);
            return result;
        }

        if inside_points_count == 2 && outside_points_count == 1 {
            let mut tri_1 = *self;
            let mut tri_2 = *self;

            tri_1[0] = *inside_points[0];
            tri_1[1] = *inside_points[1];
            tri_1[2] = intersect_plane(plane_p, &plane_n, &inside_points[0], &outside_points[0]);

            tri_2[0] = *inside_points[1];
            tri_2[1] = tri_1[2];
            tri_2[2] = intersect_plane(plane_p, &plane_n, &inside_points[1], &outside_points[0]);

            result.push(tri_1);
            result.push(tri_2);

            return result;
        }

        unreachable!()
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

impl Mul<Mat4x4> for Triangle {
    type Output = Triangle;

    fn mul(self, rhs: Mat4x4) -> Self::Output {
        Triangle::new(self[0] * rhs, self[1] * rhs, self[2] * rhs)
    }
}

#[derive(Debug)]
struct Mesh(Vec<Triangle>);

impl Mesh {
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

#[derive(Debug, Copy, Clone, Default)]
struct Mat4x4([[f32; 4]; 4]);

impl Mat4x4 {
    fn rotation_x(angle_rad: f32) -> Self {
        let mut m = Mat4x4::default();
        m[0][0] = 1.0;
        m[1][1] = angle_rad.cos();
        m[1][2] = angle_rad.sin();
        m[2][1] = -angle_rad.sin();
        m[2][2] = angle_rad.cos();
        m[3][3] = 1.0;
        m
    }

    fn rotation_y(angle_rad: f32) -> Self {
        let mut m = Mat4x4::default();
        m[0][0] = angle_rad.cos();
        m[0][2] = angle_rad.sin();
        m[2][0] = -angle_rad.sin();
        m[1][1] = 1.0;
        m[2][2] = angle_rad.cos();
        m[3][3] = 1.0;
        m
    }

    fn rotation_z(angle_rad: f32) -> Self {
        let mut m = Mat4x4::default();
        m[0][0] = angle_rad.cos();
        m[0][1] = angle_rad.sin();
        m[1][0] = -angle_rad.sin();
        m[1][1] = angle_rad.cos();
        m[2][2] = 1.0;
        m[3][3] = 1.0;
        m
    }

    fn translation(x: f32, y: f32, z: f32) -> Self {
        let mut m = Mat4x4::default();
        m[0][0] = 1.0;
        m[1][1] = 1.0;
        m[2][2] = 1.0;
        m[3][3] = 1.0;
        m[3][0] = x;
        m[3][1] = y;
        m[3][2] = z;
        m
    }

    fn projection(fov_degrees: f32, aspect_ratio: f32, near: f32, far: f32) -> Self {
        let fov_rad = 1.0 / (fov_degrees * 0.5 / 180.0 * PI).tan();
        let mut m = Mat4x4::default();
        m[0][0] = aspect_ratio * fov_rad;
        m[1][1] = fov_rad;
        m[2][2] = far / (far - near);
        m[3][2] = (-far * near) / (far - near);
        m[2][3] = 1.0;
        m[3][3] = 0.0;
        m
    }

    fn quick_inverse(&self) -> Self {
        let mut m = Mat4x4::default();
        m[0][0] = self[0][0];
        m[0][1] = self[1][0];
        m[0][2] = self[2][0];
        m[0][3] = 0.0;
        m[1][0] = self[0][1];
        m[1][1] = self[1][1];
        m[1][2] = self[2][1];
        m[1][3] = 0.0;
        m[2][0] = self[0][2];
        m[2][1] = self[1][2];
        m[2][2] = self[2][2];
        m[2][3] = 0.0;
        m[3][0] = -(self[3][0] * m[0][0] + self[3][1] * m[1][0] + self[3][2] * m[2][0]);
        m[3][1] = -(self[3][0] * m[0][1] + self[3][1] * m[1][1] + self[3][2] * m[2][1]);
        m[3][2] = -(self[3][0] * m[0][2] + self[3][1] * m[1][2] + self[3][2] * m[2][2]);
        m[3][3] = 1.0;
        m
    }
}

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

impl Mul<Mat4x4> for Mat4x4 {
    type Output = Mat4x4;

    fn mul(self, rhs: Mat4x4) -> Self::Output {
        let mut m = Mat4x4::default();
        for c in 0..4 {
            for r in 0..4 {
                m[r][c] = self[r][0] * rhs[0][c]
                    + self[r][1] * rhs[1][c]
                    + self[r][2] * rhs[2][c]
                    + self[r][3] * rhs[3][c];
            }
        }
        m
    }
}
