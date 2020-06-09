use sdl2::{rect::Point, render::Canvas, video::Window};

pub(crate) fn fill_polygon(canvas: &mut Canvas<Window>, points: &[Point]) -> Result<(), String> {
    let num_points = points.len();
    let mut num_points_proc = 1;

    let (sum_x, sum_y) = points
        .iter()
        .fold((0, 0), |(sx, sy), pt| (sx + pt.x, sy + pt.y));
    let centre = Point::new(sum_x / num_points as i32, sum_y / num_points as i32);

    let mut top_y = points[0].y;
    let mut top_cnt = 0;

    (1..num_points).for_each(|idx| {
        if points[idx].y < top_y {
            top_y = points[idx].y;
            top_cnt = idx;
        }
    });

    let mut left_cnt = if top_cnt == 0 {
        num_points - 1
    } else {
        top_cnt - 1
    };

    let mut right_cnt = top_cnt + 1;
    if right_cnt >= num_points {
        right_cnt = 0;
    }

    let mut start_x = (points[top_cnt].x) << 16;
    let mut end_x = start_x;

    let mut cnt_y = points[top_cnt].y;

    let mut left_slope = if points[left_cnt].y != points[top_cnt].y {
        ((points[left_cnt].x - points[top_cnt].x) << 16) / (points[left_cnt].y - points[top_cnt].y)
    } else {
        0
    };

    let mut right_slope = if points[right_cnt].y != points[top_cnt].y {
        ((points[right_cnt].x - points[top_cnt].x) << 16)
            / (points[right_cnt].y - points[top_cnt].y)
    } else {
        0
    };

    while num_points_proc < num_points {
        while cnt_y < points[left_cnt].y && cnt_y < points[right_cnt].y {
            canvas.draw_line((start_x >> 16, cnt_y), (end_x >> 16, cnt_y))?;
            cnt_y += 1;
            start_x += left_slope;
            end_x += right_slope;
        }
        if points[left_cnt].y <= cnt_y {
            top_cnt = left_cnt;
            if left_cnt == 0 {
                left_cnt = num_points - 1;
            } else {
                left_cnt -= 1;
            }
            if points[left_cnt].y != points[top_cnt].y {
                left_slope = ((points[left_cnt].x - points[top_cnt].x) << 16)
                    / (points[left_cnt].y - points[top_cnt].y);
            }
            start_x = (points[top_cnt].x) << 16;
            num_points_proc += 1;
        }
        if points[right_cnt].y <= cnt_y {
            top_cnt = right_cnt;
            right_cnt += 1;
            if right_cnt == num_points {
                right_cnt = 0;
            }
            if points[right_cnt].y != points[top_cnt].y {
                right_slope = ((points[right_cnt].x - points[top_cnt].x) << 16)
                    / (points[right_cnt].y - points[top_cnt].y);
            }
            end_x = (points[top_cnt].x) << 16;
            num_points_proc += 1;
        }
        canvas.draw_line((start_x >> 16, cnt_y), (end_x >> 16, cnt_y))?;
    }

    Ok(())
}
