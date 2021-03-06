use std::time::Instant;

use sdl2::{event::Event, keyboard::Keycode, pixels::Color, render::BlendMode};

mod sdl_ext;
mod world;

use world::World;

fn main() -> Result<(), String> {
    let sdl_context = sdl2::init().expect("Cannot initialise SDL");
    let video_subsystem = sdl_context.video().expect("Canot get video");

    let window = video_subsystem
        .window("3d engine", 800, 600)
        .resizable()
        .build()
        .expect("Cannot create window");

    let mut canvas = window
        .into_canvas()
        .present_vsync()
        .build()
        .expect("Cannot get canvas");

    let mut tick = 0;
    let start_time = Instant::now();
    let mut prev_loop_inst = start_time;

    let mut event_pump = sdl_context.event_pump().expect("Cannot get event pump");

    let mut world = World::new().map_err(|e| e.to_string())?;

    'running: loop {
        let this_loop_inst = Instant::now();
        let frame_dur = this_loop_inst - prev_loop_inst;
        prev_loop_inst = this_loop_inst;
        let frames_per_sec = 1.0 / frame_dur.as_secs_f32();

        for event in event_pump.poll_iter() {
            match event {
                Event::Quit { .. }
                | Event::KeyDown {
                    keycode: Some(Keycode::Escape),
                    ..
                } => break 'running,
                Event::KeyDown {
                    keycode: Some(keycode),
                    ..
                } => world.handle_key(keycode, frame_dur.as_secs_f32()),
                _ => {}
            }
        }

        {
            // Update the window title.
            let window = canvas.window_mut();

            let position = window.position();
            let size = window.size();
            let title = format!(
                "Window - pos({}x{}), size({}x{}): {} (fps: {:.2})",
                position.0, position.1, size.0, size.1, tick, frames_per_sec
            );
            window.set_title(&title).expect("Cannot set title");

            tick += 1;
        }

        canvas.set_draw_color(Color::RGB(0, 0, 0));
        canvas.set_blend_mode(BlendMode::None);
        canvas.clear();

        world
            .do_tick(&mut canvas, frame_dur.as_secs_f32())
            .map_err(|e| e.to_string())?;

        canvas.present();
    }

    Ok(())
}
