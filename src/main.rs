use sdl2::{event::Event, keyboard::Keycode, pixels::Color};

fn main() {
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

    let mut event_pump = sdl_context.event_pump().expect("Cannot get event pump");

    'running: loop {
        for event in event_pump.poll_iter() {
            match event {
                Event::Quit { .. }
                | Event::KeyDown {
                    keycode: Some(Keycode::Escape),
                    ..
                } => break 'running,
                _ => {}
            }
        }

        {
            // Update the window title.
            let window = canvas.window_mut();

            let position = window.position();
            let size = window.size();
            let title = format!(
                "Window - pos({}x{}), size({}x{}): {}",
                position.0, position.1, size.0, size.1, tick
            );
            window.set_title(&title).expect("Cannot set title");

            tick += 1;
        }

        canvas.set_draw_color(Color::RGB(0, 0, 0));
        canvas.clear();
        canvas.present();
    }
}
