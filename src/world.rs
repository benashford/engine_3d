use sdl2::{render::Canvas, video::Window};

pub(crate) struct World {}

impl World {
    pub(crate) fn new() -> Self {
        World {}
    }

    pub(crate) fn do_tick(&mut self, canvas: &mut Canvas<Window>) -> Result<(), ()> {
        Ok(())
    }
}
