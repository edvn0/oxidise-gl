//! A simple offscreen render target (color texture + depth renderbuffer) used
//! to render the scene into a texture so the editor can display it inside an
//! imgui viewport panel.

use glow::{Context, HasContext, NativeFramebuffer, NativeRenderbuffer, NativeTexture};

pub struct OffscreenTarget {
    fbo:    NativeFramebuffer,
    color:  NativeTexture,
    depth:  NativeRenderbuffer,
    width:  u32,
    height: u32,
}

impl OffscreenTarget {
    /// Create an FBO with an RGBA8 color texture and a 24-bit depth renderbuffer.
    pub unsafe fn new(gl: &Context, width: u32, height: u32) -> Self {
        let width  = width.max(1);
        let height = height.max(1);

        let color = gl.create_texture().expect("failed to create color texture");
        gl.bind_texture(glow::TEXTURE_2D, Some(color));
        gl.tex_parameter_i32(glow::TEXTURE_2D, glow::TEXTURE_MIN_FILTER, glow::LINEAR as i32);
        gl.tex_parameter_i32(glow::TEXTURE_2D, glow::TEXTURE_MAG_FILTER, glow::LINEAR as i32);
        gl.tex_parameter_i32(glow::TEXTURE_2D, glow::TEXTURE_WRAP_S, glow::CLAMP_TO_EDGE as i32);
        gl.tex_parameter_i32(glow::TEXTURE_2D, glow::TEXTURE_WRAP_T, glow::CLAMP_TO_EDGE as i32);

        let depth = gl.create_renderbuffer().expect("failed to create depth renderbuffer");
        let fbo   = gl.create_framebuffer().expect("failed to create framebuffer");

        let mut target = Self { fbo, color, depth, width, height };
        target.allocate(gl);
        target
    }

    /// (Re)allocate the color texture and depth storage and (re)attach them.
    unsafe fn allocate(&mut self, gl: &Context) {
        gl.bind_texture(glow::TEXTURE_2D, Some(self.color));
        gl.tex_image_2d(
            glow::TEXTURE_2D,
            0,
            glow::RGBA8 as i32,
            self.width as i32,
            self.height as i32,
            0,
            glow::RGBA,
            glow::UNSIGNED_BYTE,
            None,
        );

        gl.bind_renderbuffer(glow::RENDERBUFFER, Some(self.depth));
        gl.renderbuffer_storage(
            glow::RENDERBUFFER,
            glow::DEPTH_COMPONENT24,
            self.width as i32,
            self.height as i32,
        );

        gl.bind_framebuffer(glow::FRAMEBUFFER, Some(self.fbo));
        gl.framebuffer_texture_2d(
            glow::FRAMEBUFFER,
            glow::COLOR_ATTACHMENT0,
            glow::TEXTURE_2D,
            Some(self.color),
            0,
        );
        gl.framebuffer_renderbuffer(
            glow::FRAMEBUFFER,
            glow::DEPTH_ATTACHMENT,
            glow::RENDERBUFFER,
            Some(self.depth),
        );

        let status = gl.check_framebuffer_status(glow::FRAMEBUFFER);
        assert_eq!(
            status,
            glow::FRAMEBUFFER_COMPLETE,
            "offscreen framebuffer incomplete: {status:#x}"
        );

        gl.bind_framebuffer(glow::FRAMEBUFFER, None);
    }

    /// Reallocate storage if the requested size changed (ignores zero sizes).
    pub unsafe fn resize(&mut self, gl: &Context, width: u32, height: u32) {
        let width  = width.max(1);
        let height = height.max(1);
        if width == self.width && height == self.height {
            return;
        }
        self.width = width;
        self.height = height;
        self.allocate(gl);
    }

    /// Bind this FBO and set the viewport to its size.
    pub unsafe fn bind(&self, gl: &Context) {
        gl.bind_framebuffer(glow::FRAMEBUFFER, Some(self.fbo));
        gl.viewport(0, 0, self.width as i32, self.height as i32);
    }

    pub fn color_texture(&self) -> NativeTexture {
        self.color
    }

    pub fn width(&self) -> u32 {
        self.width
    }

    pub fn height(&self) -> u32 {
        self.height
    }

    pub unsafe fn cleanup(&self, gl: &Context) {
        gl.delete_framebuffer(self.fbo);
        gl.delete_texture(self.color);
        gl.delete_renderbuffer(self.depth);
    }
}
