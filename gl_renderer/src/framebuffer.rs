//! A simple offscreen render target (color texture + depth texture) used
//! to render the scene into a texture so the editor can display it inside an
//! imgui viewport panel, and so the renderer can build an HZB from the depth.

use glow::{Context, HasContext, NativeFramebuffer, NativeTexture};

pub struct OffscreenTarget {
    fbo:    NativeFramebuffer,
    color:  NativeTexture,
    depth:  NativeTexture,
    width:  u32,
    height: u32,
}

impl OffscreenTarget {
    /// Create an FBO with an RGBA8 color texture and a DEPTH_COMPONENT32F depth
    /// texture (not a renderbuffer, so the depth can be sampled for HZB).
    pub unsafe fn new(gl: &Context, width: u32, height: u32) -> Self {
        let width  = width.max(1);
        let height = height.max(1);

        let color = gl.create_texture().expect("failed to create color texture");
        gl.bind_texture(glow::TEXTURE_2D, Some(color));
        gl.tex_parameter_i32(glow::TEXTURE_2D, glow::TEXTURE_MIN_FILTER, glow::LINEAR as i32);
        gl.tex_parameter_i32(glow::TEXTURE_2D, glow::TEXTURE_MAG_FILTER, glow::LINEAR as i32);
        gl.tex_parameter_i32(glow::TEXTURE_2D, glow::TEXTURE_WRAP_S, glow::CLAMP_TO_EDGE as i32);
        gl.tex_parameter_i32(glow::TEXTURE_2D, glow::TEXTURE_WRAP_T, glow::CLAMP_TO_EDGE as i32);

        let depth = gl.create_texture().expect("failed to create depth texture");
        gl.bind_texture(glow::TEXTURE_2D, Some(depth));
        // NEAREST + no compare mode so texelFetch in the HZB reduce shader works cleanly.
        gl.tex_parameter_i32(glow::TEXTURE_2D, glow::TEXTURE_MIN_FILTER, glow::NEAREST as i32);
        gl.tex_parameter_i32(glow::TEXTURE_2D, glow::TEXTURE_MAG_FILTER, glow::NEAREST as i32);
        gl.tex_parameter_i32(glow::TEXTURE_2D, glow::TEXTURE_WRAP_S, glow::CLAMP_TO_EDGE as i32);
        gl.tex_parameter_i32(glow::TEXTURE_2D, glow::TEXTURE_WRAP_T, glow::CLAMP_TO_EDGE as i32);
        gl.tex_parameter_i32(glow::TEXTURE_2D, glow::TEXTURE_COMPARE_MODE, glow::NONE as i32);

        let fbo = gl.create_framebuffer().expect("failed to create framebuffer");

        let mut target = Self { fbo, color, depth, width, height };
        target.allocate(gl);
        target
    }

    unsafe fn allocate(&mut self, gl: &Context) {
        // ── Color ─────────────────────────────────────────────────────────────
        gl.bind_texture(glow::TEXTURE_2D, Some(self.color));
        gl.tex_image_2d(
            glow::TEXTURE_2D, 0, glow::RGBA8 as i32,
            self.width as i32, self.height as i32, 0,
            glow::RGBA, glow::UNSIGNED_BYTE, None,
        );

        // ── Depth ─────────────────────────────────────────────────────────────
        // 32F for reverse-Z precision; sampleable texture instead of renderbuffer
        // so the HZB reduction compute can texelFetch it.
        gl.bind_texture(glow::TEXTURE_2D, Some(self.depth));
        gl.tex_image_2d(
            glow::TEXTURE_2D, 0, glow::DEPTH_COMPONENT32F as i32,
            self.width as i32, self.height as i32, 0,
            glow::DEPTH_COMPONENT, glow::FLOAT, None,
        );

        // ── FBO ───────────────────────────────────────────────────────────────
        gl.bind_framebuffer(glow::FRAMEBUFFER, Some(self.fbo));
        gl.framebuffer_texture_2d(
            glow::FRAMEBUFFER, glow::COLOR_ATTACHMENT0,
            glow::TEXTURE_2D, Some(self.color), 0,
        );
        gl.framebuffer_texture_2d(
            glow::FRAMEBUFFER, glow::DEPTH_ATTACHMENT,
            glow::TEXTURE_2D, Some(self.depth), 0,
        );

        let status = gl.check_framebuffer_status(glow::FRAMEBUFFER);
        assert_eq!(
            status, glow::FRAMEBUFFER_COMPLETE,
            "offscreen framebuffer incomplete: {status:#x}"
        );

        gl.bind_framebuffer(glow::FRAMEBUFFER, None);
        gl.bind_texture(glow::TEXTURE_2D, None);
    }

    /// Reallocate storage if the requested size changed (clamps zero to 1).
    pub unsafe fn resize(&mut self, gl: &Context, width: u32, height: u32) {
        let width  = width.max(1);
        let height = height.max(1);
        if width == self.width && height == self.height {
            return;
        }
        self.width  = width;
        self.height = height;
        self.allocate(gl);
    }

    /// Bind this FBO and set the viewport to its size.
    pub unsafe fn bind(&self, gl: &Context) {
        gl.bind_framebuffer(glow::FRAMEBUFFER, Some(self.fbo));
        gl.viewport(0, 0, self.width as i32, self.height as i32);
    }

    pub fn color_texture(&self) -> NativeTexture { self.color }
    pub fn depth_texture(&self) -> NativeTexture { self.depth }

    /// Blit the color attachment to the currently-bound draw framebuffer
    /// (pass `dst_w/h` matching the destination viewport size).
    pub unsafe fn blit_color_to_default(&self, gl: &Context, dst_w: u32, dst_h: u32) {
        gl.bind_framebuffer(glow::READ_FRAMEBUFFER, Some(self.fbo));
        gl.bind_framebuffer(glow::DRAW_FRAMEBUFFER, None);
        gl.blit_framebuffer(
            0, 0, self.width as i32, self.height as i32,
            0, 0, dst_w as i32, dst_h as i32,
            glow::COLOR_BUFFER_BIT,
            glow::LINEAR,
        );
        gl.bind_framebuffer(glow::FRAMEBUFFER, None);
    }

    pub unsafe fn cleanup(&self, gl: &Context) {
        gl.delete_framebuffer(self.fbo);
        gl.delete_texture(self.color);
        gl.delete_texture(self.depth);
    }
}
