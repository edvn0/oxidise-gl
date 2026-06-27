//! Window + OpenGL context bootstrap, shared by the demo example and the editor.
//!
//! Creates a winit window, a glutin 4.3 context/surface, a `glow::Context`, and
//! loads the raw `glMultiDrawElementsIndirect` entry point. Enables GL debug
//! output in debug builds.

use crate::renderer::MultiDrawElementsIndirectFn;

use glow::{Context, HasContext};
use glutin::{
    config::ConfigTemplateBuilder,
    context::{
        ContextApi, ContextAttributesBuilder, NotCurrentGlContext, PossiblyCurrentContext,
        Version,
    },
    display::GetGlDisplay,
    prelude::*,
    surface::{Surface, WindowSurface},
};
use glutin_winit::{DisplayBuilder, GlWindow as _};
use raw_window_handle::HasRawWindowHandle;
use winit::{dpi::LogicalSize, event_loop::EventLoop, window::Window, window::WindowBuilder};

/// Everything a host needs to render: the window, the current GL surface and
/// context, the `glow::Context`, the loaded MDI entry point, and the chosen config.
pub struct GlWindow {
    pub window:    Window,
    pub surface:   Surface<WindowSurface>,
    pub context:   PossiblyCurrentContext,
    pub gl:        Context,
    pub mdi:       MultiDrawElementsIndirectFn,
    pub gl_config: glutin::config::Config,
}

/// Build a window and a current OpenGL 4.3 context for `event_loop`.
pub fn create_gl_window(
    event_loop: &EventLoop<()>,
    title: &str,
    width: u32,
    height: u32,
) -> GlWindow {
    let window_builder = WindowBuilder::new()
        .with_title(title)
        .with_inner_size(LogicalSize::new(width, height));

    let template = ConfigTemplateBuilder::new()
        .with_alpha_size(8)
        .with_depth_size(24);

    let display_builder = DisplayBuilder::new().with_window_builder(Some(window_builder));

    let (window, gl_config) = display_builder
        .build(event_loop, template, |configs| {
            configs
                .reduce(|a, b| if b.num_samples() > a.num_samples() { b } else { a })
                .expect("no suitable GL config")
        })
        .expect("failed to build display");

    let window: Window = window.expect("window creation failed");

    let context_attrs = ContextAttributesBuilder::new()
        .with_context_api(ContextApi::OpenGl(Some(Version::new(4, 3))))
        .with_debug(cfg!(debug_assertions))
        .build(Some(window.raw_window_handle()));

    let (surface, context): (Surface<WindowSurface>, PossiblyCurrentContext) = unsafe {
        let not_current = gl_config
            .display()
            .create_context(&gl_config, &context_attrs)
            .expect("failed to create GL context");

        let surface_attrs = window.build_surface_attributes(<_>::default());

        let surface = gl_config
            .display()
            .create_window_surface(&gl_config, &surface_attrs)
            .expect("failed to create window surface");

        let ctx = not_current
            .make_current(&surface)
            .expect("failed to make context current");

        (surface, ctx)
    };

    let mut gl = unsafe {
        Context::from_loader_function_cstr(|s| {
            gl_config.display().get_proc_address(s) as *const _
        })
    };

    // glow doesn't wrap glMultiDrawElementsIndirect, so load the entry point
    // ourselves from the same display. Core since GL 4.3 (guaranteed here).
    let mdi: MultiDrawElementsIndirectFn = unsafe {
        let name = std::ffi::CString::new("glMultiDrawElementsIndirect").unwrap();
        let ptr  = gl_config.display().get_proc_address(name.as_c_str());
        assert!(!ptr.is_null(), "glMultiDrawElementsIndirect unavailable (need GL 4.3+)");
        std::mem::transmute::<*const std::ffi::c_void, MultiDrawElementsIndirectFn>(ptr)
    };

    // Optional GL debug output in debug builds.
    //
    // Async by default: `DEBUG_OUTPUT_SYNCHRONOUS` makes the driver invoke the
    // callback synchronously at every GL call, which serializes the whole
    // CPU/driver pipeline and is very slow (especially on WSLg's Mesa→D3D12
    // path). We still get every message asynchronously; opt into synchronous
    // mode (precise call-site, but slow) by setting `GL_DEBUG_SYNC=1`.
    #[cfg(debug_assertions)]
    unsafe {
        if gl.supports_debug() {
            gl.enable(glow::DEBUG_OUTPUT);
            if std::env::var_os("GL_DEBUG_SYNC").is_some() {
                gl.enable(glow::DEBUG_OUTPUT_SYNCHRONOUS);
            }
            gl.debug_message_callback(|_src, _kind, _id, severity, msg| {
                if severity != glow::DEBUG_SEVERITY_NOTIFICATION {
                    eprintln!("[GL Debug] {msg}");
                }
            });
        }
    }

    GlWindow { window, surface, context, gl, mdi, gl_config }
}
