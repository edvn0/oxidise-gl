use std::f32::consts::FRAC_PI_2;

use glam::{Mat4, Vec3};
use winit::event::{ElementState, MouseButton, MouseScrollDelta};
use winit::keyboard::{KeyCode, ModifiersState};
use winit::window::{CursorGrabMode, Window};

const LOOK_SENS: f32 = 0.003;
const PAN_SENS: f32 = 0.003;
const MIN_PITCH: f32 = -FRAC_PI_2 + 0.001;
const MAX_PITCH: f32 = FRAC_PI_2 - 0.001;
const MIN_PIVOT_DIST: f32 = 0.1;

#[derive(PartialEq, Clone, Copy)]
enum CaptureMode {
    None,
    Fly,
    Orbit,
    Pan,
}

pub struct EditorCamera {
    pub position: Vec3,
    pub yaw: f32,
    pub pitch: f32,
    pivot: Vec3,
    pivot_distance: f32,
    pub fly_speed: f32,

    capture: CaptureMode,
    alt: bool,
    shift: bool,
    key_fwd: bool,
    key_back: bool,
    key_left: bool,
    key_right: bool,
    key_up: bool,
    key_down: bool,
}

impl EditorCamera {
    pub fn new() -> Self {
        let position = Vec3::new(4.5, 3.0, 5.5);
        let pivot = Vec3::ZERO;
        let dir = (pivot - position).normalize();
        let pitch = dir.y.asin();
        let yaw = f32::atan2(dir.x, -dir.z);
        Self {
            position,
            yaw,
            pitch,
            pivot,
            pivot_distance: (pivot - position).length(),
            fly_speed: 5.0,
            capture: CaptureMode::None,
            alt: false,
            shift: false,
            key_fwd: false,
            key_back: false,
            key_left: false,
            key_right: false,
            key_up: false,
            key_down: false,
        }
    }

    pub fn view_matrix(&self) -> Mat4 {
        let f = self.forward();
        Mat4::look_at_rh(self.position, self.position + f, Vec3::Y)
    }

    /// Set the orbit pivot to `point` and adjust pivot_distance.
    pub fn focus_on(&mut self, point: Vec3) {
        self.pivot = point;
        self.pivot_distance = (point - self.position).length().max(MIN_PIVOT_DIST);
    }

    pub fn is_capturing(&self) -> bool {
        self.capture != CaptureMode::None
    }

    // ── Event handlers ────────────────────────────────────────────────────────

    pub fn on_mouse_button(
        &mut self,
        button: MouseButton,
        state: ElementState,
        viewport_hovered: bool,
        window: &Window,
    ) {
        let pressed = state == ElementState::Pressed;
        match button {
            MouseButton::Right => {
                if pressed && viewport_hovered && self.capture == CaptureMode::None {
                    self.capture = CaptureMode::Fly;
                    grab_cursor(window);
                } else if !pressed && self.capture == CaptureMode::Fly {
                    self.capture = CaptureMode::None;
                    release_cursor(window);
                }
            }
            MouseButton::Left => {
                if pressed && self.alt && viewport_hovered && self.capture == CaptureMode::None {
                    self.pivot_distance = (self.pivot - self.position).length().max(MIN_PIVOT_DIST);
                    self.capture = CaptureMode::Orbit;
                    grab_cursor(window);
                } else if !pressed && self.capture == CaptureMode::Orbit {
                    self.capture = CaptureMode::None;
                    release_cursor(window);
                }
            }
            MouseButton::Middle => {
                if pressed && viewport_hovered && self.capture == CaptureMode::None {
                    self.capture = CaptureMode::Pan;
                    grab_cursor(window);
                } else if !pressed && self.capture == CaptureMode::Pan {
                    self.capture = CaptureMode::None;
                    release_cursor(window);
                }
            }
            _ => {}
        }
    }

    /// Raw mouse delta from `DeviceEvent::MouseMotion`. Drives look/orbit/pan.
    pub fn on_raw_mouse_delta(&mut self, dx: f32, dy: f32) {
        match self.capture {
            CaptureMode::None => {}
            CaptureMode::Fly | CaptureMode::Orbit => {
                self.yaw += dx * LOOK_SENS;
                self.pitch = (self.pitch - dy * LOOK_SENS).clamp(MIN_PITCH, MAX_PITCH);
                if self.capture == CaptureMode::Orbit {
                    self.position = self.pivot - self.forward() * self.pivot_distance;
                }
            }
            CaptureMode::Pan => {
                let speed = self.pivot_distance * PAN_SENS;
                let right = self.right();
                let up = right.cross(self.forward()).normalize();
                let delta = -(right * dx + up * dy) * speed;
                self.position += delta;
                self.pivot += delta;
            }
        }
    }

    pub fn on_scroll(&mut self, delta: &MouseScrollDelta, viewport_hovered: bool) {
        if !viewport_hovered && !self.is_capturing() {
            return;
        }
        let lines = match delta {
            MouseScrollDelta::LineDelta(_, y) => *y,
            MouseScrollDelta::PixelDelta(pos) => pos.y as f32 / 20.0,
        };
        if self.capture == CaptureMode::Fly {
            // Adjust fly speed; scroll up = faster.
            self.fly_speed = (self.fly_speed * 1.15_f32.powf(lines)).clamp(0.1, 200.0);
        } else {
            // Dolly: proportional to current distance for consistent feel.
            let step = lines * self.pivot_distance.max(0.2) * 0.15;
            self.position += self.forward() * step;
            self.pivot_distance = (self.pivot - self.position).length().max(MIN_PIVOT_DIST);
        }
    }

    pub fn on_key(&mut self, key: KeyCode, state: ElementState) {
        let p = state == ElementState::Pressed;
        match key {
            KeyCode::KeyW => self.key_fwd = p,
            KeyCode::KeyS => self.key_back = p,
            KeyCode::KeyA => self.key_left = p,
            KeyCode::KeyD => self.key_right = p,
            KeyCode::KeyQ => self.key_down = p,
            KeyCode::KeyE => self.key_up = p,
            _ => {}
        }
    }

    pub fn on_modifiers(&mut self, mods: ModifiersState) {
        self.alt = mods.alt_key();
        self.shift = mods.shift_key();
    }

    /// Call once per frame with `dt` in seconds. Moves the camera in fly mode.
    pub fn update(&mut self, dt: f32) {
        if self.capture != CaptureMode::Fly {
            return;
        }
        let forward = self.forward();
        let right = self.right();
        let speed = self.fly_speed * if self.shift { 4.0 } else { 1.0 };

        let mut vel = Vec3::ZERO;
        if self.key_fwd {
            vel += forward;
        }
        if self.key_back {
            vel -= forward;
        }
        if self.key_right {
            vel += right;
        }
        if self.key_left {
            vel -= right;
        }
        if self.key_up {
            vel += Vec3::Y;
        }
        if self.key_down {
            vel -= Vec3::Y;
        }

        if vel.length_squared() > 0.0 {
            self.position += vel.normalize() * speed * dt;
            self.pivot = self.position + forward * self.pivot_distance;
        }
    }

    // ── Private helpers ───────────────────────────────────────────────────────

    fn forward(&self) -> Vec3 {
        Vec3::new(
            self.yaw.sin() * self.pitch.cos(),
            self.pitch.sin(),
            -self.yaw.cos() * self.pitch.cos(),
        )
    }

    fn right(&self) -> Vec3 {
        Vec3::new(self.yaw.cos(), 0.0, self.yaw.sin())
    }
}

fn grab_cursor(window: &Window) {
    let _ = window
        .set_cursor_grab(CursorGrabMode::Confined)
        .or_else(|_| window.set_cursor_grab(CursorGrabMode::Locked));
    window.set_cursor_visible(false);
}

fn release_cursor(window: &Window) {
    let _ = window.set_cursor_grab(CursorGrabMode::None);
    window.set_cursor_visible(true);
}
