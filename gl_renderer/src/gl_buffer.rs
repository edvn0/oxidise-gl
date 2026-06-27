#![allow(dead_code)]

use glow::{Context, HasContext, NativeBuffer};
use std::marker::PhantomData;

/// Target for the buffer binding point.
#[derive(Clone, Copy, Debug)]
pub enum BufferTarget {
    ShaderStorage,
    ElementArray,
    DrawIndirect,
}

impl BufferTarget {
    fn gl_target(self) -> u32 {
        match self {
            BufferTarget::ShaderStorage => glow::SHADER_STORAGE_BUFFER,
            BufferTarget::ElementArray => glow::ELEMENT_ARRAY_BUFFER,
            BufferTarget::DrawIndirect => glow::DRAW_INDIRECT_BUFFER,
        }
    }
}

/// Usage hint for the GPU driver.
#[derive(Clone, Copy, Debug)]
pub enum BufferUsage {
    StaticDraw,
    DynamicDraw,
    StreamDraw,
}

impl BufferUsage {
    fn gl_usage(self) -> u32 {
        match self {
            BufferUsage::StaticDraw => glow::STATIC_DRAW,
            BufferUsage::DynamicDraw => glow::DYNAMIC_DRAW,
            BufferUsage::StreamDraw => glow::STREAM_DRAW,
        }
    }
}

/// A typed OpenGL buffer wrapping a `NativeBuffer`.
/// `T` is the element type; the buffer stores `capacity` elements.
pub struct GlBuffer<T: bytemuck::Pod> {
    pub handle: NativeBuffer,
    pub capacity: usize,
    _marker: PhantomData<T>,
}

impl<T: bytemuck::Pod> GlBuffer<T> {
    /// Allocate an empty buffer with `capacity` elements (no data uploaded).
    pub unsafe fn new(gl: &Context, capacity: usize, usage: BufferUsage) -> Self {
        let handle = gl.create_buffer().expect("failed to create GL buffer");
        let byte_size = capacity * std::mem::size_of::<T>();
        gl.bind_buffer(glow::ARRAY_BUFFER, Some(handle));
        gl.buffer_data_size(glow::ARRAY_BUFFER, byte_size as i32, usage.gl_usage());
        gl.bind_buffer(glow::ARRAY_BUFFER, None);
        Self {
            handle,
            capacity,
            _marker: PhantomData,
        }
    }

    /// Create a buffer and upload `data` in one call.
    pub unsafe fn from_data(gl: &Context, data: &[T], usage: BufferUsage) -> Self {
        let handle = gl.create_buffer().expect("failed to create GL buffer");
        gl.bind_buffer(glow::ARRAY_BUFFER, Some(handle));
        gl.buffer_data_u8_slice(
            glow::ARRAY_BUFFER,
            bytemuck::cast_slice(data),
            usage.gl_usage(),
        );
        gl.bind_buffer(glow::ARRAY_BUFFER, None);
        Self {
            handle,
            capacity: data.len(),
            _marker: PhantomData,
        }
    }

    /// Upload a subrange starting at `element_offset`.
    /// Panics if `element_offset + data.len() > self.capacity`.
    pub unsafe fn upload_subrange(&self, gl: &Context, data: &[T], element_offset: usize) {
        assert!(
            element_offset + data.len() <= self.capacity,
            "subrange upload out of bounds"
        );
        let byte_offset = (element_offset * std::mem::size_of::<T>()) as i32;
        gl.bind_buffer(glow::ARRAY_BUFFER, Some(self.handle));
        gl.buffer_sub_data_u8_slice(
            glow::ARRAY_BUFFER,
            byte_offset,
            bytemuck::cast_slice(data),
        );
        gl.bind_buffer(glow::ARRAY_BUFFER, None);
    }

    /// Map the buffer for write, call `f` with the raw byte slice, then unmap.
    /// Uses GL_MAP_WRITE_BIT | GL_MAP_INVALIDATE_RANGE_BIT.
    pub unsafe fn mapped_upload<F>(&self, gl: &Context, element_offset: usize, count: usize, f: F)
    where
        F: FnOnce(&mut [T]),
    {
        assert!(element_offset + count <= self.capacity, "mapped upload out of bounds");
        let byte_offset = (element_offset * std::mem::size_of::<T>()) as i32;
        let byte_length = (count * std::mem::size_of::<T>()) as i32;
        gl.bind_buffer(glow::ARRAY_BUFFER, Some(self.handle));
        let ptr = gl.map_buffer_range(
            glow::ARRAY_BUFFER,
            byte_offset,
            byte_length,
            glow::MAP_WRITE_BIT | glow::MAP_INVALIDATE_RANGE_BIT,
        );
        assert!(!ptr.is_null(), "map_buffer_range returned null");
        let slice =
            std::slice::from_raw_parts_mut(ptr as *mut T, count);
        f(slice);
        gl.unmap_buffer(glow::ARRAY_BUFFER);
        gl.bind_buffer(glow::ARRAY_BUFFER, None);
    }

    /// Map with EXPLICIT_FLUSH_BIT, call `f`, flush the range, then unmap.
    pub unsafe fn mapped_upload_explicit_flush<F>(
        &self,
        gl: &Context,
        element_offset: usize,
        count: usize,
        f: F,
    ) where
        F: FnOnce(&mut [T]),
    {
        assert!(element_offset + count <= self.capacity, "mapped upload out of bounds");
        let byte_offset = (element_offset * std::mem::size_of::<T>()) as i32;
        let byte_length = (count * std::mem::size_of::<T>()) as i32;
        gl.bind_buffer(glow::ARRAY_BUFFER, Some(self.handle));
        let ptr = gl.map_buffer_range(
            glow::ARRAY_BUFFER,
            byte_offset,
            byte_length,
            glow::MAP_WRITE_BIT | glow::MAP_INVALIDATE_RANGE_BIT | glow::MAP_FLUSH_EXPLICIT_BIT,
        );
        assert!(!ptr.is_null(), "map_buffer_range returned null");
        let slice = std::slice::from_raw_parts_mut(ptr as *mut T, count);
        f(slice);
        // Flush the entire mapped subrange
        gl.flush_mapped_buffer_range(glow::ARRAY_BUFFER, 0, byte_length);
        gl.unmap_buffer(glow::ARRAY_BUFFER);
        gl.bind_buffer(glow::ARRAY_BUFFER, None);
    }

    /// Bind as a Shader Storage Buffer at `binding_index`.
    pub unsafe fn bind_as_ssbo(&self, gl: &Context, binding_index: u32) {
        gl.bind_buffer_base(glow::SHADER_STORAGE_BUFFER, binding_index, Some(self.handle));
    }

    /// Bind as an Element Array Buffer (index buffer).
    pub unsafe fn bind_as_ibo(&self, gl: &Context) {
        gl.bind_buffer(glow::ELEMENT_ARRAY_BUFFER, Some(self.handle));
    }

    /// Bind as the Draw Indirect Buffer (source of `DrawCommand`s for MDI).
    pub unsafe fn bind_as_indirect(&self, gl: &Context) {
        gl.bind_buffer(glow::DRAW_INDIRECT_BUFFER, Some(self.handle));
    }

    /// Unbind the element array buffer.
    pub unsafe fn unbind_ibo(gl: &Context) {
        gl.bind_buffer(glow::ELEMENT_ARRAY_BUFFER, None);
    }

    /// Unbind the draw indirect buffer.
    pub unsafe fn unbind_indirect(gl: &Context) {
        gl.bind_buffer(glow::DRAW_INDIRECT_BUFFER, None);
    }

    /// Reallocate to `new_capacity` elements, preserving existing data via a GPU-side copy.
    /// The old GL buffer is deleted; `self` is updated in place.
    pub unsafe fn grow(&mut self, gl: &Context, new_capacity: usize, usage: BufferUsage) {
        assert!(new_capacity > self.capacity, "grow: new_capacity must exceed current capacity");
        let new_handle = gl.create_buffer().expect("failed to create GL buffer");
        let new_byte_size = (new_capacity * std::mem::size_of::<T>()) as i32;
        let old_byte_size = (self.capacity * std::mem::size_of::<T>()) as i32;

        gl.bind_buffer(glow::ARRAY_BUFFER, Some(new_handle));
        gl.buffer_data_size(glow::ARRAY_BUFFER, new_byte_size, usage.gl_usage());
        gl.bind_buffer(glow::ARRAY_BUFFER, None);

        gl.bind_buffer(glow::COPY_READ_BUFFER, Some(self.handle));
        gl.bind_buffer(glow::COPY_WRITE_BUFFER, Some(new_handle));
        gl.copy_buffer_sub_data(glow::COPY_READ_BUFFER, glow::COPY_WRITE_BUFFER, 0, 0, old_byte_size);
        gl.bind_buffer(glow::COPY_READ_BUFFER, None);
        gl.bind_buffer(glow::COPY_WRITE_BUFFER, None);

        gl.delete_buffer(self.handle);
        self.handle = new_handle;
        self.capacity = new_capacity;
    }

    /// Delete the underlying GL buffer.
    pub unsafe fn cleanup(&self, gl: &Context) {
        gl.delete_buffer(self.handle);
    }

    pub fn byte_capacity(&self) -> usize {
        self.capacity * std::mem::size_of::<T>()
    }
}
