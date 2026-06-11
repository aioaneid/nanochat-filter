#![cfg(all(feature = "metal", feature = "python"))]

use pyo3::exceptions::PyRuntimeError;
use pyo3::prelude::*;

use metal::{Buffer, BufferRef};
use std::ffi::c_void;

#[link(name = "mps_bridge", kind = "static")]
unsafe extern "C" {
    pub fn get_mtl_buffer_and_offset(tensor_ptr: *const c_void, offset: *mut u64) -> *mut c_void;
}

unsafe extern "C" {
    /// Dispatch a Rust callback on an MPS command buffer.
    ///
    /// # Safety
    /// - `callback` must obey the C ABI
    /// - `context` must remain valid for the duration of the callback
    pub fn dispatch_on_mps(callback: extern "C" fn(*mut c_void, *mut c_void), context: *mut c_void);
}

/// Shared buffer extraction helper
pub fn extract_buffer(obj: &Py<PyAny>) -> PyResult<(Buffer, u64)> {
    let mut offset: u64 = 0;
    let ptr = unsafe { get_mtl_buffer_and_offset(obj.as_ptr() as *mut _, &mut offset) };
    if ptr.is_null() {
        return Err(PyErr::new::<PyRuntimeError, _>("Null Metal buffer"));
    }
    let buf = unsafe { &*(ptr as *const BufferRef) };
    Ok((buf.to_owned(), offset))
}
