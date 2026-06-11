#![cfg(all(feature = "metal", feature = "python"))]

use crate::metal_utils::dispatch_on_mps;
use crate::metal_utils::extract_buffer;
use metal::{
    BufferRef, CommandBufferRef, CompileOptions, ComputePipelineState, Device, MTLSize,
    foreign_types::ForeignTypeRef,
};
use pyo3::exceptions::PyRuntimeError;
use pyo3::prelude::*;
use std::ffi::c_void;
use std::sync::OnceLock;

use log::debug;

// --- CONTEXT ---
// We use a specific context for noop to avoid conflict with other scan/shared contexts
pub struct MetalContextNoop {
    pub device: Device,
    pub pipeline: ComputePipelineState,
}

static CONTEXT: OnceLock<MetalContextNoop> = OnceLock::new();

const KERNEL_SOURCE: &str = r#"
#include <metal_stdlib>
using namespace metal;

kernel void noop_kernel(
    device const float* alpha   [[buffer(0)]],
    device const float* h_init  [[buffer(1)]],
    device const float* x       [[buffer(2)]],
    device float* out           [[buffer(3)]],
    constant uint&      T              [[buffer(4)]],
    constant uint&      STRIDE_A       [[buffer(5)]], 
    constant uint&      R              [[buffer(6)]], 
    uint id [[thread_position_in_grid]]
) {
    // Purposefully empty for benchmarking overhead
}
"#;

pub fn set_metal_context(context: MetalContextNoop) -> Result<(), MetalContextNoop> {
    CONTEXT.set(context)
}

pub fn get_metal_context() -> Option<&'static MetalContextNoop> {
    CONTEXT.get()
}

// --- ENCODING STRUCT ---
struct EncodingContextNoop<'a> {
    pipeline: &'a ComputePipelineState,

    // Buffers
    b_init: &'a BufferRef,
    off_init: u64,
    b_alpha: &'a BufferRef,
    off_alpha: u64,
    b_x: &'a BufferRef,
    off_x: u64,
    b_out: &'a BufferRef,
    off_out: u64,

    // Params
    t_dim: u32,
    parallel_a: u32,
    r: u32,
}

// --- CALLBACK ---
extern "C" fn encode_noop_callback(cmd_buf_ptr: *mut c_void, context_ptr: *mut c_void) {
    unsafe {
        let cmd_buf = CommandBufferRef::from_ptr(cmd_buf_ptr as *mut _);
        let ctx = &*(context_ptr as *const EncodingContextNoop);

        let encoder = cmd_buf.new_compute_command_encoder();
        encoder.set_compute_pipeline_state(ctx.pipeline);

        encoder.set_buffer(0, Some(ctx.b_alpha), ctx.off_alpha);
        encoder.set_buffer(1, Some(ctx.b_init), ctx.off_init);
        encoder.set_buffer(2, Some(ctx.b_x), ctx.off_x);
        encoder.set_buffer(3, Some(ctx.b_out), ctx.off_out);

        encoder.set_bytes(4, 4, &ctx.t_dim as *const _ as *const _);
        encoder.set_bytes(5, 4, &ctx.parallel_a as *const _ as *const _);
        encoder.set_bytes(6, 4, &ctx.r as *const _ as *const _);

        let grid_size = MTLSize {
            width: 1,
            height: 1,
            depth: 1,
        };
        let thread_group = MTLSize {
            width: 1,
            height: 1,
            depth: 1,
        };

        encoder.dispatch_threads(grid_size, thread_group);
        encoder.end_encoding();
    }
}

// --- PYTHON FUNCTIONS ---

#[pyfunction(name = "ltv_noop_metal")]
pub fn ltv_noop_metal(
    _py: Python<'_>,
    h_init: Py<PyAny>,
    alpha: Py<PyAny>,
    x: Py<PyAny>,
    out: Py<PyAny>,
    t_dim: u32,
    parallel_a: u32,
    r: u32,
) -> PyResult<()> {
    debug!("Start NOOP Metal Kernel");
    let metal_ctx = get_metal_context()
        .ok_or_else(|| PyErr::new::<PyRuntimeError, _>("Metal context not initialized"))?;

    let (b_init, off_init) = extract_buffer(&h_init)?;
    let (b_alpha, off_alpha) = extract_buffer(&alpha)?;
    let (b_x, off_x) = extract_buffer(&x)?;
    let (b_out, off_out) = extract_buffer(&out)?;

    let mut ctx = EncodingContextNoop {
        pipeline: &metal_ctx.pipeline,
        b_init: &b_init,
        off_init,
        b_alpha: &b_alpha,
        off_alpha,
        b_x: &b_x,
        off_x,
        b_out: &b_out,
        off_out,
        t_dim,
        parallel_a,
        r,
    };

    // Dispatch via Bridge - ensures synchrony with PyTorch's stream
    unsafe {
        dispatch_on_mps(encode_noop_callback, &mut ctx as *mut _ as *mut c_void);
    }

    debug!("End NOOP Metal Kernel");
    Ok(())
}

#[pyfunction(name = "init_ltv_noop_metal_context_for_default_device")]
pub fn init_ltv_noop_metal_context_for_default_device() -> PyResult<()> {
    let device = Device::system_default()
        .ok_or_else(|| PyErr::new::<PyRuntimeError, _>("No Metal device found"))?;

    let library = device
        .new_library_with_source(KERNEL_SOURCE, &CompileOptions::new())
        .map_err(|e| PyErr::new::<PyRuntimeError, _>(e.to_string()))?;

    let kernel = library
        .get_function("noop_kernel", None)
        .map_err(|e| PyErr::new::<PyRuntimeError, _>(e.to_string()))?;

    let pipeline = device
        .new_compute_pipeline_state_with_function(&kernel)
        .map_err(|e| PyErr::new::<PyRuntimeError, _>(e.to_string()))?;

    let ctx = MetalContextNoop { device, pipeline };

    set_metal_context(ctx)
        .map_err(|_| PyErr::new::<PyRuntimeError, _>("Metal context already initialized"))?;

    Ok(())
}

#[pyfunction(name = "release_noop_metal_context")]
pub fn release_noop_metal_context() {}
