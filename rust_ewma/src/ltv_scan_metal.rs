#![cfg(all(feature = "metal", feature = "python"))]

use crate::metal_utils::dispatch_on_mps;
use crate::metal_utils::extract_buffer;
use log::{debug, info};
use metal::{
    BufferRef, CommandBufferRef, CompileOptions, ComputePipelineState, Device, MTLSize,
    foreign_types::ForeignTypeRef,
};
use pyo3::exceptions::PyRuntimeError;
use pyo3::prelude::*;
use std::ffi::c_void;
use std::sync::OnceLock;

// --- CONTEXT ---
struct MetalContextScan {
    device: Device,
    pipeline: ComputePipelineState,
}

static CONTEXT: OnceLock<MetalContextScan> = OnceLock::new();

const KERNEL_SOURCE: &str = r#"
#include <metal_stdlib>
using namespace metal;

kernel void ltv_scan_kernel(
    device const float* alpha   [[buffer(0)]],
    device const float* x       [[buffer(1)]],
    device float* out_alpha     [[buffer(2)]],
    device float* out_x         [[buffer(3)]],
    constant uint&      t_orig         [[buffer(4)]],
    constant uint&      parallel_dim_a [[buffer(5)]], // Total spatial size of Alpha (B * Da)
    constant uint&      r              [[buffer(6)]], // Ratio (Dx / Da)
    uint id [[thread_position_in_grid]]
) {
    // Total parallel threads = parallel_dim_a * r = parallel_dim_x
    uint stride_x = parallel_dim_a * r;
    
    // On M4 Pro with dispatch_threads, we only get here if id < (parallel_a * r)
    // if (id >= stride_x) return;

    // Index mapping:
    // x and h_init are flat in the spatial dimension -> index is just 'id'
    // alpha is broadcasted -> index is 'id / r'
    uint idx_a_spatial = id / r;
    bool is_index_zero_of_r = (idx_a_spatial * r) == id;

    // Load initial value
    float acc_a = 1.0;
    float acc_x = 0.0;

    uint offset_a = idx_a_spatial;
    uint offset_x = id;

    // Sequential Scan Loop
    for (uint t = 0; t < t_orig; t++) {
        // Layout is (t_orig, Spatial)

        float val_a = alpha[offset_a];
        float val_x = x[offset_x];

        // LTV Update
        acc_x = val_a * acc_x + val_x;

        // Write output
        if (is_index_zero_of_r) {
            acc_a *= val_a;
            out_alpha[offset_a] = acc_a;
        }
        out_x[offset_x] = acc_x;

        offset_x += stride_x;
        offset_a += parallel_dim_a;
    }
}
"#;

fn set_metal_context(context: MetalContextScan) -> Result<(), MetalContextScan> {
    CONTEXT.set(context)
}

fn get_metal_context() -> Option<&'static MetalContextScan> {
    CONTEXT.get()
}

// --- ENCODING STRUCT ---
struct EncodingContextScan<'a> {
    metal_ctx: &'static MetalContextScan,

    // Input Buffers
    b_alpha: &'a BufferRef,
    off_alpha: u64,
    b_x: &'a BufferRef,
    off_x: u64,
    b_out_alpha: &'a BufferRef,
    off_out_alpha: u64,
    b_out_x: &'a BufferRef,
    off_out_x: u64,

    // Params
    parallel_a: u32,
    r: u32,
    t_seq: u32,
}

// --- CALLBACK ---
extern "C" fn encode_scan_callback(cmd_buf_ptr: *mut c_void, context_ptr: *mut c_void) {
    unsafe {
        let cmd_buf = CommandBufferRef::from_ptr(cmd_buf_ptr as *mut _);
        let ctx = &*(context_ptr as *const EncodingContextScan);

        let curr_a = ctx.b_alpha.to_owned();
        let curr_a_off = ctx.off_alpha;
        let curr_x = ctx.b_x.to_owned();
        let curr_x_off = ctx.off_x;
        let out_alpha = ctx.b_out_alpha.to_owned();
        let out_alpha_off = ctx.off_out_alpha;
        let out_x = ctx.b_out_x.to_owned();
        let out_x_off = ctx.off_out_x;

        let encoder = cmd_buf.new_compute_command_encoder();
        encoder.set_compute_pipeline_state(&ctx.metal_ctx.pipeline);

        encoder.set_buffer(0, Some(&curr_a), curr_a_off);
        encoder.set_buffer(1, Some(&curr_x), curr_x_off);
        encoder.set_buffer(2, Some(&out_alpha), out_alpha_off);
        encoder.set_buffer(3, Some(&out_x), out_x_off);

        encoder.set_bytes(4, 4, &ctx.t_seq as *const _ as *const _);
        encoder.set_bytes(5, 4, &ctx.parallel_a as *const _ as *const _);
        encoder.set_bytes(6, 4, &ctx.r as *const _ as *const _);

        let stride_x = (ctx.parallel_a * ctx.r) as u64;

        let grid_size = MTLSize {
            width: stride_x,
            height: 1,
            depth: 1,
        };

        let max_threads = ctx.metal_ctx.pipeline.max_total_threads_per_threadgroup();
        let thread_group = MTLSize {
            width: max_threads.min(stride_x),
            height: 1,
            depth: 1,
        };

        encoder.dispatch_threads(grid_size, thread_group);
        encoder.end_encoding();
    }
}

// --- PYTHON FUNCTIONS ---

#[pyfunction(name = "ltv_scan_metal")]
pub fn ltv_scan_metal(
    _py: Python<'_>,
    alpha: Py<PyAny>,
    x: Py<PyAny>,
    out_a: Py<PyAny>,
    out_x: Py<PyAny>,
    t_seq: u32,
    parallel_a: u32,
    r: u32,
) -> PyResult<()> {
    if parallel_a == 0 {
        // extract_buffer will fail
        return Ok(());
    }
    // 1. Get Context
    let metal_ctx = get_metal_context()
        .ok_or_else(|| PyErr::new::<PyRuntimeError, _>("Metal context not initialized"))?;

    let (b_alpha, off_alpha) = extract_buffer(&alpha)?;
    let (b_x, off_x) = extract_buffer(&x)?;
    let (b_out_alpha, off_out_alpha) = extract_buffer(&out_a)?;
    let (b_out_x, off_out_x) = extract_buffer(&out_x)?;

    // 3. Pack Context
    let mut ctx = EncodingContextScan {
        metal_ctx,
        b_alpha: &b_alpha,
        off_alpha,
        b_x: &b_x,
        off_x,
        b_out_alpha: &b_out_alpha,
        off_out_alpha,
        b_out_x: &b_out_x,
        off_out_x,
        parallel_a,
        r,
        t_seq,
    };

    // 4. Dispatch via Bridge
    unsafe {
        dispatch_on_mps(encode_scan_callback, &mut ctx as *mut _ as *mut c_void);
    }

    debug!("End");
    Ok(())
}

#[pyfunction(name = "init_ltv_scan_metal_context_for_default_device")]
pub fn init_ltv_scan_metal_context_for_default_device() -> PyResult<()> {
    let device = Device::system_default()
        .ok_or_else(|| PyErr::new::<PyRuntimeError, _>("No Metal device found"))?;

    let library = device
        .new_library_with_source(KERNEL_SOURCE, &CompileOptions::new())
        .map_err(|e| PyErr::new::<PyRuntimeError, _>(e.to_string()))?;

    let kernel = library
        .get_function("ltv_scan_kernel", None)
        .map_err(|e| PyErr::new::<PyRuntimeError, _>(e.to_string()))?;

    let pipeline = device
        .new_compute_pipeline_state_with_function(&kernel)
        .map_err(|e| PyErr::new::<PyRuntimeError, _>(e.to_string()))?;

    info!(
        "Metal pipeline max_total_threads_per_threadgroup: {}",
        pipeline.max_total_threads_per_threadgroup()
    );

    // Note: We no longer create a local command queue.
    let ctx = MetalContextScan { device, pipeline };

    set_metal_context(ctx)
        .map_err(|_| PyErr::new::<PyRuntimeError, _>("Metal context already initialized"))?;

    Ok(())
}

#[pyfunction(name = "release_ltv_scan_metal_context")]
pub fn release_ltv_scan_metal_context() {}
