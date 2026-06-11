#![cfg(all(feature = "metal", feature = "python"))]

use crate::ltv_blelloch_utils::{
    EncodingContext, MetalContext, encode_callback, metal_context_for_default_device,
};
use crate::ltv_utils::create_geometry_from_schedule;
use crate::metal_utils::extract_buffer;

use crate::metal_utils::dispatch_on_mps;
use pyo3::exceptions::PyRuntimeError;
use pyo3::prelude::*;
use std::ffi::c_void;
use std::sync::OnceLock;

use log::info;

static CONTEXT: OnceLock<MetalContext> = OnceLock::new();

const UP_SWEEP_KERNEL_SOURCE: &str = r#"
#include <metal_stdlib>
using namespace metal;

kernel void up_sweep_kernel_plane(
    device const float* input_a      [[buffer(0)]],
    device const float* input_x      [[buffer(1)]],
    device float* stack_a            [[buffer(2)]],
    device float* stack_x            [[buffer(3)]],
    device float* summary_a          [[buffer(4)]],
    device float* summary_x          [[buffer(5)]],
    constant uint&      t_orig       [[buffer(6)]],
    constant uint&      chunk_size   [[buffer(7)]],
    constant uint&      r            [[buffer(8)]], 
    constant uint&      t_padded     [[buffer(9)]], 
    uint3 gid [[threadgroup_position_in_grid]],
    uint3 tid [[thread_position_in_threadgroup]]
) {
    uint chunk_idx = gid.x;   
    uint line_id_x = gid.y;   
    uint unit_idx  = tid.x;   

    uint line_id_a = line_id_x / r;
    bool is_index_zero_of_r = (line_id_a * r) == line_id_x;

    uint global_t  = chunk_idx * chunk_size + unit_idx;
    
    // Explicit indexing for (Spatial, Time) layout
    uint in_idx_a  = (line_id_a * t_orig) + global_t;
    uint in_idx_x  = (line_id_x * t_orig) + global_t;
    uint idx_a     = (line_id_a * t_padded) + global_t;
    uint idx_x     = (line_id_x * t_padded) + global_t;

    bool within_input = global_t < t_orig;

    float cur_a = within_input ? input_a[in_idx_a] : 1.0f;
    float cur_x = within_input ? input_x[in_idx_x] : 0.0f;

    for (uint offset = 1; offset < chunk_size; offset <<= 1) {
        float n_a = simd_shuffle_up(cur_a, offset);
        float n_x = simd_shuffle_up(cur_x, offset);

        if (unit_idx >= offset) {
            cur_x += cur_a * n_x;
            cur_a *= n_a;
        }
    }

    if (is_index_zero_of_r) {
        stack_a[idx_a] = cur_a;
    }
    stack_x[idx_x] = cur_x;

    if (unit_idx == chunk_size - 1) {
        uint n_chunks = t_padded / chunk_size;
        if (is_index_zero_of_r) {
            summary_a[(line_id_a * n_chunks) + chunk_idx] = cur_a;
        }
        summary_x[(line_id_x * n_chunks) + chunk_idx] = cur_x;
    }
}
"#;

pub fn set_metal_context(context: MetalContext) -> Result<(), MetalContext> {
    CONTEXT.set(context)
}

pub fn get_metal_context() -> Option<&'static MetalContext> {
    CONTEXT.get()
}

#[pyfunction(name = "ltv_plane_metal")]
pub fn ltv_plane_metal(
    _py: Python<'_>,
    alpha: Py<PyAny>,
    x: Py<PyAny>,
    out_a: Py<PyAny>,
    out_x: Py<PyAny>,
    t_seq: u32,
    parallel_a: u32,
    r: u32,
    schedule: &str,
) -> PyResult<()> {
    if parallel_a == 0 {
        // extract_buffer will fail
        return Ok(());
    }
    let metal_ctx = get_metal_context()
        .ok_or_else(|| PyErr::new::<PyRuntimeError, _>("Metal context not initialized"))?;

    let simd_width = metal_ctx.up_sweep_pipeline.thread_execution_width() as u32;

    let geometry = create_geometry_from_schedule(schedule, t_seq, parallel_a, r, simd_width)
        .map_err(|e| PyErr::new::<PyRuntimeError, _>(e.to_string()))?;

    let (b_alpha, off_alpha) = extract_buffer(&alpha)?;
    let (b_x, off_x) = extract_buffer(&x)?;
    let (b_out_alpha, off_out_alpha) = extract_buffer(&out_a)?;
    let (b_out_x, off_out_x) = extract_buffer(&out_x)?;

    let mut ctx = EncodingContext {
        metal_ctx,
        geometry: &geometry,
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

    // 5. Dispatch via Bridge
    // This blocks until the encoding is done on the MPS queue.
    unsafe {
        dispatch_on_mps(encode_callback, &mut ctx as *mut _ as *mut c_void);
    }

    Ok(())
}

#[pyfunction(name = "init_ltv_plane_metal_context_for_default_device")]
pub fn init_ltv_plane_metal_context_for_default_device() -> PyResult<()> {
    let metal_context =
        metal_context_for_default_device(UP_SWEEP_KERNEL_SOURCE, "up_sweep_kernel_plane")?;
    info!(
        "Metal Up Sweep thread_execution_width: {}",
        metal_context.up_sweep_pipeline.thread_execution_width()
    );
    set_metal_context(metal_context)
        .map_err(|_| PyErr::new::<PyRuntimeError, _>("Metal context already initialized"))?;

    Ok(())
}

#[pyfunction(name = "release_ltv_plane_metal_context")]
pub fn release_ltv_plane_metal_context() {}
