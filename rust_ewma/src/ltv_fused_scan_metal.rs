#![cfg(all(feature = "metal", feature = "python"))]

use log::{info, warn};

use crate::metal_utils::{dispatch_on_mps, extract_buffer};
use metal::{
    BufferRef, CommandBufferRef, CompileOptions, ComputePipelineState, Device, MTLSize,
    foreign_types::ForeignTypeRef,
};
use pyo3::exceptions::PyRuntimeError;
use pyo3::prelude::*;
use std::ffi::c_void;
use std::sync::OnceLock;

// --- CONTEXT ---
struct MetalContextFused {
    pub device: Device,
    pub pipeline_forward: ComputePipelineState,
    pub pipeline_backward_direct: ComputePipelineState,
    pub pipeline_backward_atomic: ComputePipelineState,
}

static CONTEXT: OnceLock<MetalContextFused> = OnceLock::new();

const FORWARD_KERNEL_SOURCE: &str = include_str!("ltv_fused_scan_forward.msl");
const BACKWARD_KERNEL_SOURCE: &str = include_str!("ltv_fused_scan_backward.msl");

fn set_metal_context(context: MetalContextFused) -> Result<(), MetalContextFused> {
    CONTEXT.set(context)
}

fn get_metal_context() -> Option<&'static MetalContextFused> {
    CONTEXT.get()
}

// --- FORWARD ENCODING ---
struct EncodingContextFusedForward<'a> {
    pub metal_ctx: &'static MetalContextFused,
    pub b_q: &'a BufferRef,
    pub off_q: u64,
    pub b_k: &'a BufferRef,
    pub off_k: u64,
    pub b_v: &'a BufferRef,
    pub off_v: u64,
    pub b_logits: &'a BufferRef,
    pub off_logits: u64,
    pub b_inits: &'a BufferRef,
    pub off_inits: u64,
    pub b_out_q: &'a BufferRef,
    pub off_out_q: u64,
    pub b_out_k: &'a BufferRef,
    pub off_out_k: u64,
    pub b_out_v: &'a BufferRef,
    pub off_out_v: u64,
    pub b: u32,
    pub t: u32,
    pub nh: u32,
    pub nkvh: u32,
    pub da: u32,
    pub r: u32,
}

extern "C" fn encode_fused_forward_callback(cmd_buf_ptr: *mut c_void, context_ptr: *mut c_void) {
    unsafe {
        let cmd_buf = CommandBufferRef::from_ptr(cmd_buf_ptr as *mut _);
        let ctx = &*(context_ptr as *const EncodingContextFusedForward);
        let encoder = cmd_buf.new_compute_command_encoder();
        encoder.set_compute_pipeline_state(&ctx.metal_ctx.pipeline_forward);

        encoder.set_buffer(0, Some(ctx.b_q), ctx.off_q);
        encoder.set_buffer(1, Some(ctx.b_k), ctx.off_k);
        encoder.set_buffer(2, Some(ctx.b_v), ctx.off_v);
        encoder.set_buffer(3, Some(ctx.b_logits), ctx.off_logits);
        encoder.set_buffer(4, Some(ctx.b_inits), ctx.off_inits);
        encoder.set_buffer(5, Some(ctx.b_out_q), ctx.off_out_q);
        encoder.set_buffer(6, Some(ctx.b_out_k), ctx.off_out_k);
        encoder.set_buffer(7, Some(ctx.b_out_v), ctx.off_out_v);

        encoder.set_bytes(8, 4, &ctx.b as *const _ as *const _);
        encoder.set_bytes(9, 4, &ctx.t as *const _ as *const _);
        encoder.set_bytes(10, 4, &ctx.nh as *const _ as *const _);
        encoder.set_bytes(11, 4, &ctx.nkvh as *const _ as *const _);
        encoder.set_bytes(12, 4, &ctx.da as *const _ as *const _);
        encoder.set_bytes(13, 4, &ctx.r as *const _ as *const _);

        let full_d = ctx.da * ctx.r;
        let grid = MTLSize {
            width: ctx.b as u64,
            height: (ctx.nh + 2 * ctx.nkvh) as u64,
            depth: full_d as u64,
        };
        let max_threads = ctx
            .metal_ctx
            .pipeline_forward
            .max_total_threads_per_threadgroup();
        let thread_group = MTLSize {
            width: 1,
            height: 1,
            depth: max_threads.min(full_d as u64),
        };

        encoder.dispatch_threads(grid, thread_group);
        encoder.end_encoding();
    }
}

// --- BACKWARD ENCODING ---
struct EncodingContextFusedBackward<'a> {
    pub metal_ctx: &'static MetalContextFused,
    pub b_q: &'a BufferRef,
    pub off_q: u64,
    pub b_k: &'a BufferRef,
    pub off_k: u64,
    pub b_v: &'a BufferRef,
    pub off_v: u64,
    pub b_logits: &'a BufferRef,
    pub off_logits: u64,
    pub b_inits: &'a BufferRef,
    pub off_inits: u64,
    pub b_hq: &'a BufferRef,
    pub off_hq: u64,
    pub b_hk: &'a BufferRef,
    pub off_hk: u64,
    pub b_hv: &'a BufferRef,
    pub off_hv: u64,
    pub b_gq: &'a BufferRef,
    pub off_gq: u64,
    pub b_gk: &'a BufferRef,
    pub off_gk: u64,
    pub b_gv: &'a BufferRef,
    pub off_gv: u64,
    pub b_dq: &'a BufferRef,
    pub off_dq: u64,
    pub b_dk: &'a BufferRef,
    pub off_dk: u64,
    pub b_dv: &'a BufferRef,
    pub off_dv: u64,
    pub b_dl: &'a BufferRef,
    pub off_dl: u64,
    pub b_di: &'a BufferRef,
    pub off_di: u64,
    pub sq: [u32; 4],
    pub sk: [u32; 4],
    pub sv: [u32; 4],
    pub b: u32,
    pub t: u32,
    pub nh: u32,
    pub nkvh: u32,
    pub da: u32,
    pub r: u32,
}

extern "C" fn encode_fused_backward_callback(cmd_buf_ptr: *mut c_void, context_ptr: *mut c_void) {
    unsafe {
        let cmd_buf = CommandBufferRef::from_ptr(cmd_buf_ptr as *mut _);
        let ctx = &*(context_ptr as *const EncodingContextFusedBackward);
        let encoder = cmd_buf.new_compute_command_encoder();

        // Select pipeline based on r
        let pipeline = if ctx.r == 1 {
            &ctx.metal_ctx.pipeline_backward_direct
        } else {
            &ctx.metal_ctx.pipeline_backward_atomic
        };
        encoder.set_compute_pipeline_state(pipeline);

        encoder.set_buffer(0, Some(ctx.b_q), ctx.off_q);
        encoder.set_buffer(1, Some(ctx.b_k), ctx.off_k);
        encoder.set_buffer(2, Some(ctx.b_v), ctx.off_v);
        encoder.set_buffer(3, Some(ctx.b_logits), ctx.off_logits);
        encoder.set_buffer(4, Some(ctx.b_inits), ctx.off_inits);
        encoder.set_buffer(5, Some(ctx.b_hq), ctx.off_hq);
        encoder.set_buffer(6, Some(ctx.b_hk), ctx.off_hk);
        encoder.set_buffer(7, Some(ctx.b_hv), ctx.off_hv);
        encoder.set_buffer(8, Some(ctx.b_gq), ctx.off_gq);
        encoder.set_buffer(9, Some(ctx.b_gk), ctx.off_gk);
        encoder.set_buffer(10, Some(ctx.b_gv), ctx.off_gv);
        encoder.set_buffer(11, Some(ctx.b_dq), ctx.off_dq);
        encoder.set_buffer(12, Some(ctx.b_dk), ctx.off_dk);
        encoder.set_buffer(13, Some(ctx.b_dv), ctx.off_dv);
        encoder.set_buffer(14, Some(ctx.b_dl), ctx.off_dl);
        encoder.set_buffer(15, Some(ctx.b_di), ctx.off_di);

        encoder.set_bytes(16, 16, ctx.sq.as_ptr() as *const _);
        encoder.set_bytes(17, 16, ctx.sk.as_ptr() as *const _);
        encoder.set_bytes(18, 16, ctx.sv.as_ptr() as *const _);
        encoder.set_bytes(19, 4, &ctx.b as *const _ as *const _);
        encoder.set_bytes(20, 4, &ctx.t as *const _ as *const _);
        encoder.set_bytes(21, 4, &ctx.nh as *const _ as *const _);
        encoder.set_bytes(22, 4, &ctx.nkvh as *const _ as *const _);
        encoder.set_bytes(23, 4, &ctx.da as *const _ as *const _);
        encoder.set_bytes(24, 4, &ctx.r as *const _ as *const _);

        let width: u64 = ctx.b as u64;
        let height: u64 = (ctx.nh + 2 * ctx.nkvh) as u64;
        let full_d: u64 = (ctx.da * ctx.r) as u64;

        let max_threads = pipeline.max_total_threads_per_threadgroup();
        // SIMD groups seem to be unrelated to thread groups, for both dispatch_threads and dispatch_thread_groups.
        // So do not even try to specify an r-aligned thread group size or anything like that.
        encoder.dispatch_threads(
            MTLSize {
                width: width,
                height: height,
                depth: full_d,
            },
            MTLSize {
                width: 1,
                height: 1,
                depth: full_d.min(max_threads),
            },
        );
        encoder.end_encoding();
    }
}

// --- PYTHON FUNCTIONS ---

#[pyfunction(name = "ltv_fused_scan_metal_forward")]
pub fn ltv_fused_scan_metal_forward(
    _py: Python<'_>,
    q: Py<PyAny>,
    k: Py<PyAny>,
    v: Py<PyAny>,
    logits: Py<PyAny>,
    inits: Py<PyAny>,
    out_q: Py<PyAny>,
    out_k: Py<PyAny>,
    out_v: Py<PyAny>,
    b: u32,
    t: u32,
    nh: u32,
    nkvh: u32,
    da: u32,
    r: u32,
) -> PyResult<()> {
    let metal_ctx = get_metal_context()
        .ok_or_else(|| PyErr::new::<PyRuntimeError, _>("Metal context not initialized"))?;

    let (b_q, off_q) = extract_buffer(&q)?;
    let (b_k, off_k) = extract_buffer(&k)?;
    let (b_v, off_v) = extract_buffer(&v)?;
    let (b_logits, off_logits) = extract_buffer(&logits)?;
    let (b_inits, off_inits) = extract_buffer(&inits)?;
    let (b_out_q, off_out_q) = extract_buffer(&out_q)?;
    let (b_out_k, off_out_k) = extract_buffer(&out_k)?;
    let (b_out_v, off_out_v) = extract_buffer(&out_v)?;

    let mut ctx = EncodingContextFusedForward {
        metal_ctx,
        b_q: &b_q,
        off_q,
        b_k: &b_k,
        off_k,
        b_v: &b_v,
        off_v,
        b_logits: &b_logits,
        off_logits,
        b_inits: &b_inits,
        off_inits,
        b_out_q: &b_out_q,
        off_out_q,
        b_out_k: &b_out_k,
        off_out_k,
        b_out_v: &b_out_v,
        off_out_v,
        b,
        t,
        nh,
        nkvh,
        da,
        r,
    };

    unsafe {
        dispatch_on_mps(
            encode_fused_forward_callback,
            &mut ctx as *mut _ as *mut c_void,
        );
    }
    Ok(())
}

#[pyfunction(name = "ltv_fused_scan_metal_backward")]
pub fn ltv_fused_scan_metal_backward(
    _py: Python<'_>,
    q: Py<PyAny>,
    k: Py<PyAny>,
    v: Py<PyAny>,
    logits: Py<PyAny>,
    inits: Py<PyAny>,
    h_fwd_q: Py<PyAny>,
    h_fwd_k: Py<PyAny>,
    h_fwd_v: Py<PyAny>,
    go_q: Py<PyAny>,
    go_k: Py<PyAny>,
    go_v: Py<PyAny>,
    dx_q: Py<PyAny>,
    dx_k: Py<PyAny>,
    dx_v: Py<PyAny>,
    dl: Py<PyAny>,
    di: Py<PyAny>,
    sbq: u32,
    shq: u32,
    stq: u32,
    sdq: u32,
    sbk: u32,
    shk: u32,
    stk: u32,
    sdk: u32,
    sbv: u32,
    shv: u32,
    stv: u32,
    sdv: u32,
    b: u32,
    t: u32,
    nh: u32,
    nkvh: u32,
    da: u32,
    r: u32,
) -> PyResult<()> {
    let metal_ctx = get_metal_context()
        .ok_or_else(|| PyErr::new::<PyRuntimeError, _>("Metal context not initialized"))?;

    let (b_q, off_q) = extract_buffer(&q)?;
    let (b_k, off_k) = extract_buffer(&k)?;
    let (b_v, off_v) = extract_buffer(&v)?;
    let (b_logits, off_logits) = extract_buffer(&logits)?;
    let (b_inits, off_inits) = extract_buffer(&inits)?;
    let (b_hq, off_hq) = extract_buffer(&h_fwd_q)?;
    let (b_hk, off_hk) = extract_buffer(&h_fwd_k)?;
    let (b_hv, off_hv) = extract_buffer(&h_fwd_v)?;
    let (b_gq, off_gq) = extract_buffer(&go_q)?;
    let (b_gk, off_gk) = extract_buffer(&go_k)?;
    let (b_gv, off_gv) = extract_buffer(&go_v)?;
    let (b_dq, off_dq) = extract_buffer(&dx_q)?;
    let (b_dk, off_dk) = extract_buffer(&dx_k)?;
    let (b_dv, off_dv) = extract_buffer(&dx_v)?;
    let (b_dl, off_dl) = extract_buffer(&dl)?;
    let (b_di, off_di) = extract_buffer(&di)?;

    let mut ctx = EncodingContextFusedBackward {
        metal_ctx,
        b_q: &b_q,
        off_q,
        b_k: &b_k,
        off_k,
        b_v: &b_v,
        off_v,
        b_logits: &b_logits,
        off_logits,
        b_inits: &b_inits,
        off_inits,
        b_hq: &b_hq,
        off_hq,
        b_hk: &b_hk,
        off_hk,
        b_hv: &b_hv,
        off_hv,
        b_gq: &b_gq,
        off_gq,
        b_gk: &b_gk,
        off_gk,
        b_gv: &b_gv,
        off_gv,
        b_dq: &b_dq,
        off_dq,
        b_dk: &b_dk,
        off_dk,
        b_dv: &b_dv,
        off_dv,
        b_dl: &b_dl,
        off_dl,
        b_di: &b_di,
        off_di,
        sq: [sbq, shq, stq, sdq],
        sk: [sbk, shk, stk, sdk],
        sv: [sbv, shv, stv, sdv],
        b,
        t,
        nh,
        nkvh,
        da,
        r,
    };

    unsafe {
        dispatch_on_mps(
            encode_fused_backward_callback,
            &mut ctx as *mut _ as *mut c_void,
        );
    }
    Ok(())
}

#[pyfunction(name = "init_ltv_fused_scan_metal_context_for_default_device")]
pub fn init_ltv_fused_scan_metal_context_for_default_device() -> PyResult<()> {
    let device = Device::system_default()
        .ok_or_else(|| PyErr::new::<PyRuntimeError, _>("No Metal device found"))?;

    let lib_fwd = device
        .new_library_with_source(FORWARD_KERNEL_SOURCE, &CompileOptions::new())
        .map_err(|e| PyErr::new::<PyRuntimeError, _>(e.to_string()))?;
    let pipeline_forward = device
        .new_compute_pipeline_state_with_function(
            &lib_fwd
                .get_function("ltv_fused_scan_forward", None)
                .unwrap(),
        )
        .map_err(|e| PyErr::new::<PyRuntimeError, _>(e.to_string()))?;

    let lib_bwd = device
        .new_library_with_source(BACKWARD_KERNEL_SOURCE, &CompileOptions::new())
        .map_err(|e| PyErr::new::<PyRuntimeError, _>(e.to_string()))?;

    let pipeline_backward_direct = device
        .new_compute_pipeline_state_with_function(
            &lib_bwd
                .get_function("ltv_fused_scan_backward_direct", None)
                .unwrap(),
        )
        .map_err(|e| PyErr::new::<PyRuntimeError, _>(e.to_string()))?;

    let pipeline_backward_atomic = device
        .new_compute_pipeline_state_with_function(
            &lib_bwd
                .get_function("ltv_fused_scan_backward_atomic", None)
                .unwrap(),
        )
        .map_err(|e| PyErr::new::<PyRuntimeError, _>(e.to_string()))?;

    set_metal_context(MetalContextFused {
        device,
        pipeline_forward,
        pipeline_backward_direct,
        pipeline_backward_atomic,
    })
    .map_err(|_| PyErr::new::<PyRuntimeError, _>("Metal context already initialized"))?;

    Ok(())
}

#[pyfunction(name = "release_ltv_fused_scan_metal_context")]
pub fn release_ltv_fused_scan_metal_context() {}
