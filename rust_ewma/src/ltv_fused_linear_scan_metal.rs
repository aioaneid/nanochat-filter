#![cfg(all(feature = "metal", feature = "python"))]

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
    pub pipeline_backward: ComputePipelineState,
}

static CONTEXT: OnceLock<MetalContextFused> = OnceLock::new();

const FORWARD_KERNEL_SOURCE: &str = include_str!("ltv_fused_linear_scan_forward.msl");
const BACKWARD_KERNEL_SOURCE: &str = include_str!("ltv_fused_linear_scan_backward.msl");

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
    pub b_xgate: &'a BufferRef,
    pub off_xgate: u64,
    pub b_wgate: &'a BufferRef,
    pub off_wgate: u64,
    pub b_bgate: &'a BufferRef,
    pub off_bgate: u64,
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
    pub n_embd: u32,
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
        encoder.set_buffer(3, Some(ctx.b_xgate), ctx.off_xgate);
        encoder.set_buffer(4, Some(ctx.b_wgate), ctx.off_wgate);
        encoder.set_buffer(5, Some(ctx.b_bgate), ctx.off_bgate);
        encoder.set_buffer(6, Some(ctx.b_inits), ctx.off_inits);
        encoder.set_buffer(7, Some(ctx.b_out_q), ctx.off_out_q);
        encoder.set_buffer(8, Some(ctx.b_out_k), ctx.off_out_k);
        encoder.set_buffer(9, Some(ctx.b_out_v), ctx.off_out_v);

        encoder.set_bytes(10, 4, &ctx.b as *const _ as *const _);
        encoder.set_bytes(11, 4, &ctx.t as *const _ as *const _);
        encoder.set_bytes(12, 4, &ctx.nh as *const _ as *const _);
        encoder.set_bytes(13, 4, &ctx.nkvh as *const _ as *const _);
        encoder.set_bytes(14, 4, &ctx.da as *const _ as *const _);
        encoder.set_bytes(15, 4, &ctx.r as *const _ as *const _);
        encoder.set_bytes(16, 4, &ctx.n_embd as *const _ as *const _);

        let full_d = (ctx.da * ctx.r) as u64;
        let grid = MTLSize {
            width: ctx.b as u64,
            height: (ctx.nh + 2 * ctx.nkvh) as u64,
            depth: full_d,
        };
        let max_threads = ctx
            .metal_ctx
            .pipeline_forward
            .max_total_threads_per_threadgroup();
        encoder.dispatch_threads(
            grid,
            MTLSize {
                width: 1,
                height: 1,
                depth: full_d.min(max_threads),
            },
        );
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
    pub b_xgate: &'a BufferRef,
    pub off_xgate: u64,
    pub b_wgate: &'a BufferRef,
    pub off_wgate: u64,
    pub b_bgate: &'a BufferRef,
    pub off_bgate: u64,
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
    pub b_dxgate: &'a BufferRef,
    pub off_dxgate: u64,
    pub b_dwgate: &'a BufferRef,
    pub off_dwgate: u64,
    pub b_dbgate: &'a BufferRef,
    pub off_dbgate: u64,
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
        encoder.set_compute_pipeline_state(&ctx.metal_ctx.pipeline_backward);

        encoder.set_buffer(0, Some(ctx.b_q), ctx.off_q);
        encoder.set_buffer(1, Some(ctx.b_k), ctx.off_k);
        encoder.set_buffer(2, Some(ctx.b_v), ctx.off_v);
        encoder.set_buffer(3, Some(ctx.b_xgate), ctx.off_xgate);
        encoder.set_buffer(4, Some(ctx.b_wgate), ctx.off_wgate);
        encoder.set_buffer(5, Some(ctx.b_bgate), ctx.off_bgate);
        encoder.set_buffer(6, Some(ctx.b_inits), ctx.off_inits);
        encoder.set_buffer(7, Some(ctx.b_hq), ctx.off_hq);
        encoder.set_buffer(8, Some(ctx.b_hk), ctx.off_hk);
        encoder.set_buffer(9, Some(ctx.b_hv), ctx.off_hv);
        encoder.set_buffer(10, Some(ctx.b_gq), ctx.off_gq);
        encoder.set_buffer(11, Some(ctx.b_gk), ctx.off_gk);
        encoder.set_buffer(12, Some(ctx.b_gv), ctx.off_gv);
        encoder.set_buffer(13, Some(ctx.b_dq), ctx.off_dq);
        encoder.set_buffer(14, Some(ctx.b_dk), ctx.off_dk);
        encoder.set_buffer(15, Some(ctx.b_dv), ctx.off_dv);
        encoder.set_buffer(16, Some(ctx.b_dxgate), ctx.off_dxgate);
        encoder.set_buffer(17, Some(ctx.b_dwgate), ctx.off_dwgate);
        encoder.set_buffer(18, Some(ctx.b_dbgate), ctx.off_dbgate);
        encoder.set_buffer(19, Some(ctx.b_di), ctx.off_di);

        // Strides (passed as uint4/16 bytes each)
        encoder.set_bytes(20, 16, ctx.sq.as_ptr() as *const _);
        encoder.set_bytes(21, 16, ctx.sk.as_ptr() as *const _);
        encoder.set_bytes(22, 16, ctx.sv.as_ptr() as *const _);

        encoder.set_bytes(23, 4, &ctx.b as *const _ as *const _);
        encoder.set_bytes(24, 4, &ctx.t as *const _ as *const _);
        encoder.set_bytes(25, 4, &ctx.nh as *const _ as *const _);
        encoder.set_bytes(26, 4, &ctx.nkvh as *const _ as *const _);
        encoder.set_bytes(27, 4, &ctx.da as *const _ as *const _);
        encoder.set_bytes(28, 4, &ctx.r as *const _ as *const _);

        let full_d = (ctx.da * ctx.r) as u64;
        let max_threads = ctx
            .metal_ctx
            .pipeline_backward
            .max_total_threads_per_threadgroup();
        encoder.dispatch_threads(
            MTLSize {
                width: ctx.b as u64,
                height: (ctx.nh + 2 * ctx.nkvh) as u64,
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

#[pyfunction(name = "ltv_fused_linear_scan_metal_forward")]
pub fn ltv_fused_linear_scan_metal_forward(
    _py: Python<'_>,
    q: Py<PyAny>,
    k: Py<PyAny>,
    v: Py<PyAny>,
    x_gate: Py<PyAny>,
    w_gate: Py<PyAny>,
    b_gate: Py<PyAny>,
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
    n_embd: u32,
) -> PyResult<()> {
    let metal_ctx = get_metal_context()
        .ok_or_else(|| PyErr::new::<PyRuntimeError, _>("Metal context not initialized"))?;

    let (b_q, off_q) = extract_buffer(&q)?;
    let (b_k, off_k) = extract_buffer(&k)?;
    let (b_v, off_v) = extract_buffer(&v)?;
    let (b_xg, off_xg) = extract_buffer(&x_gate)?;
    let (b_wg, off_wg) = extract_buffer(&w_gate)?;
    let (b_bg, off_bg) = extract_buffer(&b_gate)?;
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
        b_xgate: &b_xg,
        off_xgate: off_xg,
        b_wgate: &b_wg,
        off_wgate: off_wg,
        b_bgate: &b_bg,
        off_bgate: off_bg,
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
        n_embd,
    };

    unsafe {
        dispatch_on_mps(
            encode_fused_forward_callback,
            &mut ctx as *mut _ as *mut c_void,
        );
    }
    Ok(())
}

#[pyfunction(name = "ltv_fused_linear_scan_metal_backward")]
pub fn ltv_fused_linear_scan_metal_backward(
    _py: Python<'_>,
    q: Py<PyAny>,
    k: Py<PyAny>,
    v: Py<PyAny>,
    x_gate: Py<PyAny>,
    w_gate: Py<PyAny>,
    b_gate: Py<PyAny>,
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
    dx_gate: Py<PyAny>,
    dw_gate: Py<PyAny>,
    db_gate: Py<PyAny>,
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
    let (b_xg, off_xg) = extract_buffer(&x_gate)?;
    let (b_wg, off_wg) = extract_buffer(&w_gate)?;
    let (b_bg, off_bg) = extract_buffer(&b_gate)?;
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
    let (b_dxg, off_dxg) = extract_buffer(&dx_gate)?;
    let (b_dw, off_dw) = extract_buffer(&dw_gate)?;
    let (b_db, off_db) = extract_buffer(&db_gate)?;
    let (b_di, off_di) = extract_buffer(&di)?;

    let mut ctx = EncodingContextFusedBackward {
        metal_ctx,
        b_q: &b_q,
        off_q,
        b_k: &b_k,
        off_k,
        b_v: &b_v,
        off_v,
        b_xgate: &b_xg,
        off_xgate: off_xg,
        b_wgate: &b_wg,
        off_wgate: off_wg,
        b_bgate: &b_bg,
        off_bgate: off_bg,
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
        b_dxgate: &b_dxg,
        off_dxgate: off_dxg,
        b_dwgate: &b_dw,
        off_dwgate: off_dw,
        b_dbgate: &b_db,
        off_dbgate: off_db,
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

#[pyfunction(name = "init_ltv_fused_linear_scan_metal_context_for_default_device")]
pub fn init_ltv_fused_linear_scan_metal_context_for_default_device() -> PyResult<()> {
    let device = Device::system_default()
        .ok_or_else(|| PyErr::new::<PyRuntimeError, _>("No Metal device found"))?;

    let lib_fwd = device
        .new_library_with_source(FORWARD_KERNEL_SOURCE, &CompileOptions::new())
        .map_err(|e| PyErr::new::<PyRuntimeError, _>(e.to_string()))?;
    let pipeline_forward = device
        .new_compute_pipeline_state_with_function(
            &lib_fwd
                .get_function("ltv_fused_linear_scan_forward", None)
                .unwrap(),
        )
        .map_err(|e| PyErr::new::<PyRuntimeError, _>(e.to_string()))?;

    let lib_bwd = device
        .new_library_with_source(BACKWARD_KERNEL_SOURCE, &CompileOptions::new())
        .map_err(|e| PyErr::new::<PyRuntimeError, _>(e.to_string()))?;
    let pipeline_backward = device
        .new_compute_pipeline_state_with_function(
            &lib_bwd
                .get_function("ltv_fused_linear_scan_backward", None)
                .unwrap(),
        )
        .map_err(|e| PyErr::new::<PyRuntimeError, _>(e.to_string()))?;

    set_metal_context(MetalContextFused {
        device,
        pipeline_forward,
        pipeline_backward,
    })
    .map_err(|_| PyErr::new::<PyRuntimeError, _>("Metal context already initialized"))?;

    Ok(())
}

#[pyfunction(name = "release_ltv_fused_linear_scan_metal_context")]
pub fn release_ltv_fused_linear_scan_metal_context() {
    // Drop implementation if needed, though OnceLock is static
}
