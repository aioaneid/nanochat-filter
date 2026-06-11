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

// --- CONTEXT (original float+bias kernels) ---
struct MetalContextFused {
    pub device: Device,
    pub pipeline_forward: ComputePipelineState,
    pub pipeline_backward_atomic: ComputePipelineState,
    pub pipeline_backward_direct: ComputePipelineState,
}

static CONTEXT: OnceLock<MetalContextFused> = OnceLock::new();

const FORWARD_KERNEL_SOURCE: &str = include_str!("ltv_fused_concat_scan_forward.msl");
const BACKWARD_COMMON_H: &str = include_str!("ltv_fused_scan_backward_common.h");
const BACKWARD_DIRECT_SRC: &str = include_str!("ltv_fused_concat_scan_backward_direct.msl");
const BACKWARD_ATOMIC_SRC: &str = include_str!("ltv_fused_concat_scan_backward_atomic.msl");

fn set_metal_context(context: MetalContextFused) -> Result<(), MetalContextFused> {
    CONTEXT.set(context)
}

fn get_metal_context() -> Option<&'static MetalContextFused> {
    CONTEXT.get()
}

// --- CONTEXT (register-cast bfloat16 kernels, no bias) ---
struct MetalContextRegisterCast {
    pub device: Device,
    pub pipeline_forward: ComputePipelineState,
    pub pipeline_backward_atomic: ComputePipelineState,
    pub pipeline_backward_direct: ComputePipelineState,
}

static CONTEXT_RC: OnceLock<MetalContextRegisterCast> = OnceLock::new();

const RC_FORWARD_KERNEL_SOURCE: &str =
    include_str!("ltv_fused_concat_register_cast_scan_forward.msl");
const RC_BACKWARD_DIRECT_SRC: &str =
    include_str!("ltv_fused_concat_register_cast_scan_backward_direct.msl");
const RC_BACKWARD_ATOMIC_SRC: &str =
    include_str!("ltv_fused_concat_register_cast_scan_backward_atomic.msl");

fn set_metal_context_rc(context: MetalContextRegisterCast) -> Result<(), MetalContextRegisterCast> {
    CONTEXT_RC.set(context)
}

fn get_metal_context_rc() -> Option<&'static MetalContextRegisterCast> {
    CONTEXT_RC.get()
}

// --- FORWARD ENCODING ---
struct EncodingContextFusedConcatForward<'a> {
    pub metal_ctx: &'static MetalContextFused,
    pub b_combined: &'a BufferRef,
    pub off_combined: u64,
    pub b_bias: &'a BufferRef,
    pub off_bias: u64,
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
    pub sc_combined: [u32; 3],
    pub nh: u32,
    pub nkvh: u32,
    pub da: u32,
    pub r: u32,
}

extern "C" fn encode_fused_concat_forward_callback(
    cmd_buf_ptr: *mut c_void,
    context_ptr: *mut c_void,
) {
    unsafe {
        let cmd_buf = CommandBufferRef::from_ptr(cmd_buf_ptr as *mut _);
        let ctx = &*(context_ptr as *const EncodingContextFusedConcatForward);
        let encoder = cmd_buf.new_compute_command_encoder();

        let pipeline = &ctx.metal_ctx.pipeline_forward;
        encoder.set_compute_pipeline_state(pipeline);

        encoder.set_buffer(0, Some(ctx.b_combined), ctx.off_combined);
        encoder.set_buffer(1, Some(ctx.b_bias), ctx.off_bias);
        encoder.set_buffer(2, Some(ctx.b_inits), ctx.off_inits);
        encoder.set_buffer(3, Some(ctx.b_out_q), ctx.off_out_q);
        encoder.set_buffer(4, Some(ctx.b_out_k), ctx.off_out_k);
        encoder.set_buffer(5, Some(ctx.b_out_v), ctx.off_out_v);

        encoder.set_bytes(6, 12, ctx.sc_combined.as_ptr() as *const _);
        encoder.set_bytes(7, 4, &ctx.b as *const _ as *const _);
        encoder.set_bytes(8, 4, &ctx.t as *const _ as *const _);
        encoder.set_bytes(9, 4, &ctx.nh as *const _ as *const _);
        encoder.set_bytes(10, 4, &ctx.nkvh as *const _ as *const _);
        encoder.set_bytes(11, 4, &ctx.da as *const _ as *const _);
        encoder.set_bytes(12, 4, &ctx.r as *const _ as *const _);

        let full_d = (ctx.da * ctx.r) as u64;
        let grid = MTLSize {
            width: ctx.b as u64,
            height: (ctx.nh + 2 * ctx.nkvh) as u64,
            depth: full_d,
        };
        let max_threads = pipeline.max_total_threads_per_threadgroup();
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
struct EncodingContextFusedConcatBackward<'a> {
    pub pipeline: &'a ComputePipelineState,
    pub b_combined: &'a BufferRef,
    pub off_combined: u64,
    pub b_bias: &'a BufferRef,
    pub off_bias: u64,
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
    pub b_d_comb: &'a BufferRef,
    pub off_d_comb: u64,
    pub b_di: &'a BufferRef,
    pub off_di: u64,
    pub sc_combined: [u32; 3],
    pub sgc_combined: [u32; 3],
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

extern "C" fn encode_fused_concat_backward_callback(
    cmd_buf_ptr: *mut c_void,
    context_ptr: *mut c_void,
) {
    unsafe {
        let cmd_buf = CommandBufferRef::from_ptr(cmd_buf_ptr as *mut _);
        let ctx = &*(context_ptr as *const EncodingContextFusedConcatBackward);
        let encoder = cmd_buf.new_compute_command_encoder();

        encoder.set_compute_pipeline_state(ctx.pipeline);

        encoder.set_buffer(0, Some(ctx.b_combined), ctx.off_combined);
        encoder.set_buffer(1, Some(ctx.b_bias), ctx.off_bias);
        encoder.set_buffer(2, Some(ctx.b_inits), ctx.off_inits);
        encoder.set_buffer(3, Some(ctx.b_hq), ctx.off_hq);
        encoder.set_buffer(4, Some(ctx.b_hk), ctx.off_hk);
        encoder.set_buffer(5, Some(ctx.b_hv), ctx.off_hv);
        encoder.set_buffer(6, Some(ctx.b_gq), ctx.off_gq);
        encoder.set_buffer(7, Some(ctx.b_gk), ctx.off_gk);
        encoder.set_buffer(8, Some(ctx.b_gv), ctx.off_gv);
        encoder.set_buffer(9, Some(ctx.b_d_comb), ctx.off_d_comb);
        encoder.set_buffer(10, Some(ctx.b_di), ctx.off_di);

        encoder.set_bytes(11, 12, ctx.sc_combined.as_ptr() as *const _);
        encoder.set_bytes(12, 12, ctx.sgc_combined.as_ptr() as *const _);
        encoder.set_bytes(13, 16, ctx.sq.as_ptr() as *const _);
        encoder.set_bytes(14, 16, ctx.sk.as_ptr() as *const _);
        encoder.set_bytes(15, 16, ctx.sv.as_ptr() as *const _);
        encoder.set_bytes(16, 4, &ctx.b as *const _ as *const _);
        encoder.set_bytes(17, 4, &ctx.t as *const _ as *const _);
        encoder.set_bytes(18, 4, &ctx.nh as *const _ as *const _);
        encoder.set_bytes(19, 4, &ctx.nkvh as *const _ as *const _);
        encoder.set_bytes(20, 4, &ctx.da as *const _ as *const _);
        encoder.set_bytes(21, 4, &ctx.r as *const _ as *const _);

        // Set threadgroup memory only if r > 1 (atomic path)
        if ctx.r > 1 {
            encoder.set_threadgroup_memory_length(0, (ctx.da * 4) as u64);
        }

        let full_d = (ctx.da * ctx.r) as u64;
        let grid = MTLSize {
            width: ctx.b as u64,
            height: (ctx.nh + 2 * ctx.nkvh) as u64,
            depth: full_d,
        };

        let max_threads = ctx.pipeline.max_total_threads_per_threadgroup();
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

// --- REGISTER-CAST FORWARD ENCODING (no bias, bfloat16) ---
struct EncodingContextRcForward<'a> {
    pub metal_ctx: &'static MetalContextRegisterCast,
    pub b_combined: &'a BufferRef,
    pub off_combined: u64,
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
    pub sc_combined: [u32; 3],
    pub nh: u32,
    pub nkvh: u32,
    pub da: u32,
    pub r: u32,
}

extern "C" fn encode_rc_forward_callback(cmd_buf_ptr: *mut c_void, context_ptr: *mut c_void) {
    unsafe {
        let cmd_buf = CommandBufferRef::from_ptr(cmd_buf_ptr as *mut _);
        let ctx = &*(context_ptr as *const EncodingContextRcForward);
        let encoder = cmd_buf.new_compute_command_encoder();

        let pipeline = &ctx.metal_ctx.pipeline_forward;
        encoder.set_compute_pipeline_state(pipeline);

        encoder.set_buffer(0, Some(ctx.b_combined), ctx.off_combined);
        encoder.set_buffer(1, Some(ctx.b_inits), ctx.off_inits);
        encoder.set_buffer(2, Some(ctx.b_out_q), ctx.off_out_q);
        encoder.set_buffer(3, Some(ctx.b_out_k), ctx.off_out_k);
        encoder.set_buffer(4, Some(ctx.b_out_v), ctx.off_out_v);

        encoder.set_bytes(5, 12, ctx.sc_combined.as_ptr() as *const _);
        encoder.set_bytes(6, 4, &ctx.t as *const _ as *const _);
        encoder.set_bytes(7, 4, &ctx.nh as *const _ as *const _);
        encoder.set_bytes(8, 4, &ctx.nkvh as *const _ as *const _);
        encoder.set_bytes(9, 4, &ctx.da as *const _ as *const _);
        encoder.set_bytes(10, 4, &ctx.r as *const _ as *const _);

        let full_d = (ctx.da * ctx.r) as u64;
        let grid = MTLSize {
            width: ctx.b as u64,
            height: (ctx.nh + 2 * ctx.nkvh) as u64,
            depth: full_d,
        };
        let max_threads = pipeline.max_total_threads_per_threadgroup();
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

// --- REGISTER-CAST BACKWARD ENCODING (no bias, bfloat16, single kernel) ---
struct EncodingContextRcBackward<'a> {
    pub metal_ctx: &'static MetalContextRegisterCast,
    pub b_combined: &'a BufferRef,
    pub off_combined: u64,
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
    pub b_d_comb: &'a BufferRef,
    pub off_d_comb: u64,
    pub b_di: &'a BufferRef,
    pub off_di: u64,
    pub sc_combined: [u32; 3],
    pub sgc_combined: [u32; 3],
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

extern "C" fn encode_rc_backward_callback(cmd_buf_ptr: *mut c_void, context_ptr: *mut c_void) {
    unsafe {
        let cmd_buf = CommandBufferRef::from_ptr(cmd_buf_ptr as *mut _);
        let ctx = &*(context_ptr as *const EncodingContextRcBackward);
        let encoder = cmd_buf.new_compute_command_encoder();

        if ctx.r == 1 {
            encoder.set_compute_pipeline_state(&ctx.metal_ctx.pipeline_backward_direct);
        } else {
            encoder.set_compute_pipeline_state(&ctx.metal_ctx.pipeline_backward_atomic);
            encoder.set_threadgroup_memory_length(0, (ctx.da * 4) as u64);
        }

        encoder.set_buffer(0, Some(ctx.b_combined), ctx.off_combined);
        encoder.set_buffer(1, Some(ctx.b_inits), ctx.off_inits);
        encoder.set_buffer(2, Some(ctx.b_hq), ctx.off_hq);
        encoder.set_buffer(3, Some(ctx.b_hk), ctx.off_hk);
        encoder.set_buffer(4, Some(ctx.b_hv), ctx.off_hv);
        encoder.set_buffer(5, Some(ctx.b_gq), ctx.off_gq);
        encoder.set_buffer(6, Some(ctx.b_gk), ctx.off_gk);
        encoder.set_buffer(7, Some(ctx.b_gv), ctx.off_gv);
        encoder.set_buffer(8, Some(ctx.b_d_comb), ctx.off_d_comb);
        encoder.set_buffer(9, Some(ctx.b_di), ctx.off_di);

        encoder.set_bytes(10, 12, ctx.sc_combined.as_ptr() as *const _);
        encoder.set_bytes(11, 12, ctx.sgc_combined.as_ptr() as *const _);
        encoder.set_bytes(12, 16, ctx.sq.as_ptr() as *const _);
        encoder.set_bytes(13, 16, ctx.sk.as_ptr() as *const _);
        encoder.set_bytes(14, 16, ctx.sv.as_ptr() as *const _);
        encoder.set_bytes(15, 4, &ctx.t as *const _ as *const _);
        encoder.set_bytes(16, 4, &ctx.nh as *const _ as *const _);
        encoder.set_bytes(17, 4, &ctx.nkvh as *const _ as *const _);
        encoder.set_bytes(18, 4, &ctx.da as *const _ as *const _);
        encoder.set_bytes(19, 4, &ctx.r as *const _ as *const _);

        let full_d = (ctx.da * ctx.r) as u64;
        let grid = MTLSize {
            width: ctx.b as u64,
            height: (ctx.nh + 2 * ctx.nkvh) as u64,
            depth: full_d,
        };

        let max_threads = if ctx.r == 1 {
            ctx.metal_ctx
                .pipeline_backward_direct
                .max_total_threads_per_threadgroup()
        } else {
            ctx.metal_ctx
                .pipeline_backward_atomic
                .max_total_threads_per_threadgroup()
        };
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

// --- PYTHON FUNCTIONS ---

#[pyfunction(name = "ltv_fused_concat_scan_metal_forward")]
pub fn ltv_fused_concat_scan_metal_forward(
    _py: Python<'_>,
    combined: Py<PyAny>,
    logit_bias: Py<PyAny>,
    inits: Py<PyAny>,
    out_q: Py<PyAny>,
    out_k: Py<PyAny>,
    out_v: Py<PyAny>,
    b: u32,
    t: u32,
    sc_b: u32,
    sc_t: u32,
    sc_d: u32,
    nh: u32,
    nkvh: u32,
    da: u32,
    r: u32,
) -> PyResult<()> {
    let metal_ctx =
        get_metal_context().ok_or_else(|| PyErr::new::<PyRuntimeError, _>("Not init"))?;

    let (b_comb, off_comb) = extract_buffer(&combined)?;
    let (b_bias, off_bias) = extract_buffer(&logit_bias)?;
    let (b_inits, off_inits) = extract_buffer(&inits)?;
    let (b_oq, off_oq) = extract_buffer(&out_q)?;
    let (b_ok, off_ok) = extract_buffer(&out_k)?;
    let (b_ov, off_ov) = extract_buffer(&out_v)?;

    let mut ctx = EncodingContextFusedConcatForward {
        metal_ctx,
        b_combined: &b_comb,
        off_combined: off_comb,
        b_bias: &b_bias,
        off_bias: off_bias,
        b_inits: &b_inits,
        off_inits: off_inits,
        b_out_q: &b_oq,
        off_out_q: off_oq,
        b_out_k: &b_ok,
        off_out_k: off_ok,
        b_out_v: &b_ov,
        off_out_v: off_ov,
        b,
        t,
        sc_combined: [sc_b, sc_t, sc_d],
        nh,
        nkvh,
        da,
        r,
    };

    unsafe {
        dispatch_on_mps(
            encode_fused_concat_forward_callback,
            &mut ctx as *mut _ as *mut c_void,
        );
    }
    Ok(())
}

#[pyfunction(name = "ltv_fused_concat_scan_metal_backward")]
pub fn ltv_fused_concat_scan_metal_backward(
    _py: Python<'_>,
    combined: Py<PyAny>,
    logit_bias: Py<PyAny>,
    inits: Py<PyAny>,
    h_fwd_q: Py<PyAny>,
    h_fwd_k: Py<PyAny>,
    h_fwd_v: Py<PyAny>,
    go_q: Py<PyAny>,
    go_k: Py<PyAny>,
    go_v: Py<PyAny>,
    grad_combined: Py<PyAny>,
    grad_inits: Py<PyAny>,
    b: u32,
    t: u32,
    sc_b: u32,
    sc_t: u32,
    sc_d: u32,
    sgc_b: u32,
    sgc_t: u32,
    sgc_d: u32,
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
    nh: u32,
    nkvh: u32,
    da: u32,
    r: u32,
) -> PyResult<()> {
    let metal_ctx =
        get_metal_context().ok_or_else(|| PyErr::new::<PyRuntimeError, _>("Not init"))?;

    let (b_comb, off_comb) = extract_buffer(&combined)?;
    let (b_bias, off_bias) = extract_buffer(&logit_bias)?;
    let (b_inits, off_inits) = extract_buffer(&inits)?;
    let (b_hq, off_hq) = extract_buffer(&h_fwd_q)?;
    let (b_hk, off_hk) = extract_buffer(&h_fwd_k)?;
    let (b_hv, off_hv) = extract_buffer(&h_fwd_v)?;
    let (b_gq, off_gq) = extract_buffer(&go_q)?;
    let (b_gk, off_gk) = extract_buffer(&go_k)?;
    let (b_gv, off_gv) = extract_buffer(&go_v)?;
    let (b_dc, off_dc) = extract_buffer(&grad_combined)?;
    let (b_di, off_di) = extract_buffer(&grad_inits)?;

    // Select the correct pipeline
    let pipeline = if r == 1 {
        &metal_ctx.pipeline_backward_direct
    } else {
        &metal_ctx.pipeline_backward_atomic
    };

    let mut ctx = EncodingContextFusedConcatBackward {
        pipeline,
        b_combined: &b_comb,
        off_combined: off_comb,
        b_bias: &b_bias,
        off_bias: off_bias,
        b_inits: &b_inits,
        off_inits: off_inits,
        b_hq: &b_hq,
        off_hq: off_hq,
        b_hk: &b_hk,
        off_hk: off_hk,
        b_hv: &b_hv,
        off_hv: off_hv,
        b_gq: &b_gq,
        off_gq: off_gq,
        b_gk: &b_gk,
        off_gk: off_gk,
        b_gv: &b_gv,
        off_gv: off_gv,
        b_d_comb: &b_dc,
        off_d_comb: off_dc,
        b_di: &b_di,
        off_di: off_di,
        sc_combined: [sc_b, sc_t, sc_d],
        sgc_combined: [sgc_b, sgc_t, sgc_d],
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
            encode_fused_concat_backward_callback,
            &mut ctx as *mut _ as *mut c_void,
        );
    }
    Ok(())
}

#[pyfunction(name = "init_ltv_fused_concat_scan_metal_context_for_default_device")]
pub fn init_ltv_fused_concat_scan_metal_context_for_default_device() -> PyResult<()> {
    let device = Device::system_default()
        .ok_or_else(|| PyErr::new::<PyRuntimeError, _>("No Metal device"))?;

    let lib_fwd = device
        .new_library_with_source(FORWARD_KERNEL_SOURCE, &CompileOptions::new())
        .map_err(|e| PyErr::new::<PyRuntimeError, _>(e.to_string()))?;
    let pipeline_forward = device
        .new_compute_pipeline_state_with_function(
            &lib_fwd
                .get_function("ltv_fused_concat_scan_forward", None)
                .unwrap(),
        )
        .map_err(|e| PyErr::new::<PyRuntimeError, _>(e.to_string()))?;

    let lib_back_direct = device
        .new_library_with_source(
            &format!("{}\n{}", BACKWARD_COMMON_H, BACKWARD_DIRECT_SRC),
            &CompileOptions::new(),
        )
        .map_err(|e| PyErr::new::<PyRuntimeError, _>(e.to_string()))?;
    let lib_back_atomic = device
        .new_library_with_source(
            &format!("{}\n{}", BACKWARD_COMMON_H, BACKWARD_ATOMIC_SRC),
            &CompileOptions::new(),
        )
        .map_err(|e| PyErr::new::<PyRuntimeError, _>(e.to_string()))?;

    let pipeline_backward_direct = device
        .new_compute_pipeline_state_with_function(
            &lib_back_direct
                .get_function("ltv_fused_concat_scan_backward_direct", None)
                .unwrap(),
        )
        .map_err(|e| PyErr::new::<PyRuntimeError, _>(e.to_string()))?;

    let pipeline_backward_atomic = device
        .new_compute_pipeline_state_with_function(
            &lib_back_atomic
                .get_function("ltv_fused_concat_scan_backward_atomic", None)
                .unwrap(),
        )
        .map_err(|e| PyErr::new::<PyRuntimeError, _>(e.to_string()))?;

    set_metal_context(MetalContextFused {
        device,
        pipeline_forward,
        pipeline_backward_atomic,
        pipeline_backward_direct,
    })
    .map_err(|_| PyErr::new::<PyRuntimeError, _>("Already init"))?;

    Ok(())
}

#[pyfunction(name = "release_ltv_fused_concat_scan_metal_context")]
pub fn release_ltv_fused_concat_scan_metal_context() {}

// --- REGISTER-CAST PYTHON FUNCTIONS (bfloat16, no bias) ---

#[pyfunction(name = "ltv_fused_concat_register_cast_scan_metal_forward")]
pub fn ltv_fused_concat_register_cast_scan_metal_forward(
    _py: Python<'_>,
    combined: Py<PyAny>,
    inits: Py<PyAny>,
    out_q: Py<PyAny>,
    out_k: Py<PyAny>,
    out_v: Py<PyAny>,
    b: u32,
    t: u32,
    sc_b: u32,
    sc_t: u32,
    sc_d: u32,
    nh: u32,
    nkvh: u32,
    da: u32,
    r: u32,
) -> PyResult<()> {
    let metal_ctx =
        get_metal_context_rc().ok_or_else(|| PyErr::new::<PyRuntimeError, _>("Not init (rc)"))?;

    let (b_comb, off_comb) = extract_buffer(&combined)?;
    let (b_inits, off_inits) = extract_buffer(&inits)?;
    let (b_oq, off_oq) = extract_buffer(&out_q)?;
    let (b_ok, off_ok) = extract_buffer(&out_k)?;
    let (b_ov, off_ov) = extract_buffer(&out_v)?;

    let mut ctx = EncodingContextRcForward {
        metal_ctx,
        b_combined: &b_comb,
        off_combined: off_comb,
        b_inits: &b_inits,
        off_inits: off_inits,
        b_out_q: &b_oq,
        off_out_q: off_oq,
        b_out_k: &b_ok,
        off_out_k: off_ok,
        b_out_v: &b_ov,
        off_out_v: off_ov,
        b,
        t,
        sc_combined: [sc_b, sc_t, sc_d],
        nh,
        nkvh,
        da,
        r,
    };

    unsafe {
        dispatch_on_mps(
            encode_rc_forward_callback,
            &mut ctx as *mut _ as *mut c_void,
        );
    }
    Ok(())
}

#[pyfunction(name = "ltv_fused_concat_register_cast_scan_metal_backward")]
pub fn ltv_fused_concat_register_cast_scan_metal_backward(
    _py: Python<'_>,
    combined: Py<PyAny>,
    inits: Py<PyAny>,
    h_fwd_q: Py<PyAny>,
    h_fwd_k: Py<PyAny>,
    h_fwd_v: Py<PyAny>,
    go_q: Py<PyAny>,
    go_k: Py<PyAny>,
    go_v: Py<PyAny>,
    grad_combined: Py<PyAny>,
    grad_inits: Py<PyAny>,
    b: u32,
    t: u32,
    sc_b: u32,
    sc_t: u32,
    sc_d: u32,
    sgc_b: u32,
    sgc_t: u32,
    sgc_d: u32,
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
    nh: u32,
    nkvh: u32,
    da: u32,
    r: u32,
) -> PyResult<()> {
    let metal_ctx =
        get_metal_context_rc().ok_or_else(|| PyErr::new::<PyRuntimeError, _>("Not init (rc)"))?;

    let (b_comb, off_comb) = extract_buffer(&combined)?;
    let (b_inits, off_inits) = extract_buffer(&inits)?;
    let (b_hq, off_hq) = extract_buffer(&h_fwd_q)?;
    let (b_hk, off_hk) = extract_buffer(&h_fwd_k)?;
    let (b_hv, off_hv) = extract_buffer(&h_fwd_v)?;
    let (b_gq, off_gq) = extract_buffer(&go_q)?;
    let (b_gk, off_gk) = extract_buffer(&go_k)?;
    let (b_gv, off_gv) = extract_buffer(&go_v)?;
    let (b_dc, off_dc) = extract_buffer(&grad_combined)?;
    let (b_di, off_di) = extract_buffer(&grad_inits)?;

    let mut ctx = EncodingContextRcBackward {
        metal_ctx,
        b_combined: &b_comb,
        off_combined: off_comb,
        b_inits: &b_inits,
        off_inits: off_inits,
        b_hq: &b_hq,
        off_hq: off_hq,
        b_hk: &b_hk,
        off_hk: off_hk,
        b_hv: &b_hv,
        off_hv: off_hv,
        b_gq: &b_gq,
        off_gq: off_gq,
        b_gk: &b_gk,
        off_gk: off_gk,
        b_gv: &b_gv,
        off_gv: off_gv,
        b_d_comb: &b_dc,
        off_d_comb: off_dc,
        b_di: &b_di,
        off_di: off_di,
        sc_combined: [sc_b, sc_t, sc_d],
        sgc_combined: [sgc_b, sgc_t, sgc_d],
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
            encode_rc_backward_callback,
            &mut ctx as *mut _ as *mut c_void,
        );
    }
    Ok(())
}

#[pyfunction(name = "init_ltv_fused_concat_register_cast_scan_metal_context_for_default_device")]
pub fn init_ltv_fused_concat_register_cast_scan_metal_context_for_default_device() -> PyResult<()> {
    let device = Device::system_default()
        .ok_or_else(|| PyErr::new::<PyRuntimeError, _>("No Metal device"))?;

    let lib_fwd = device
        .new_library_with_source(RC_FORWARD_KERNEL_SOURCE, &CompileOptions::new())
        .map_err(|e| PyErr::new::<PyRuntimeError, _>(e.to_string()))?;
    let pipeline_forward = device
        .new_compute_pipeline_state_with_function(
            &lib_fwd
                .get_function("ltv_fused_concat_register_cast_scan_forward", None)
                .unwrap(),
        )
        .map_err(|e| PyErr::new::<PyRuntimeError, _>(e.to_string()))?;

    let lib_back_direct = device
        .new_library_with_source(
            &format!("{}\n{}", BACKWARD_COMMON_H, RC_BACKWARD_DIRECT_SRC),
            &CompileOptions::new(),
        )
        .map_err(|e| PyErr::new::<PyRuntimeError, _>(e.to_string()))?;
    let lib_back_atomic = device
        .new_library_with_source(
            &format!("{}\n{}", BACKWARD_COMMON_H, RC_BACKWARD_ATOMIC_SRC),
            &CompileOptions::new(),
        )
        .map_err(|e| PyErr::new::<PyRuntimeError, _>(e.to_string()))?;

    let pipeline_backward_direct = device
        .new_compute_pipeline_state_with_function(
            &lib_back_direct
                .get_function("ltv_fused_concat_register_cast_scan_backward_direct", None)
                .unwrap(),
        )
        .map_err(|e| PyErr::new::<PyRuntimeError, _>(e.to_string()))?;
    let pipeline_backward_atomic = device
        .new_compute_pipeline_state_with_function(
            &lib_back_atomic
                .get_function("ltv_fused_concat_register_cast_scan_backward_atomic", None)
                .unwrap(),
        )
        .map_err(|e| PyErr::new::<PyRuntimeError, _>(e.to_string()))?;

    set_metal_context_rc(MetalContextRegisterCast {
        device,
        pipeline_forward,
        pipeline_backward_atomic,
        pipeline_backward_direct,
    })
    .map_err(|_| PyErr::new::<PyRuntimeError, _>("Already init (rc)"))?;

    Ok(())
}

#[pyfunction(name = "release_ltv_fused_concat_register_cast_scan_metal_context")]
pub fn release_ltv_fused_concat_register_cast_scan_metal_context() {}
