#![cfg(all(feature = "metal", feature = "python"))]

use crate::metal_utils::{dispatch_on_mps, extract_buffer};
use metal::foreign_types::ForeignTypeRef;
use metal::*;
use pyo3::exceptions::PyRuntimeError;
use pyo3::prelude::*;
use std::ffi::c_void;
use std::sync::RwLock;

// ------------------------------------------------------------
// DTYPE
// ------------------------------------------------------------

#[derive(Clone, Copy)]
enum DType {
    BF16,
    F32,
    F64,
}

impl DType {
    fn from_tensor(t: &Py<PyAny>) -> PyResult<Self> {
        Python::attach(|py| {
            let dtype_obj = t.getattr(py, "dtype")?;
            let dtype: String = dtype_obj.call_method0(py, "__str__")?.extract(py)?;

            if dtype.contains("bfloat16") {
                Ok(DType::BF16)
            } else if dtype.contains("float32") {
                Ok(DType::F32)
            } else if dtype.contains("float64") {
                Ok(DType::F64)
            } else {
                Err(PyErr::new::<PyRuntimeError, _>("Unsupported dtype"))
            }
        })
    }
}

// ------------------------------------------------------------
// PIPELINE BUNDLE
// ------------------------------------------------------------

#[derive(Debug)]
struct PipelineBundle {
    pipeline: ComputePipelineState,
    compute_t_size: u64,
}

// ------------------------------------------------------------
// CONTEXT
// ------------------------------------------------------------

#[derive(Debug)]
struct MetalContextFusedScan {
    fwd_bf16: PipelineBundle,
    fwd_f32: PipelineBundle,
    bwd_bf16: PipelineBundle,
    bwd_f32: PipelineBundle,

    thread_execution_width: u32,
    max_threads: u32,
    max_tg_mem: u64,
    banks: u32,
}

static CONTEXT: RwLock<Option<MetalContextFusedScan>> = RwLock::new(None);

// ------------------------------------------------------------
// BUILD HELPERS
// ------------------------------------------------------------

fn build_bundle(
    device: &Device,
    library: &Library,
    name: &str,
    compute_t_size: u64,
) -> PipelineBundle {
    let func = library.get_function(name, None).unwrap();

    let pipeline = device
        .new_compute_pipeline_state_with_function(&func)
        .unwrap();

    PipelineBundle {
        pipeline,
        compute_t_size,
    }
}

// ------------------------------------------------------------
// MACRO: DEFINE ALL PIPELINES
// ------------------------------------------------------------

macro_rules! build_all {
    ($device:expr, $library:expr) => {{
        MetalContextFusedScan {
            fwd_bf16: build_bundle(
                $device,
                $library,
                "ltv_fused_concat_register_cast_blelloch_scan_forward_bf16_f32_bf16",
                4,
            ),
            fwd_f32: build_bundle(
                $device,
                $library,
                "ltv_fused_concat_register_cast_blelloch_scan_forward_f32_f32_f32",
                4,
            ),
            bwd_bf16: build_bundle(
                $device,
                $library,
                "ltv_fused_concat_register_cast_blelloch_scan_backward_bf16_f32_bf16",
                4,
            ),
            bwd_f32: build_bundle(
                $device,
                $library,
                "ltv_fused_concat_register_cast_blelloch_scan_backward_f32_f32_f32",
                4,
            ),

            thread_execution_width: 0, // filled below
            max_threads: 0,
            max_tg_mem: $device.max_threadgroup_memory_length(),
            banks: 0,
        }
    }};
}

fn metal_gcd(mut a: u32, mut b: u32) -> u32 {
    while b != 0 {
        let temp = b;
        b = a % b;
        a = temp;
    }
    a
}

fn get_padding_denominators(banks: u32, c: u32, max_elements: u64) -> Vec<u32> {
    if c <= 1 {
        return Vec::new();
    }
    
    let mut denominators = Vec::new();
    let mut m = 1u32;
    loop {
        let current_stride = c.pow(m);
        let gcd_val = metal_gcd(banks, current_stride);
        if gcd_val > 1 {
            let lcm_val = (banks * current_stride) / gcd_val;
            if lcm_val as u64 > max_elements {
                break;
            }
            denominators.push(lcm_val);
        } else {
            if current_stride as u64 > max_elements {
                break;
            }
        }
        m += 1;
    }
    denominators
}

// ------------------------------------------------------------
// INIT
// ------------------------------------------------------------

pub fn init_ltv_metal_fused_concat_register_cast_blelloch_scan_for_default_device(
    thread_execution_width: u32,
    max_threads: u32,
    max_tg_mem: u64,
    banks: u32,
) -> PyResult<()> {
    let device = Device::system_default()
        .ok_or_else(|| PyErr::new::<PyRuntimeError, _>("No Metal device"))?;

    let header = include_str!("ltv_blelloch_affine_scan.mslh");
    let mut forward =
        include_str!("ltv_fused_concat_register_cast_blelloch_scan_forward.msl").to_string();
    let mut backward =
        include_str!("ltv_fused_concat_register_cast_blelloch_scan_backward.msl").to_string();

    // Remove the #include directives that reference the header
    forward = forward.replace("#include \"ltv_blelloch_affine_scan.mslh\"", "");
    backward = backward.replace("#include \"ltv_blelloch_affine_scan.mslh\"", "");

    let c = if thread_execution_width == 0 {
        32
    } else {
        thread_execution_width
    };
    let actual_max_tg_mem = if max_tg_mem == 0 {
        device.max_threadgroup_memory_length()
    } else {
        max_tg_mem
    };

    let mut bank_pad_macro = String::from("#define BANK_IDX(idx) (idx)\n");
    let num_banks = if banks == 0 { 32 } else { banks };

    // T_compute inside the Metal kernels is `float` for both f32 and bf16
    // variants (see threadgroup float* declarations in .msl).
    // 4 bytes per float.
    let max_elements = actual_max_tg_mem / 4;
    let denoms = get_padding_denominators(num_banks, c, max_elements);

    if !denoms.is_empty() {
        bank_pad_macro = String::from("#define BANK_IDX(idx) ((idx)");
        for d in denoms {
            bank_pad_macro.push_str(&format!(" + (idx) / {}", d));
        }
        bank_pad_macro.push_str(")\n");
    }

    let source = format!("{}\n{}\n{}\n{}", bank_pad_macro, header, forward, backward);

    let library = device
        .new_library_with_source(&source, &CompileOptions::new())
        .unwrap();

    let mut ctx = build_all!(&device, &library);

    let default_thread_execution_width = ctx.fwd_f32.pipeline.thread_execution_width() as u32;
    ctx.thread_execution_width = if thread_execution_width == 0 {
        default_thread_execution_width
    } else {
        thread_execution_width
    };

    let default_max_threads = ctx.fwd_f32.pipeline.max_total_threads_per_threadgroup() as u32;
    ctx.max_threads = if max_threads == 0 {
        default_max_threads
    } else {
        max_threads
    };

    if max_tg_mem != 0 {
        ctx.max_tg_mem = max_tg_mem;
    }

    ctx.banks = if banks == 0 { 32 } else { banks };

    *CONTEXT.write().expect("Metal context lock poisoned") = Some(ctx);

    Ok(())
}

pub fn release_ltv_metal_fused_concat_register_cast_blelloch_scan_for_default_device() {
    *CONTEXT.write().expect("Metal context lock poisoned") = None;
}

// ------------------------------------------------------------
// CHUNK SIZE
// ------------------------------------------------------------

fn compute_chunk_size(ctx: &MetalContextFusedScan, t_seq: u32, sizeof_t: u64) -> u32 {
    let c = ctx.thread_execution_width;
    let mut chunk = 1u32;

    if c != 1 {
        let max_elements = ctx.max_tg_mem / sizeof_t;
        let denoms = get_padding_denominators(ctx.banks, c, max_elements);

        while chunk < t_seq && chunk * c <= ctx.max_threads {
            let next = chunk * c;
            let mut padded_next = next;

            for d in &denoms {
                padded_next += next / d;
            }

            let mem = padded_next as u64 * sizeof_t * 2;

            if mem > ctx.max_tg_mem {
                break;
            }

            chunk = next;
        }
    }

    chunk
}

fn get_padded_elements(chunk_size: u32, ctx: &MetalContextFusedScan, sizeof_t: u64) -> u32 {
    if chunk_size == 0 {
        return 0;
    }

    let max_elements = ctx.max_tg_mem / sizeof_t;
    let denoms = get_padding_denominators(ctx.banks, ctx.thread_execution_width, max_elements);

    let max_idx = chunk_size - 1;
    let mut padded_max = max_idx;
    for d in &denoms {
        padded_max += max_idx / d;
    }

    padded_max + 1
}

// ------------------------------------------------------------
// PIPELINE SELECTION
// ------------------------------------------------------------

fn select_fwd(ctx: &MetalContextFusedScan, dtype: DType) -> &PipelineBundle {
    match dtype {
        DType::BF16 => &ctx.fwd_bf16,
        DType::F32 => &ctx.fwd_f32,
        DType::F64 => panic!("F64 not supported on Metal"),
    }
}

fn select_bwd(ctx: &MetalContextFusedScan, dtype: DType) -> &PipelineBundle {
    match dtype {
        DType::BF16 => &ctx.bwd_bf16,
        DType::F32 => &ctx.bwd_f32,
        DType::F64 => panic!("F64 not supported on Metal"),
    }
}

// ------------------------------------------------------------
// FORWARD CALLBACK DATA
// ------------------------------------------------------------

struct ForwardCallbackData {
    bundle_pipeline: *const ComputePipelineState,
    b_combined: *const BufferRef,
    off_combined: u64,
    b_inits: *const BufferRef,
    off_inits: u64,
    b_out_q: *const BufferRef,
    off_out_q: u64,
    b_out_k: *const BufferRef,
    off_out_k: u64,
    b_out_v: *const BufferRef,
    off_out_v: u64,
    sc: (u32, u32, u32),
    b: u32,
    t_seq: u32,
    nh: u32,
    nkvh: u32,
    da: u32,
    r: u32,
    chunk_size: u32,
    c: u32,
    use_sigmoid: bool,
    tg_mem: u64,
}

extern "C" fn forward_callback(cmd_buf_ptr: *mut c_void, context_ptr: *mut c_void) {
    unsafe {
        let data = &*(context_ptr as *const ForwardCallbackData);
        let cmd_buf = CommandBufferRef::from_ptr(cmd_buf_ptr as *mut _);
        let enc = cmd_buf.new_compute_command_encoder();

        enc.set_compute_pipeline_state(&*data.bundle_pipeline);

        enc.set_buffer(0, Some(&*data.b_combined), data.off_combined);
        enc.set_buffer(1, Some(&*data.b_inits), data.off_inits);
        enc.set_buffer(2, Some(&*data.b_out_q), data.off_out_q);
        enc.set_buffer(3, Some(&*data.b_out_k), data.off_out_k);
        enc.set_buffer(4, Some(&*data.b_out_v), data.off_out_v);

        enc.set_bytes(5, 12, &data.sc as *const _ as _);
        enc.set_bytes(6, 4, &data.b as *const _ as _);
        enc.set_bytes(7, 4, &data.t_seq as *const _ as _);
        enc.set_bytes(8, 4, &data.nh as *const _ as _);
        enc.set_bytes(9, 4, &data.nkvh as *const _ as _);
        enc.set_bytes(10, 4, &data.da as *const _ as _);
        enc.set_bytes(11, 4, &data.r as *const _ as _);
        enc.set_bytes(12, 4, &data.chunk_size as *const _ as _);
        enc.set_bytes(13, 4, &data.c as *const _ as _);
        enc.set_bytes(14, 1, &data.use_sigmoid as *const _ as _);

        enc.set_threadgroup_memory_length(0, data.tg_mem);
        enc.set_threadgroup_memory_length(1, data.tg_mem);

        enc.dispatch_thread_groups(
            MTLSize {
                width: data.b as u64,
                height: (data.nh + 2 * data.nkvh) as u64,
                depth: (data.da * data.r) as u64,
            },
            MTLSize {
                width: data.chunk_size as u64,
                height: 1,
                depth: 1,
            },
        );

        enc.end_encoding();
    }
}

// ------------------------------------------------------------
// FORWARD
// ------------------------------------------------------------

#[allow(clippy::too_many_arguments)]
pub fn ltv_metal_fused_concat_register_cast_blelloch_scan_forward(
    combined: Py<PyAny>,
    inits: Py<PyAny>,
    out_q: Py<PyAny>,
    out_k: Py<PyAny>,
    out_v: Py<PyAny>,
    sc: (u32, u32, u32),
    b: u32,
    t_seq: u32,
    nh: u32,
    nkvh: u32,
    da: u32,
    r: u32,
    use_sigmoid: bool,
) -> PyResult<()> {
    let ctx_guard = CONTEXT.read().expect("Metal context lock poisoned");
    let ctx = ctx_guard.as_ref().expect("Metal context not initialized");

    let dtype = DType::from_tensor(&combined)?;
    let bundle = select_fwd(ctx, dtype);

    let c = ctx.thread_execution_width;
    let chunk_size = compute_chunk_size(ctx, t_seq, bundle.compute_t_size);

    let (b_combined, off_combined) = extract_buffer(&combined)?;
    let (b_inits, off_inits) = extract_buffer(&inits)?;
    let (b_out_q, off_out_q) = extract_buffer(&out_q)?;
    let (b_out_k, off_out_k) = extract_buffer(&out_k)?;
    let (b_out_v, off_out_v) = extract_buffer(&out_v)?;

    let tg_mem =
        get_padded_elements(chunk_size, ctx, bundle.compute_t_size) as u64 * bundle.compute_t_size;

    let data = ForwardCallbackData {
        bundle_pipeline: &bundle.pipeline as *const _,
        b_combined: &*b_combined as *const _,
        off_combined,
        b_inits: &*b_inits as *const _,
        off_inits,
        b_out_q: &*b_out_q as *const _,
        off_out_q,
        b_out_k: &*b_out_k as *const _,
        off_out_k,
        b_out_v: &*b_out_v as *const _,
        off_out_v,
        sc,
        b,
        t_seq,
        nh,
        nkvh,
        da,
        r,
        chunk_size,
        c,
        use_sigmoid,
        tg_mem,
    };

    unsafe {
        dispatch_on_mps(forward_callback, &data as *const _ as *mut _);
    }

    Ok(())
}

// ------------------------------------------------------------
// BACKWARD CALLBACK DATA
// ------------------------------------------------------------

struct BackwardCallbackData {
    bundle_pipeline: *const ComputePipelineState,
    b_combined: *const BufferRef,
    off_combined: u64,
    b_inits: *const BufferRef,
    off_inits: u64,
    b_hq: *const BufferRef,
    off_hq: u64,
    b_hk: *const BufferRef,
    off_hk: u64,
    b_hv: *const BufferRef,
    off_hv: u64,
    b_grad_hq: *const BufferRef,
    off_grad_hq: u64,
    b_grad_hk: *const BufferRef,
    off_grad_hk: u64,
    b_grad_hv: *const BufferRef,
    off_grad_hv: u64,
    b_grad_combined: *const BufferRef,
    off_grad_combined: u64,
    b_grad_inits: *const BufferRef,
    off_grad_inits: u64,
    b_grad_logits: *const BufferRef,
    off_grad_logits: u64,
    sc_in: (u32, u32, u32),
    sc_out: (u32, u32, u32),
    b: u32,
    t_seq: u32,
    nh: u32,
    nkvh: u32,
    da: u32,
    r: u32,
    chunk_size: u32,
    c: u32,
    use_sigmoid: bool,
    tg_mem: u64,
}

extern "C" fn backward_callback(cmd_buf_ptr: *mut c_void, context_ptr: *mut c_void) {
    unsafe {
        let data = &*(context_ptr as *const BackwardCallbackData);
        let cmd_buf = CommandBufferRef::from_ptr(cmd_buf_ptr as *mut _);
        let enc = cmd_buf.new_compute_command_encoder();

        enc.set_compute_pipeline_state(&*data.bundle_pipeline);

        enc.set_buffer(0, Some(&*data.b_combined), data.off_combined);
        enc.set_buffer(1, Some(&*data.b_inits), data.off_inits);
        enc.set_buffer(2, Some(&*data.b_hq), data.off_hq);
        enc.set_buffer(3, Some(&*data.b_hk), data.off_hk);
        enc.set_buffer(4, Some(&*data.b_hv), data.off_hv);
        enc.set_buffer(5, Some(&*data.b_grad_hq), data.off_grad_hq);
        enc.set_buffer(6, Some(&*data.b_grad_hk), data.off_grad_hk);
        enc.set_buffer(7, Some(&*data.b_grad_hv), data.off_grad_hv);
        enc.set_buffer(8, Some(&*data.b_grad_combined), data.off_grad_combined);
        enc.set_buffer(9, Some(&*data.b_grad_inits), data.off_grad_inits);
        enc.set_buffer(10, Some(&*data.b_grad_logits), data.off_grad_logits);

        enc.set_bytes(11, 12, &data.sc_in as *const _ as _);
        enc.set_bytes(12, 12, &data.sc_out as *const _ as _);
        enc.set_bytes(13, 4, &data.b as *const _ as _);
        enc.set_bytes(14, 4, &data.t_seq as *const _ as _);
        enc.set_bytes(15, 4, &data.nh as *const _ as _);
        enc.set_bytes(16, 4, &data.nkvh as *const _ as _);
        enc.set_bytes(17, 4, &data.da as *const _ as _);
        enc.set_bytes(18, 4, &data.r as *const _ as _);
        enc.set_bytes(19, 4, &data.chunk_size as *const _ as _);
        enc.set_bytes(20, 4, &data.c as *const _ as _);
        enc.set_bytes(21, 1, &data.use_sigmoid as *const _ as _);

        enc.set_threadgroup_memory_length(0, data.tg_mem);
        enc.set_threadgroup_memory_length(1, data.tg_mem);
        enc.set_threadgroup_memory_length(2, data.tg_mem);

        enc.dispatch_thread_groups(
            MTLSize {
                width: data.b as u64,
                height: (data.nh + 2 * data.nkvh) as u64,
                depth: (data.da * data.r) as u64,
            },
            MTLSize {
                width: data.chunk_size as u64,
                height: 1,
                depth: 1,
            },
        );

        enc.end_encoding();
    }
}

// ------------------------------------------------------------
// BACKWARD
// ------------------------------------------------------------

#[allow(clippy::too_many_arguments)]
pub fn ltv_metal_fused_concat_register_cast_blelloch_scan_backward(
    combined: Py<PyAny>,
    inits: Py<PyAny>,
    hq: Py<PyAny>,
    hk: Py<PyAny>,
    hv: Py<PyAny>,
    grad_hq: Py<PyAny>,
    grad_hk: Py<PyAny>,
    grad_hv: Py<PyAny>,
    grad_combined: Py<PyAny>,
    grad_inits: Py<PyAny>,
    grad_logits: Py<PyAny>,
    sc_in: (u32, u32, u32),
    sc_out: (u32, u32, u32),
    b: u32,
    t_seq: u32,
    nh: u32,
    nkvh: u32,
    da: u32,
    r: u32,
    use_sigmoid: bool,
) -> PyResult<()> {
    let ctx_guard = CONTEXT.read().expect("Metal context lock poisoned");
    let ctx = ctx_guard.as_ref().expect("Metal context not initialized");

    let dtype = DType::from_tensor(&combined)?;
    let bundle = select_bwd(ctx, dtype);

    let c = ctx.thread_execution_width;
    let chunk_size = compute_chunk_size(ctx, t_seq, bundle.compute_t_size);
    let use_sigmoid = use_sigmoid; // directly use bool

    let (b_combined, off_combined) = extract_buffer(&combined)?;
    let (b_inits, off_inits) = extract_buffer(&inits)?;
    let (b_hq, off_hq) = extract_buffer(&hq)?;
    let (b_hk, off_hk) = extract_buffer(&hk)?;
    let (b_hv, off_hv) = extract_buffer(&hv)?;
    let (b_grad_hq, off_grad_hq) = extract_buffer(&grad_hq)?;
    let (b_grad_hk, off_grad_hk) = extract_buffer(&grad_hk)?;
    let (b_grad_hv, off_grad_hv) = extract_buffer(&grad_hv)?;
    let (b_grad_combined, off_grad_combined) = extract_buffer(&grad_combined)?;
    let (b_grad_inits, off_grad_inits) = extract_buffer(&grad_inits)?;
    let (b_grad_logits, off_grad_logits) = extract_buffer(&grad_logits)?;

    let tg_mem =
        get_padded_elements(chunk_size, ctx, bundle.compute_t_size) as u64 * bundle.compute_t_size;

    let data = BackwardCallbackData {
        bundle_pipeline: &bundle.pipeline as *const _,
        b_combined: &*b_combined as *const _,
        off_combined,
        b_inits: &*b_inits as *const _,
        off_inits,
        b_hq: &*b_hq as *const _,
        off_hq,
        b_hk: &*b_hk as *const _,
        off_hk,
        b_hv: &*b_hv as *const _,
        off_hv,
        b_grad_hq: &*b_grad_hq as *const _,
        off_grad_hq,
        b_grad_hk: &*b_grad_hk as *const _,
        off_grad_hk,
        b_grad_hv: &*b_grad_hv as *const _,
        off_grad_hv,
        b_grad_combined: &*b_grad_combined as *const _,
        off_grad_combined,
        b_grad_inits: &*b_grad_inits as *const _,
        off_grad_inits,
        b_grad_logits: &*b_grad_logits as *const _,
        off_grad_logits,
        sc_in,
        sc_out,
        b,
        t_seq,
        nh,
        nkvh,
        da,
        r,
        chunk_size,
        c,
        use_sigmoid,
        tg_mem,
    };

    unsafe {
        dispatch_on_mps(backward_callback, &data as *const _ as *mut _);
    }

    Ok(())
}
