#![feature(adt_const_params)]
#![feature(generic_const_exprs)]
#![allow(incomplete_features)]

// This file implements a numerically-stable, chunked, recursive log-space
// EWMA forward + backward and exposes Python-callable functions via PyO3.


#[cfg(all(feature = "metal", feature = "python"))]
use crate::ltv_utils::create_geometry_from_schedule;


#[cfg(feature = "python")]
use pyo3::exceptions::PyRuntimeError;
#[cfg(feature = "python")]
use pyo3::prelude::*;
#[cfg(feature = "python")]
use pyo3::wrap_pyfunction;

#[cfg(feature = "python")]
use numpy::ndarray::ArrayD;
#[cfg(feature = "python")]
use numpy::{IntoPyArray, PyArrayDyn, PyArrayMethods, PyUntypedArrayMethods};

use std::fmt::Debug;

#[cfg(all(feature = "metal", feature = "python"))]
pub mod ltv_utils;

#[cfg(all(feature = "metal", feature = "python"))]
pub mod ltv_blelloch_utils;
#[cfg(all(feature = "metal", feature = "python"))]
pub mod ltv_fused_concat_register_cast_blelloch_scan;
#[cfg(all(feature = "metal", feature = "python"))]
pub mod ltv_fused_concat_scan_metal;
#[cfg(all(feature = "metal", feature = "python"))]
pub mod ltv_fused_linear_scan_metal;
#[cfg(all(feature = "metal", feature = "python"))]
pub mod ltv_fused_scan_metal;
#[cfg(all(feature = "metal", feature = "python"))]
pub mod ltv_noop_metal;
#[cfg(all(feature = "metal", feature = "python"))]
pub mod ltv_plane_metal;
#[cfg(all(feature = "metal", feature = "python"))]
pub mod ltv_scan_metal;
#[cfg(all(feature = "metal", feature = "python"))]
pub mod ltv_shared_metal;

#[cfg(all(feature = "metal", feature = "python"))]
pub mod metal_utils;


// 1. Define the Single Public Trait
pub trait SafeDbg {
    fn sdbg_resolve(&self) -> String;
}

// 2. Define the Sealed Helper Trait
mod safe_dbg_private {
    pub trait SafeDbgHelper {
        fn sdbg_format(&self) -> String;
    }
}
use safe_dbg_private::SafeDbgHelper;



// 4. Implementation 2: The Generic (Low Priority) Fallback
impl<'a, T: Debug> SafeDbgHelper for &'a T {
    fn sdbg_format(&self) -> String {
        format!("<{}> @ {:p}", std::any::type_name::<T>(), self)
    }
}

// 5. Final implementation of the public trait for everything (blanket implementation)
impl<T: SafeDbgHelper> SafeDbg for T {
    fn sdbg_resolve(&self) -> String {
        self.sdbg_format()
    }
}

#[macro_export]
macro_rules! maybe_dbg( ($($tt:tt)*) => {} );



// --- LTV Implementations ---

/// CPU LTV Scan
///
/// Inputs must be (T, B, D) flattened or capable of being viewed as T x (B*D).
/// In Python, use .transpose(0, 1).contiguous() on (B, T, D) tensors.
#[cfg(feature = "python")]
#[pyfunction]
fn ltv_scan_cpu<'py>(
    py: Python<'py>,
    h_init: Bound<'py, PyArrayDyn<f32>>,
    alpha: Bound<'py, PyArrayDyn<f32>>,
    x: Bound<'py, PyArrayDyn<f32>>,
) -> PyResult<Bound<'py, PyArrayDyn<f32>>> {
    let h_init = h_init.readonly();
    let alpha = alpha.readonly();
    let x = x.readonly();

    // Verify shapes
    // x: (T, B, D_x) -> (T, BD_x)
    // alpha: (T, B, D_a) -> (T, BD_a)
    // h_init: (B, D_x) -> (BD_x)

    // Check dimensionality
    if x.ndim() < 2 {
        return Err(PyRuntimeError::new_err(
            "Input x must be at least 2D (T, B...)",
        ));
    }
    let t_dim = x.shape()[0];
    let bd_dim_x: usize = x.shape()[1..].iter().product();

    // Alpha shape check
    let bd_dim_a: usize = alpha.shape()[1..].iter().product();
    let r = if bd_dim_a == 0 {
        if bd_dim_x != 0 {
            return Err(PyRuntimeError::new_err(format!(
                "x size {} > 0 but alpha size 0",
                bd_dim_x
            )));
        }
        1
    } else {
        if bd_dim_x % bd_dim_a != 0 {
            return Err(PyRuntimeError::new_err(format!(
                "x size {} must be multiple of alpha size {}",
                bd_dim_x, bd_dim_a
            )));
        }
        bd_dim_x / bd_dim_a
    };

    // Contiguity Checks
    let h_init_slice = h_init
        .as_slice()
        .map_err(|_| PyRuntimeError::new_err("h_init must be C-contiguous"))?;
    let alpha_slice = alpha
        .as_slice()
        .map_err(|_| PyRuntimeError::new_err("alpha must be C-contiguous"))?;
    let x_slice = x
        .as_slice()
        .map_err(|_| PyRuntimeError::new_err("x must be C-contiguous"))?;

    if alpha_slice.len() * r != x_slice.len() {
        return Err(PyRuntimeError::new_err("alpha * r must match x size"));
    }
    if h_init_slice.len() != bd_dim_x {
        return Err(PyRuntimeError::new_err(format!(
            "h_init size {} mismatch with BD_x {}",
            h_init_slice.len(),
            bd_dim_x
        )));
    }

    // Run Logic (copied from run_seq_rust)
    let parallel_dim_a = bd_dim_a;
    let t = t_dim;
    let mut out = vec![0.0f32; t * bd_dim_x];

    for step in 0..t {
        let curr_off_x = step * bd_dim_x;
        let curr_off_a = step * parallel_dim_a;

        let alpha_row = &alpha_slice[curr_off_a..curr_off_a + parallel_dim_a];
        let x_row = &x_slice[curr_off_x..curr_off_x + bd_dim_x];

        // Handle borrowing
        let (prev_part, curr_part) = out.split_at_mut(curr_off_x);
        let out_row = &mut curr_part[..bd_dim_x];

        let prev_h_row = if step == 0 {
            h_init_slice
        } else {
            // prev_part end is at curr_off_x.
            // We want the LAST bd_dim_x elements of prev_part.
            &prev_part[prev_part.len() - bd_dim_x..]
        };

        for i in 0..bd_dim_x {
            // alpha maps to i / r
            out_row[i] = alpha_row[i / r] * prev_h_row[i] + x_row[i];
        }
    }

    let out_shape = x.shape();
    let out_arr = ArrayD::from_shape_vec(out_shape, out)
        .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;

    Ok(out_arr.into_pyarray(py))
}

#[cfg(feature = "python")]
#[pymodule]
fn rust_ewma(m: &Bound<'_, PyModule>) -> PyResult<()> {
    pyo3_log::init();
    m.add_function(wrap_pyfunction!(ltv_scan_cpu, m)?)?;

    // Metal implementations (Always registered to prevent AttributeError)
    m.add_function(wrap_pyfunction!(ltv_noop_metal_py, m)?)?;
    m.add_function(wrap_pyfunction!(
        init_ltv_noop_metal_context_for_default_device_py,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(release_ltv_noop_metal_context_py, m)?)?;
    m.add_function(wrap_pyfunction!(ltv_scan_metal_py, m)?)?;
    m.add_function(wrap_pyfunction!(
        init_ltv_scan_metal_context_for_default_device_py,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(release_ltv_scan_metal_context_py, m)?)?;

    m.add_function(wrap_pyfunction!(ltv_fused_scan_metal_forward_py, m)?)?;
    m.add_function(wrap_pyfunction!(ltv_fused_scan_metal_backward_py, m)?)?;
    m.add_function(wrap_pyfunction!(
        init_ltv_fused_scan_metal_context_for_default_device_py,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        release_ltv_fused_scan_metal_context_py,
        m
    )?)?;

    m.add_function(wrap_pyfunction!(ltv_fused_concat_scan_metal_forward_py, m)?)?;
    m.add_function(wrap_pyfunction!(
        ltv_fused_concat_scan_metal_backward_py,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        init_ltv_fused_concat_scan_metal_context_for_default_device_py,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        release_ltv_fused_concat_scan_metal_context_py,
        m
    )?)?;

    m.add_function(wrap_pyfunction!(
        ltv_fused_concat_register_cast_scan_metal_forward_py,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        ltv_fused_concat_register_cast_scan_metal_backward_py,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        init_ltv_fused_concat_register_cast_scan_metal_context_for_default_device_py,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        release_ltv_fused_concat_register_cast_scan_metal_context_py,
        m
    )?)?;

    m.add_function(wrap_pyfunction!(
        ltv_metal_fused_concat_register_cast_blelloch_scan_forward_py,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        ltv_metal_fused_concat_register_cast_blelloch_scan_backward_py,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        init_ltv_metal_fused_concat_register_cast_blelloch_scan_context_for_default_device_py,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        release_ltv_metal_fused_concat_register_cast_blelloch_scan_context_py,
        m
    )?)?;

    m.add_function(wrap_pyfunction!(ltv_fused_linear_scan_metal_forward_py, m)?)?;
    m.add_function(wrap_pyfunction!(
        ltv_fused_linear_scan_metal_backward_py,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        init_ltv_fused_linear_scan_metal_context_for_default_device_py,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        release_ltv_fused_linear_scan_metal_context_py,
        m
    )?)?;

    m.add_function(wrap_pyfunction!(ltv_plane_metal_py, m)?)?;
    m.add_function(wrap_pyfunction!(
        init_ltv_plane_metal_context_for_default_device_py,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(release_ltv_plane_metal_context_py, m)?)?;
    m.add_function(wrap_pyfunction!(ltv_shared_metal_py, m)?)?;
    m.add_function(wrap_pyfunction!(
        init_ltv_shared_metal_context_for_default_device_py,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(release_ltv_shared_metal_context_py, m)?)?;
    Ok(())
}

// --- Metal Feature-Gated Wrappers (Always Exported for Python) ---

#[cfg(feature = "python")]
fn metal_feature_error() -> PyErr {
    PyRuntimeError::new_err(
        "Metal feature not enabled in this build. Please build with --features metal and ensure PyTorch headers are available.",
    )
}


#[cfg(feature = "python")]
#[pyfunction(name = "ltv_noop_metal")]
pub fn ltv_noop_metal_py(
    py: Python<'_>,
    alpha: Py<PyAny>,
    x: Py<PyAny>,
    out_a: Py<PyAny>,
    out_x: Py<PyAny>,
    t_seq: u32,
    parallel_a: u32,
    r: u32,
) -> PyResult<()> {
    #[cfg(feature = "metal")]
    {
        crate::ltv_noop_metal::ltv_noop_metal(py, alpha, x, out_a, out_x, t_seq, parallel_a, r)
    }
    #[cfg(not(feature = "metal"))]
    {
        Err(metal_feature_error())
    }
}

#[cfg(feature = "python")]
#[pyfunction(name = "init_ltv_noop_metal_context_for_default_device")]
pub fn init_ltv_noop_metal_context_for_default_device_py() -> PyResult<()> {
    #[cfg(feature = "metal")]
    {
        crate::ltv_noop_metal::init_ltv_noop_metal_context_for_default_device()
    }
    #[cfg(not(feature = "metal"))]
    {
        Err(metal_feature_error())
    }
}

#[cfg(feature = "python")]
#[pyfunction(name = "release_noop_metal_context")]
pub fn release_ltv_noop_metal_context_py() {
    #[cfg(feature = "metal")]
    {
        crate::ltv_noop_metal::release_noop_metal_context();
    }
}

#[cfg(feature = "python")]
#[pyfunction(name = "ltv_scan_metal")]
pub fn ltv_scan_metal_py(
    py: Python<'_>,
    alpha: Py<PyAny>,
    x: Py<PyAny>,
    out_a: Py<PyAny>,
    out_x: Py<PyAny>,
    t_seq: u32,
    parallel_a: u32,
    r: u32,
) -> PyResult<()> {
    #[cfg(feature = "metal")]
    {
        crate::ltv_scan_metal::ltv_scan_metal(py, alpha, x, out_a, out_x, t_seq, parallel_a, r)
    }
    #[cfg(not(feature = "metal"))]
    {
        Err(metal_feature_error())
    }
}

#[cfg(feature = "python")]
#[pyfunction(name = "init_ltv_scan_metal_context_for_default_device")]
pub fn init_ltv_scan_metal_context_for_default_device_py() -> PyResult<()> {
    #[cfg(feature = "metal")]
    {
        crate::ltv_scan_metal::init_ltv_scan_metal_context_for_default_device()
    }
    #[cfg(not(feature = "metal"))]
    {
        Err(metal_feature_error())
    }
}

#[cfg(feature = "python")]
#[pyfunction(name = "release_ltv_scan_metal_context")]
pub fn release_ltv_scan_metal_context_py() {
    #[cfg(feature = "metal")]
    {
        crate::ltv_scan_metal::release_ltv_scan_metal_context();
    }
}

#[cfg(feature = "python")]
#[pyfunction(name = "ltv_fused_scan_metal_forward")]
pub fn ltv_fused_scan_metal_forward_py(
    py: Python<'_>,
    q: Py<PyAny>,
    k: Py<PyAny>,
    v: Py<PyAny>,
    logits: Py<PyAny>,
    inits: Py<PyAny>,
    out_q: Py<PyAny>,
    out_k: Py<PyAny>,
    out_v: Py<PyAny>,
    B: u32,
    T: u32,
    NH: u32,
    NKVH: u32,
    D_a: u32,
    r: u32,
) -> PyResult<()> {
    #[cfg(feature = "metal")]
    {
        // 1. Get raw pointers/buffers from the PyTorch Tensors
        // 2. Load the 'ltv_fwd_qkv' pipeline state
        // 3. Bind buffers 0 through 14
        // 4. Threading: grid=(B, NH + 2*NKVH, D), block=(1, 1, 1) or optimized
        crate::ltv_fused_scan_metal::ltv_fused_scan_metal_forward(
            py, q, k, v, logits, inits, out_q, out_k, out_v, B, T, NH, NKVH, D_a, r,
        )
    }
    #[cfg(not(feature = "metal"))]
    {
        Err(metal_feature_error())
    }
}

#[cfg(feature = "python")]
#[pyfunction(name = "ltv_fused_scan_metal_backward")]
pub fn ltv_fused_scan_metal_backward_py(
    py: Python<'_>,
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
    B: u32,
    T: u32,
    NH: u32,
    NKVH: u32,
    D_a: u32,
    r: u32,
) -> PyResult<()> {
    #[cfg(feature = "metal")]
    {
        // 1. Bind 16 buffers (x, l, i, h_fwd, grad_out, output_grads)
        // 2. Dispatch 'ltv_bwd_qkv'
        crate::ltv_fused_scan_metal::ltv_fused_scan_metal_backward(
            py, q, k, v, logits, inits, h_fwd_q, h_fwd_k, h_fwd_v, go_q, go_k, go_v, dx_q, dx_k,
            dx_v, dl, di, sbq, shq, stq, sdq, sbk, shk, stk, sdk, sbv, shv, stv, sdv, B, T, NH,
            NKVH, D_a, r,
        )
    }
    #[cfg(not(feature = "metal"))]
    {
        Err(metal_feature_error())
    }
}

#[cfg(feature = "python")]
#[pyfunction(name = "init_ltv_fused_scan_metal_context_for_default_device")]
pub fn init_ltv_fused_scan_metal_context_for_default_device_py() -> PyResult<()> {
    #[cfg(feature = "metal")]
    {
        crate::ltv_fused_scan_metal::init_ltv_fused_scan_metal_context_for_default_device()
    }
    #[cfg(not(feature = "metal"))]
    {
        Err(metal_feature_error())
    }
}

#[cfg(feature = "python")]
#[pyfunction(name = "release_ltv_fused_scan_metal_context")]
pub fn release_ltv_fused_scan_metal_context_py() {
    #[cfg(feature = "metal")]
    {
        crate::ltv_fused_scan_metal::release_ltv_fused_scan_metal_context();
    }
}

#[cfg(feature = "python")]
#[pyfunction(name = "ltv_fused_concat_scan_metal_forward")]
pub fn ltv_fused_concat_scan_metal_forward_py(
    py: Python<'_>,
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
    #[cfg(feature = "metal")]
    {
        crate::ltv_fused_concat_scan_metal::ltv_fused_concat_scan_metal_forward(
            py, combined, logit_bias, inits, out_q, out_k, out_v, b, t, sc_b, sc_t, sc_d, nh, nkvh,
            da, r,
        )
    }
    #[cfg(not(feature = "metal"))]
    {
        Err(metal_feature_error())
    }
}

#[cfg(feature = "python")]
#[pyfunction(name = "ltv_fused_concat_scan_metal_backward")]
pub fn ltv_fused_concat_scan_metal_backward_py(
    py: Python<'_>,
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
    #[cfg(feature = "metal")]
    {
        // 1. Bind 16 buffers (x, l, i, h_fwd, grad_out, output_grads)
        // 2. Dispatch 'ltv_bwd_qkv'
        crate::ltv_fused_concat_scan_metal::ltv_fused_concat_scan_metal_backward(
            py,
            combined,
            logit_bias,
            inits,
            h_fwd_q,
            h_fwd_k,
            h_fwd_v,
            go_q,
            go_k,
            go_v,
            grad_combined,
            grad_inits,
            b,
            t,
            sc_b,
            sc_t,
            sc_d,
            sgc_b,
            sgc_t,
            sgc_d,
            sbq,
            shq,
            stq,
            sdq,
            sbk,
            shk,
            stk,
            sdk,
            sbv,
            shv,
            stv,
            sdv,
            nh,
            nkvh,
            da,
            r,
        )
    }
    #[cfg(not(feature = "metal"))]
    {
        Err(metal_feature_error())
    }
}

#[cfg(feature = "python")]
#[pyfunction(name = "init_ltv_fused_concat_scan_metal_context_for_default_device")]
pub fn init_ltv_fused_concat_scan_metal_context_for_default_device_py() -> PyResult<()> {
    #[cfg(feature = "metal")]
    {
        crate::ltv_fused_concat_scan_metal::init_ltv_fused_concat_scan_metal_context_for_default_device()
    }
    #[cfg(not(feature = "metal"))]
    {
        Err(metal_feature_error())
    }
}

#[cfg(feature = "python")]
#[pyfunction(name = "release_ltv_fused_concat_scan_metal_context")]
pub fn release_ltv_fused_concat_scan_metal_context_py() {
    #[cfg(feature = "metal")]
    {
        crate::ltv_fused_concat_scan_metal::release_ltv_fused_concat_scan_metal_context();
    }
}

#[cfg(feature = "python")]
#[pyfunction(name = "ltv_fused_concat_register_cast_scan_metal_forward")]
pub fn ltv_fused_concat_register_cast_scan_metal_forward_py(
    py: Python<'_>,
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
    #[cfg(feature = "metal")]
    {
        crate::ltv_fused_concat_scan_metal::ltv_fused_concat_register_cast_scan_metal_forward(
            py, combined, inits, out_q, out_k, out_v, b, t, sc_b, sc_t, sc_d, nh, nkvh, da, r,
        )
    }
    #[cfg(not(feature = "metal"))]
    {
        Err(metal_feature_error())
    }
}

#[cfg(feature = "python")]
#[pyfunction(name = "ltv_fused_concat_register_cast_scan_metal_backward")]
pub fn ltv_fused_concat_register_cast_scan_metal_backward_py(
    py: Python<'_>,
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
    #[cfg(feature = "metal")]
    {
        crate::ltv_fused_concat_scan_metal::ltv_fused_concat_register_cast_scan_metal_backward(
            py,
            combined,
            inits,
            h_fwd_q,
            h_fwd_k,
            h_fwd_v,
            go_q,
            go_k,
            go_v,
            grad_combined,
            grad_inits,
            b,
            t,
            sc_b,
            sc_t,
            sc_d,
            sgc_b,
            sgc_t,
            sgc_d,
            sbq,
            shq,
            stq,
            sdq,
            sbk,
            shk,
            stk,
            sdk,
            sbv,
            shv,
            stv,
            sdv,
            nh,
            nkvh,
            da,
            r,
        )
    }
    #[cfg(not(feature = "metal"))]
    {
        Err(metal_feature_error())
    }
}

#[cfg(feature = "python")]
#[pyfunction(name = "init_ltv_fused_concat_register_cast_scan_metal_context_for_default_device")]
pub fn init_ltv_fused_concat_register_cast_scan_metal_context_for_default_device_py() -> PyResult<()>
{
    #[cfg(feature = "metal")]
    {
        crate::ltv_fused_concat_scan_metal::init_ltv_fused_concat_register_cast_scan_metal_context_for_default_device()
    }
    #[cfg(not(feature = "metal"))]
    {
        Err(metal_feature_error())
    }
}

#[cfg(feature = "python")]
#[pyfunction(name = "release_ltv_fused_concat_register_cast_scan_metal_context")]
pub fn release_ltv_fused_concat_register_cast_scan_metal_context_py() {
    #[cfg(feature = "metal")]
    {
        crate::ltv_fused_concat_scan_metal::release_ltv_fused_concat_register_cast_scan_metal_context();
    }
}

#[cfg(feature = "python")]
#[pyfunction(name = "ltv_metal_fused_concat_register_cast_blelloch_scan_forward")]
pub fn ltv_metal_fused_concat_register_cast_blelloch_scan_forward_py(
    combined: Py<PyAny>,
    inits: Py<PyAny>,
    out_q: Py<PyAny>,
    out_k: Py<PyAny>,
    out_v: Py<PyAny>,
    sc: (u32, u32, u32),
    b: u32,
    t: u32,
    nh: u32,
    nkvh: u32,
    da: u32,
    r: u32,
    use_sigmoid: bool,
) -> PyResult<()> {
    #[cfg(feature = "metal")]
    {
        crate::ltv_fused_concat_register_cast_blelloch_scan::ltv_metal_fused_concat_register_cast_blelloch_scan_forward(
            combined, inits, out_q, out_k, out_v, sc, b, t, nh, nkvh, da, r, use_sigmoid,
        )
    }
    #[cfg(not(feature = "metal"))]
    {
        Err(metal_feature_error())
    }
}

#[cfg(feature = "python")]
#[pyfunction(name = "ltv_metal_fused_concat_register_cast_blelloch_scan_backward")]
pub fn ltv_metal_fused_concat_register_cast_blelloch_scan_backward_py(
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
    t: u32,
    nh: u32,
    nkvh: u32,
    da: u32,
    r: u32,
    use_sigmoid: bool,
) -> PyResult<()> {
    #[cfg(feature = "metal")]
    {
        crate::ltv_fused_concat_register_cast_blelloch_scan::ltv_metal_fused_concat_register_cast_blelloch_scan_backward(
            combined,
            inits,
            hq,
            hk,
            hv,
            grad_hq,
            grad_hk,
            grad_hv,
            grad_combined,
            grad_inits,
            grad_logits,
            sc_in,
            sc_out,
            b,
            t,
            nh,
            nkvh,
            da,
            r,
            use_sigmoid,
        )
    }
    #[cfg(not(feature = "metal"))]
    {
        Err(metal_feature_error())
    }
}

#[cfg(feature = "python")]
#[pyfunction(
    name = "init_ltv_metal_fused_concat_register_cast_blelloch_scan_context_for_default_device",
)]
pub fn init_ltv_metal_fused_concat_register_cast_blelloch_scan_context_for_default_device_py(
    thread_execution_width: u32,
    max_threads: u32,
    max_tg_mem: u64,
    banks: u32,
) -> PyResult<()> {
    #[cfg(feature = "metal")]
    {
        crate::ltv_fused_concat_register_cast_blelloch_scan::init_ltv_metal_fused_concat_register_cast_blelloch_scan_for_default_device(
            thread_execution_width,
            max_threads,
            max_tg_mem,
            banks,
        )
    }
    #[cfg(not(feature = "metal"))]
    {
        Err(metal_feature_error())
    }
}

#[cfg(feature = "python")]
#[pyfunction(name = "release_ltv_metal_fused_concat_register_cast_blelloch_scan_context")]
pub fn release_ltv_metal_fused_concat_register_cast_blelloch_scan_context_py() {
    #[cfg(feature = "metal")]
    {
        crate::ltv_fused_concat_register_cast_blelloch_scan::release_ltv_metal_fused_concat_register_cast_blelloch_scan_for_default_device();
    }
}

#[cfg(feature = "python")]
#[pyfunction(name = "ltv_fused_linear_scan_metal_forward")]
pub fn ltv_fused_linear_scan_metal_forward_py(
    py: Python<'_>,
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
    #[cfg(feature = "metal")]
    {
        crate::ltv_fused_linear_scan_metal::ltv_fused_linear_scan_metal_forward(
            py, q, k, v, x_gate, w_gate, b_gate, inits, out_q, out_k, out_v, b, t, nh, nkvh, da, r,
            n_embd,
        )
    }
    #[cfg(not(feature = "metal"))]
    {
        Err(metal_feature_error())
    }
}

#[cfg(feature = "python")]
#[pyfunction(name = "ltv_fused_linear_scan_metal_backward")]
pub fn ltv_fused_linear_scan_metal_backward_py(
    py: Python<'_>,
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
    #[cfg(feature = "metal")]
    {
        crate::ltv_fused_linear_scan_metal::ltv_fused_linear_scan_metal_backward(
            py, q, k, v, x_gate, w_gate, b_gate, inits, h_fwd_q, h_fwd_k, h_fwd_v, go_q, go_k,
            go_v, dx_q, dx_k, dx_v, dx_gate, dw_gate, db_gate, di, sbq, shq, stq, sdq, sbk, shk,
            stk, sdk, sbv, shv, stv, sdv, b, t, nh, nkvh, da, r,
        )
    }
    #[cfg(not(feature = "metal"))]
    {
        Err(metal_feature_error())
    }
}

#[cfg(feature = "python")]
#[pyfunction(name = "init_ltv_fused_linear_scan_metal_context_for_default_device")]
pub fn init_ltv_fused_linear_scan_metal_context_for_default_device_py() -> PyResult<()> {
    #[cfg(feature = "metal")]
    {
        crate::ltv_fused_linear_scan_metal::init_ltv_fused_linear_scan_metal_context_for_default_device()
    }
    #[cfg(not(feature = "metal"))]
    {
        Err(metal_feature_error())
    }
}

#[cfg(feature = "python")]
#[pyfunction(name = "release_ltv_fused_linear_scan_metal_context")]
pub fn release_ltv_fused_linear_scan_metal_context_py() {
    #[cfg(feature = "metal")]
    {
        crate::ltv_fused_linear_scan_metal::release_ltv_fused_linear_scan_metal_context();
    }
}

#[cfg(feature = "python")]
#[pyfunction(name = "ltv_plane_metal")]
pub fn ltv_plane_metal_py(
    py: Python<'_>,
    alpha: Py<PyAny>,
    x: Py<PyAny>,
    out_a: Py<PyAny>,
    out_x: Py<PyAny>,
    t_seq: u32,
    parallel_a: u32,
    r: u32,
    schedule: &str,
) -> PyResult<()> {
    #[cfg(feature = "metal")]
    {
        crate::ltv_plane_metal::ltv_plane_metal(
            py, alpha, x, out_a, out_x, t_seq, parallel_a, r, schedule,
        )
    }
    #[cfg(not(feature = "metal"))]
    {
        Err(metal_feature_error())
    }
}

#[cfg(feature = "python")]
#[pyfunction(name = "init_ltv_plane_metal_context_for_default_device")]
pub fn init_ltv_plane_metal_context_for_default_device_py() -> PyResult<()> {
    #[cfg(feature = "metal")]
    {
        crate::ltv_plane_metal::init_ltv_plane_metal_context_for_default_device()
    }
    #[cfg(not(feature = "metal"))]
    {
        Err(metal_feature_error())
    }
}

#[cfg(feature = "python")]
#[pyfunction(name = "release_ltv_plane_metal_context")]
pub fn release_ltv_plane_metal_context_py() {
    #[cfg(feature = "metal")]
    {
        crate::ltv_plane_metal::release_ltv_plane_metal_context();
    }
}

#[cfg(feature = "python")]
#[pyfunction(name = "ltv_shared_metal")]
pub fn ltv_shared_metal_py(
    py: Python<'_>,
    alpha: Py<PyAny>,
    x: Py<PyAny>,
    out_a: Py<PyAny>,
    out_x: Py<PyAny>,
    t_seq: u32,
    parallel_a: u32,
    r: u32,
    schedule: &str,
) -> PyResult<()> {
    #[cfg(feature = "metal")]
    {
        crate::ltv_shared_metal::ltv_shared_metal(
            py, alpha, x, out_a, out_x, t_seq, parallel_a, r, schedule,
        )
    }
    #[cfg(not(feature = "metal"))]
    {
        Err(metal_feature_error())
    }
}

#[cfg(feature = "python")]
#[pyfunction(name = "init_ltv_shared_metal_context_for_default_device")]
pub fn init_ltv_shared_metal_context_for_default_device_py() -> PyResult<()> {
    #[cfg(feature = "metal")]
    {
        crate::ltv_shared_metal::init_ltv_shared_metal_context_for_default_device()
    }
    #[cfg(not(feature = "metal"))]
    {
        Err(metal_feature_error())
    }
}

#[cfg(feature = "python")]
#[pyfunction(name = "release_ltv_shared_metal_context")]
pub fn release_ltv_shared_metal_context_py() {
    #[cfg(feature = "metal")]
    {
        crate::ltv_shared_metal::release_ltv_shared_metal_context();
    }
}
