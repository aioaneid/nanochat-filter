#![cfg(all(feature = "metal", feature = "python"))]

use crate::ltv_utils::LevelGeometry;

use metal::{
    Buffer, BufferRef, CommandBufferRef, CompileOptions, ComputePipelineState, Device,
    MTLResourceOptions, MTLSize, foreign_types::ForeignTypeRef,
};

use arrayvec::ArrayVec;
use pyo3::{exceptions::PyRuntimeError, prelude::*};
use std::ffi::c_void;

pub struct Level {
    pub stack_a: Buffer,
    pub stack_x: Buffer,
    pub geom: LevelGeometry,
}

pub struct MetalContext {
    pub device: Device,
    pub up_sweep_pipeline: ComputePipelineState,
    pub down_sweep_pipeline: ComputePipelineState,
}

// Struct to hold all data needed during the callback
pub struct EncodingContext<'a> {
    pub metal_ctx: &'static MetalContext,
    pub geometry: &'a [LevelGeometry],

    // Input Buffers
    pub b_alpha: &'a BufferRef,
    pub off_alpha: u64,
    pub b_x: &'a BufferRef,
    pub off_x: u64,
    pub b_out_alpha: &'a BufferRef,
    pub off_out_alpha: u64,
    pub b_out_x: &'a BufferRef,
    pub off_out_x: u64,

    // Params
    pub parallel_a: u32,
    pub r: u32,
    pub t_seq: u32,
}

const DOWN_SWEEP_KERNEL_SOURCE: &str = include_str!("ltv_blelloch_down_sweep.msl");

pub fn metal_context_for_default_device(
    up_sweep_kernel_source: &str,
    up_sweep_kernel_name: &str,
) -> PyResult<MetalContext> {
    let device = Device::system_default()
        .ok_or_else(|| PyErr::new::<PyRuntimeError, _>("No Metal device found"))?;

    let up_lib = device
        .new_library_with_source(up_sweep_kernel_source, &CompileOptions::new())
        .map_err(|e| PyErr::new::<PyRuntimeError, _>(e.to_string()))?;

    let up_fn = up_lib
        .get_function(up_sweep_kernel_name, None)
        .map_err(|e| PyErr::new::<PyRuntimeError, _>(e.to_string()))?;

    let up_pipeline = device
        .new_compute_pipeline_state_with_function(&up_fn)
        .map_err(|e| PyErr::new::<PyRuntimeError, _>(e.to_string()))?;

    let down_lib = device
        .new_library_with_source(DOWN_SWEEP_KERNEL_SOURCE, &CompileOptions::new())
        .map_err(|e| PyErr::new::<PyRuntimeError, _>(e.to_string()))?;

    let down_fn = down_lib
        .get_function("down_sweep_kernel", None)
        .map_err(|e| PyErr::new::<PyRuntimeError, _>(e.to_string()))?;

    let down_pipeline = device
        .new_compute_pipeline_state_with_function(&down_fn)
        .map_err(|e| PyErr::new::<PyRuntimeError, _>(e.to_string()))?;

    Ok(MetalContext {
        device,
        up_sweep_pipeline: up_pipeline,
        down_sweep_pipeline: down_pipeline,
    })
}

pub extern "C" fn encode_callback(cmd_buf_ptr: *mut c_void, context_ptr: *mut c_void) {
    unsafe {
        let cmd_buf = CommandBufferRef::from_ptr(cmd_buf_ptr as *mut _);
        let ctx = &*(context_ptr as *const EncodingContext);

        // Track the "Current Result" (Source for the next operation)
        // Initialize with the raw inputs.
        let mut curr_a = ctx.b_alpha.to_owned();
        let mut curr_a_off = ctx.off_alpha;
        let mut curr_x = ctx.b_x.to_owned();
        let mut curr_x_off = ctx.off_x;

        let mut levels = ArrayVec::<Level, 10>::new();

        // --- UP-SWEEP ---
        // If geometry is empty, this loop is skipped.
        // curr_a/curr_x remain pointing to inputs.
        for geom in ctx.geometry {
            let stack_a = ctx.metal_ctx.device.new_buffer(
                geom.stack_a_bytes as u64,
                MTLResourceOptions::StorageModePrivate,
            );
            let stack_x = ctx.metal_ctx.device.new_buffer(
                geom.stack_x_bytes as u64,
                MTLResourceOptions::StorageModePrivate,
            );
            let summary_a = ctx.metal_ctx.device.new_buffer(
                geom.summary_a_bytes as u64,
                MTLResourceOptions::StorageModePrivate,
            );
            let summary_x = ctx.metal_ctx.device.new_buffer(
                geom.summary_x_bytes as u64,
                MTLResourceOptions::StorageModePrivate,
            );

            let encoder = cmd_buf.new_compute_command_encoder();
            encoder.set_compute_pipeline_state(&ctx.metal_ctx.up_sweep_pipeline);

            encoder.set_buffer(0, Some(&curr_a), curr_a_off);
            encoder.set_buffer(1, Some(&curr_x), curr_x_off);
            encoder.set_buffer(2, Some(&stack_a), 0);
            encoder.set_buffer(3, Some(&stack_x), 0);
            encoder.set_buffer(4, Some(&summary_a), 0);
            encoder.set_buffer(5, Some(&summary_x), 0);

            encoder.set_bytes(6, 4, &geom.t_orig as *const _ as *const _);
            encoder.set_bytes(7, 4, &geom.chunk_size as *const _ as *const _);
            encoder.set_bytes(8, 4, &ctx.r as *const _ as *const _);
            encoder.set_bytes(9, 4, &geom.t_padded as *const _ as *const _);

            let grid_size = MTLSize {
                width: geom.n_chunks as u64,
                height: (ctx.parallel_a * ctx.r) as u64,
                depth: 1,
            };
            let thread_group = MTLSize {
                width: geom.chunk_size as u64,
                height: 1,
                depth: 1,
            };

            // Because we dispatch thread groups, rather than threads, there is no
            // need to check that line_id_x < parallel_dim_x and unit_idx < chunk_size
            // inside the kernels.
            encoder.dispatch_thread_groups(grid_size, thread_group);
            encoder.end_encoding();

            levels.push(Level {
                stack_a,
                stack_x,
                geom: *geom,
            });

            // Update "current" to point to the summary of this level
            curr_a = summary_a;
            curr_a_off = 0;
            curr_x = summary_x;
            curr_x_off = 0;
        }

        // --- DOWN-SWEEP ---
        // curr_a/x currently hold the top-most summary (or the input if no levels).
        // These act as the "Carry" into the Down-Sweep.
        let mut carry_a = curr_a;
        let mut carry_a_off = curr_a_off;
        let mut carry_x = curr_x;
        let mut carry_x_off = curr_x_off;

        // If levels is empty, this loop is skipped.
        for level in levels.iter().rev() {
            // Uniform Allocation: Output is tightly packed (based on t_orig)
            let target_a = ctx.metal_ctx.device.new_buffer(
                level.geom.out_a_bytes as u64,
                MTLResourceOptions::StorageModePrivate,
            );
            let target_x = ctx.metal_ctx.device.new_buffer(
                level.geom.out_x_bytes as u64,
                MTLResourceOptions::StorageModePrivate,
            );

            let encoder = cmd_buf.new_compute_command_encoder();
            encoder.set_compute_pipeline_state(&ctx.metal_ctx.down_sweep_pipeline);

            encoder.set_buffer(0, Some(&level.stack_a), 0);
            encoder.set_buffer(1, Some(&level.stack_x), 0);
            encoder.set_buffer(2, Some(&carry_a), carry_a_off);
            encoder.set_buffer(3, Some(&carry_x), carry_x_off);
            encoder.set_buffer(4, Some(&target_a), 0);
            encoder.set_buffer(5, Some(&target_x), 0);

            encoder.set_bytes(6, 4, &level.geom.t_orig as *const _ as *const _);
            encoder.set_bytes(7, 4, &level.geom.chunk_size as *const _ as *const _);
            encoder.set_bytes(8, 4, &ctx.r as *const _ as *const _);
            encoder.set_bytes(9, 4, &level.geom.t_padded as *const _ as *const _);

            let grid_size = MTLSize {
                width: level.geom.n_chunks as u64,
                height: (ctx.parallel_a * ctx.r) as u64,
                depth: 1,
            };
            let thread_group = MTLSize {
                width: level.geom.chunk_size as u64,
                height: 1,
                depth: 1,
            };

            encoder.dispatch_thread_groups(grid_size, thread_group);
            encoder.end_encoding();

            // The output of this level becomes the carry for the next level down
            carry_a = target_a;
            carry_a_off = 0;
            carry_x = target_x;
            carry_x_off = 0;
        }

        let blit_encoder = cmd_buf.new_blit_command_encoder();
        // Copy the full contiguous block.
        // carry_a/x (from Level 0) is size (parallel * t_seq * 4), which matches b_out size.
        let alpha_copy_size = ctx.t_seq * ctx.parallel_a * 4;
        blit_encoder.copy_from_buffer(
            &carry_a,
            carry_a_off,
            ctx.b_out_alpha,
            ctx.off_out_alpha,
            alpha_copy_size as u64,
        );
        blit_encoder.copy_from_buffer(
            &carry_x,
            carry_x_off,
            ctx.b_out_x,
            ctx.off_out_x,
            (alpha_copy_size * ctx.r) as u64,
        );
        blit_encoder.end_encoding();
    }
}
