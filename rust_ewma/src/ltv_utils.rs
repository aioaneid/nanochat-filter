use arrayvec::ArrayVec;
use pyo3::exceptions::PyRuntimeError;
use pyo3::prelude::*;

#[derive(Copy, Clone, Debug)]
pub struct LevelGeometry {
    pub n_chunks: u32,
    pub t_orig: u32,
    pub t_padded: u32,
    pub chunk_size: u32,
    pub stack_a_bytes: usize,
    pub stack_x_bytes: usize,
    pub summary_a_bytes: usize,
    pub summary_x_bytes: usize,
    pub out_a_bytes: usize,
    pub out_x_bytes: usize,
}

pub fn make_scan_geometry(
    t_seq: u32,
    parallel_dim_a: u32,
    r: u32,
    chunk_sizes: &[u32],
) -> ArrayVec<LevelGeometry, 10> {
    let mut geometry = ArrayVec::new();
    let mut curr_t = t_seq;
    let parallel_dim_x = parallel_dim_a * r;
    let elem_size = std::mem::size_of::<f32>() as u32;

    for &chunk_size in chunk_sizes {
        let n_chunks = (curr_t + chunk_size - 1) / chunk_size;
        let t_padded = n_chunks * chunk_size;
        geometry.push(LevelGeometry {
            n_chunks,
            t_orig: curr_t,
            t_padded,
            chunk_size,
            stack_a_bytes: (t_padded * parallel_dim_a * elem_size) as usize,
            stack_x_bytes: (t_padded * parallel_dim_x * elem_size) as usize,
            summary_a_bytes: (n_chunks * parallel_dim_a * elem_size) as usize,
            summary_x_bytes: (n_chunks * parallel_dim_x * elem_size) as usize,
            out_a_bytes: (curr_t * parallel_dim_a * elem_size) as usize,
            out_x_bytes: (curr_t * parallel_dim_x * elem_size) as usize,
        });
        curr_t = n_chunks;
    }

    geometry
}

pub fn make_geometry_greedy_max(
    t_seq: u32,
    parallel_dim_a: u32,
    r: u32,
    max_chunk_size: u32,
) -> ArrayVec<LevelGeometry, 10> {
    if t_seq == 0 {
        return ArrayVec::new();
    }
    let mut chunk_sizes = ArrayVec::<u32, 10>::new();
    let mut curr = t_seq;

    loop {
        let chunk_size = std::cmp::min(curr, max_chunk_size);
        chunk_sizes.push(chunk_size);
        curr = (curr + chunk_size - 1) / chunk_size;

        if curr <= 1 {
            break;
        }
    }
    make_scan_geometry(t_seq, parallel_dim_a, r, &chunk_sizes)
}

pub fn make_geometry_greedy_log(
    t_seq: u32,
    parallel_dim_a: u32,
    r: u32,
    max_chunk_size: u32,
) -> ArrayVec<LevelGeometry, 10> {
    let depth = if t_seq <= 1 {
        1f64
    } else {
        (t_seq as f64).ln() / (max_chunk_size as f64).ln()
    };
    let depth = depth.ceil() as u32;

    let b_f64 = (t_seq as f64).powf(1.0 / depth as f64);
    let b = b_f64.ceil() as u32;
    let b = std::cmp::max(2, b);
    let b = std::cmp::min(b, max_chunk_size);
    make_geometry_greedy_max(t_seq, parallel_dim_a, r, b)
}

pub fn create_geometry_from_schedule(
    schedule: &str,
    t_seq: u32,
    parallel_dim_a: u32,
    r: u32,
    max_chunk_size: u32,
) -> PyResult<ArrayVec<LevelGeometry, 10>> {
    match schedule {
        "max" => Ok(make_geometry_greedy_max(
            t_seq,
            parallel_dim_a,
            r,
            max_chunk_size,
        )),
        "log" => Ok(make_geometry_greedy_log(
            t_seq,
            parallel_dim_a,
            r,
            max_chunk_size,
        )),
        s => Err(PyRuntimeError::new_err(format!("Unknown schedule: {}", s))),
    }
}
