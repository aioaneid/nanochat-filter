use std::hint::black_box;
use std::time::Instant;

use crate::strategies::min_balanced_product_flex;

pub fn time_op_cpu<F>(mut op: F, warmup: u32, repeat: u32) -> f64
where
    F: FnMut(),
{
    for _ in 0..warmup {
        black_box(op());
    }

    let start = Instant::now();
    for _ in 0..repeat {
        black_box(op());
    }
    start.elapsed().as_secs_f64() / (repeat as f64)
}

pub fn generate_equal_schedules(
    t: u32,
    min_depth: usize,
    max_depth: usize,
) -> Vec<(String, Vec<u32>)> {
    let mut schedules = Vec::new();
    for depth in (min_depth..=max_depth).rev() {
        let c = (t as f64).powf(1.0 / depth as f64).ceil() as u32;
        if c >= 2 {
            let product: u64 = (c as u64).pow(depth as u32);
            if product >= t as u64 {
                schedules.push((format!("Equal {}", c), vec![c; depth]));
            }
        }
    }
    schedules
}

pub fn generate_min_balanced_schedules(
    t: u32,
    max_balance_k: u32,
    max_diff: u32,
) -> Vec<(String, Vec<u32>)> {
    let mut schedules = Vec::new();
    for k in 2..=max_balance_k {
        let m = min_balanced_product_flex(t, k, max_diff);
        schedules.push((format!("MinBal k={}", k), m));
    }
    schedules
}

pub fn make_eager_schedule(t: u32, max_chunk_size: u32) -> (String, Vec<u32>) {
    if t == 0 {
        return (format!("Eager({})", max_chunk_size), vec![]);
    }
    let mut schedule = vec![max_chunk_size as u32; t.ilog(max_chunk_size) as usize];
    let p: u32 = schedule.iter().product();
    if p < t as u32 {
        schedule.push((t as u32 + p - 1) / p);
    }
    return (format!("Eager({})", max_chunk_size), schedule);
}
