#include <metal_stdlib>
using namespace metal;

// Shared pointer infrastructure for backward scan kernels
// Templated to support both float (legacy) and bfloat16 (register-cast)
template<typename T>
struct Pointers {
    device const T* x;
    device const T* l;
    device const T* tr;
    uint h_fwd_off;
};

template<typename T>
inline Pointers<T> get_ptrs(
    device const T* combined,
    device const T* h_fwd_q, device const T* h_fwd_k, device const T* h_fwd_v,
    device const T* go_q, device const T* go_k, device const T* go_v,
    constant uint3& sc, constant uint4& sq, constant uint4& sk, constant uint4& sv,
    uint b, uint h_glob, uint d, uint Da, uint r, uint T_size, uint NH, uint NKVH
) {
    const uint D = Da * r;
    const uint total_h = NH + 2 * NKVH;
    const uint da_idx = d / r;
    const uint offset_l = total_h * D;
    const uint last_t = T_size - 1;

    device const T *h_ptr, *go_ptr;
    uint h_loc, h_comp_total, offset_in_row;
    uint4 s;

    if (h_glob < NH) {
        h_ptr = h_fwd_q; go_ptr = go_q; h_loc = h_glob; h_comp_total = NH;
        offset_in_row = 0; s = sq;
    } else if (h_glob < NH + NKVH) {
        h_ptr = h_fwd_k; go_ptr = go_k; h_loc = h_glob - NH; h_comp_total = NKVH;
        offset_in_row = NH * D; s = sk;
    } else {
        h_ptr = h_fwd_v; go_ptr = go_v; h_loc = h_glob - (NH + NKVH); h_comp_total = NKVH;
        offset_in_row = (NH + NKVH) * D; s = sv;
    }

    Pointers<T> p;
    p.x = combined + (b * sc.x) + (last_t * sc.y) + (offset_in_row + h_loc * D + d) * sc.z;
    p.l = combined + (b * sc.x) + (last_t * sc.y) + (offset_l + h_glob * Da + da_idx) * sc.z;
    p.tr = go_ptr + (b * s.x + h_loc * s.y + last_t * s.z + d * s.w);
    p.h_fwd_off = b * (h_comp_total * T_size * D) + h_loc * (T_size * D) + (last_t - 1) * D + d;
    return p;
}

// Store a float into an atomic uint slot
inline void atomic_store_float_tg(threadgroup atomic_uint* addr, float value) {
    atomic_store_explicit(addr, as_type<uint>(value), memory_order_relaxed);
}

// Load a float from an atomic uint slot
inline float atomic_load_float_tg(threadgroup atomic_uint* addr) {
    return as_type<float>(atomic_load_explicit(addr, memory_order_relaxed));
}

// Atomic add for floats using a uint compare-exchange loop
inline void atomic_add_float_tg(threadgroup atomic_uint* addr, float delta) {
    uint expected_u = atomic_load_explicit(addr, memory_order_relaxed);
    while (true) {
        float expected_f = as_type<float>(expected_u);
        float desired_f = expected_f + delta;
        uint desired_u = as_type<uint>(desired_f);
        if (atomic_compare_exchange_weak_explicit(addr, &expected_u, desired_u, memory_order_relaxed, memory_order_relaxed)) {
            break;
        }
    }
}
