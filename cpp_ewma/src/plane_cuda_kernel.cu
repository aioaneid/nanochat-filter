#include <c10/cuda/CUDAStream.h>
#include <cuda_runtime.h>
#include <torch/extension.h>

#include <cmath>
#include <vector>

struct PlaneLevelGeom {
  int n_chunks;
  int t_orig;
  int t_padded;
  int chunk_size;
};

static inline std::vector<PlaneLevelGeom>
make_plane_geometry_max(int t_seq, int max_chunk_size) {
  std::vector<PlaneLevelGeom> geometry;
  if (t_seq == 0) {
    return geometry;
  }

  int curr = t_seq;
  do {
    int chunk_size = std::min(curr, max_chunk_size);
    int n_chunks = (curr + chunk_size - 1) / chunk_size;
    int t_padded = n_chunks * chunk_size;

    geometry.push_back(PlaneLevelGeom{n_chunks, curr, t_padded, chunk_size});
    curr = n_chunks;
  } while (curr > 1);

  return geometry;
}

static inline std::vector<PlaneLevelGeom>
make_plane_geometry_log(int t_seq, int max_chunk_size) {
  double depth =
      (t_seq <= 1) ? 1.0
                   : std::log((double)t_seq) / std::log((double)max_chunk_size);
  int d = (int)std::ceil(depth);
  d = std::max(1, d);

  double b_f64 = std::pow((double)t_seq, 1.0 / (double)d);
  int b = (int)std::ceil(b_f64);
  b = std::max(2, b);
  b = std::min(b, max_chunk_size);

  return make_plane_geometry_max(t_seq, b);
}

template <typename scalar_t>
__global__ void plane_up_sweep_kernel(
    const scalar_t *__restrict__ input_a, const scalar_t *__restrict__ input_x,
    scalar_t *__restrict__ stack_a, scalar_t *__restrict__ stack_x,
    scalar_t *__restrict__ summary_a, scalar_t *__restrict__ summary_x,
    int t_orig, int chunk_size, int r, int t_padded, int parallel_a) {
  int chunk_idx = blockIdx.x;
  int line_id_x = blockIdx.y;
  int unit_idx = threadIdx.x;

  int line_id_a = line_id_x / r;
  bool is_index_zero_of_r = (line_id_a * r) == line_id_x;

  int global_t = chunk_idx * chunk_size + unit_idx;

  int in_idx_a = line_id_a * t_orig + global_t;
  int in_idx_x = line_id_x * t_orig + global_t;
  int idx_a = line_id_a * t_padded + global_t;
  int idx_x = line_id_x * t_padded + global_t;

  bool within_input = global_t < t_orig;
  float cur_a = within_input ? (float)input_a[in_idx_a] : 1.0f;
  float cur_x = within_input ? (float)input_x[in_idx_x] : 0.0f;

  unsigned mask = (chunk_size == 32) ? 0xffffffffu : ((1u << chunk_size) - 1u);
  for (int offset = 1; offset < chunk_size; offset <<= 1) {
    float n_a = __shfl_up_sync(mask, cur_a, offset);
    float n_x = __shfl_up_sync(mask, cur_x, offset);
    if (unit_idx >= offset) {
      cur_x += cur_a * n_x;
      cur_a *= n_a;
    }
  }

  if (is_index_zero_of_r) {
    stack_a[idx_a] = (scalar_t)cur_a;
  }
  stack_x[idx_x] = (scalar_t)cur_x;

  if (unit_idx == chunk_size - 1) {
    int n_chunks = t_padded / chunk_size;
    if (is_index_zero_of_r) {
      summary_a[(line_id_a * n_chunks) + chunk_idx] = (scalar_t)cur_a;
    }
    summary_x[(line_id_x * n_chunks) + chunk_idx] = (scalar_t)cur_x;
  }
}

template <typename scalar_t>
__global__ void plane_down_sweep_kernel(
    const scalar_t *__restrict__ stack_a, const scalar_t *__restrict__ stack_x,
    const scalar_t *__restrict__ carry_a, const scalar_t *__restrict__ carry_x,
    scalar_t *__restrict__ output_a, scalar_t *__restrict__ output_x,
    int t_orig, int chunk_size, int r, int t_padded, int parallel_a) {
  int chunk_idx = blockIdx.x;
  int line_id_x = blockIdx.y;
  int unit_idx = threadIdx.x;

  int line_id_a = line_id_x / r;
  bool is_index_zero_of_r = (line_id_a * r) == line_id_x;

  __shared__ float shm_carry_a;
  __shared__ float shm_carry_x;

  if (unit_idx == 0) {
    if (chunk_idx > 0) {
      int n_chunks = t_padded / chunk_size;
      if (is_index_zero_of_r) {
        int carry_idx_a = line_id_a * n_chunks + (chunk_idx - 1);
        shm_carry_a = (float)carry_a[carry_idx_a];
      }
      int carry_idx_x = line_id_x * n_chunks + (chunk_idx - 1);
      shm_carry_x = (float)carry_x[carry_idx_x];
    } else {
      if (is_index_zero_of_r) {
        shm_carry_a = 1.0f;
      }
      shm_carry_x = 0.0f;
    }
  }
  __syncthreads();

  int global_t = chunk_idx * chunk_size + unit_idx;
  if (global_t < t_orig) {
    int idx_a_out = line_id_a * t_orig + global_t;
    int idx_x_out = line_id_x * t_orig + global_t;

    int idx_a_stack = line_id_a * t_padded + global_t;
    int idx_x_stack = line_id_x * t_padded + global_t;

    float s_a = (float)stack_a[idx_a_stack];
    float s_x = (float)stack_x[idx_x_stack];

    float res_x = s_a * shm_carry_x + s_x;
    if (is_index_zero_of_r) {
      float res_a = s_a * shm_carry_a;
      output_a[idx_a_out] = (scalar_t)res_a;
    }
    output_x[idx_x_out] = (scalar_t)res_x;
  }
}

void ltv_plane_cuda(torch::Tensor alpha, torch::Tensor x,
                    torch::Tensor out_alpha, torch::Tensor out_x, int t_seq,
                    int parallel_a, int r, int schedule_kind) {
  TORCH_CHECK(alpha.is_cuda() && x.is_cuda(), "alpha/x must be CUDA");
  TORCH_CHECK(alpha.is_contiguous() && x.is_contiguous(),
              "alpha/x must be contiguous");
  TORCH_CHECK(out_alpha.is_cuda() && out_x.is_cuda(), "out must be CUDA");
  TORCH_CHECK(out_alpha.is_contiguous() && out_x.is_contiguous(),
              "out must be contiguous");

  if (parallel_a == 0) {
    return;
  }

  constexpr int kMaxChunk = 32;
  std::vector<PlaneLevelGeom> geometry =
      (schedule_kind == 0) ? make_plane_geometry_max(t_seq, kMaxChunk)
                           : make_plane_geometry_log(t_seq, kMaxChunk);

  int parallel_x = parallel_a * r;

  cudaStream_t stream = at::cuda::getCurrentCUDAStream();

  torch::Tensor curr_a = alpha;
  torch::Tensor curr_x = x;

  struct LevelBuf {
    torch::Tensor stack_a;
    torch::Tensor stack_x;
    PlaneLevelGeom geom;
  };
  std::vector<LevelBuf> levels;

  for (const auto &geom : geometry) {
    auto stack_a = torch::empty({parallel_a, geom.t_padded}, curr_a.options());
    auto stack_x = torch::empty({parallel_x, geom.t_padded}, curr_x.options());
    auto summary_a =
        torch::empty({parallel_a, geom.n_chunks}, curr_a.options());
    auto summary_x =
        torch::empty({parallel_x, geom.n_chunks}, curr_x.options());

    dim3 grid(geom.n_chunks, parallel_x);
    dim3 block(geom.chunk_size);

    AT_DISPATCH_FLOATING_TYPES(
        curr_a.scalar_type(), "plane_up_sweep", ([&] {
          plane_up_sweep_kernel<scalar_t><<<grid, block, 0, stream>>>(
              curr_a.data_ptr<scalar_t>(), curr_x.data_ptr<scalar_t>(),
              stack_a.data_ptr<scalar_t>(), stack_x.data_ptr<scalar_t>(),
              summary_a.data_ptr<scalar_t>(), summary_x.data_ptr<scalar_t>(),
              geom.t_orig, geom.chunk_size, r, geom.t_padded, parallel_a);
        }));

    levels.push_back(LevelBuf{stack_a, stack_x, geom});

    curr_a = summary_a;
    curr_x = summary_x;
  }

  torch::Tensor carry_a = curr_a;
  torch::Tensor carry_x = curr_x;

  for (int li = (int)levels.size() - 1; li >= 0; --li) {
    const auto &level = levels[li];

    torch::Tensor target_a;
    torch::Tensor target_x;

    if (li == 0) {
      target_a = out_alpha;
      target_x = out_x;
    } else {
      target_a = torch::empty({parallel_a, level.geom.t_orig}, alpha.options());
      target_x = torch::empty({parallel_x, level.geom.t_orig}, x.options());
    }

    dim3 grid(level.geom.n_chunks, parallel_x);
    dim3 block(level.geom.chunk_size);

    AT_DISPATCH_FLOATING_TYPES(
        alpha.scalar_type(), "plane_down_sweep", ([&] {
          plane_down_sweep_kernel<scalar_t><<<grid, block, 0, stream>>>(
              level.stack_a.data_ptr<scalar_t>(),
              level.stack_x.data_ptr<scalar_t>(), carry_a.data_ptr<scalar_t>(),
              carry_x.data_ptr<scalar_t>(), target_a.data_ptr<scalar_t>(),
              target_x.data_ptr<scalar_t>(), level.geom.t_orig,
              level.geom.chunk_size, r, level.geom.t_padded, parallel_a);
        }));

    carry_a = target_a;
    carry_x = target_x;
  }
}
