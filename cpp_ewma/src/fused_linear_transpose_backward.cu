#include <cublas_v2.h>
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>

inline cudaDataType_t get_cublas_datatype(const torch::Tensor &tensor)
{
    switch (tensor.scalar_type())
    {
    case torch::kFloat:
        return CUDA_R_32F;
    case torch::kHalf:
        return CUDA_R_16F;
    case torch::kBFloat16:
        return CUDA_R_16BF;
    case torch::kDouble:
        return CUDA_R_64F;
    default:
        TORCH_CHECK(false, "Unsupported tensor data type: ", tensor.scalar_type());
    }
}

void fused_linear_transpose_backward(torch::Tensor x, torch::Tensor weight, torch::Tensor grad_out, torch::Tensor grad_x, torch::Tensor grad_weight)
{
    const auto B = x.size(0);
    const auto T = x.size(1);
    const auto K = x.size(2);
    const auto F = weight.size(0);

    // x and weight are guaranteed contiguous from Python checks
    auto x_safe = x.contiguous();
    auto weight_cast = weight.to(grad_out.scalar_type()).contiguous();

    bool grad_x_is_contiguous = grad_x.is_contiguous();
    torch::Tensor grad_x_safe = grad_x_is_contiguous ? grad_x : torch::empty_like(grad_x);

    cublasHandle_t handle = at::cuda::getCurrentCUDABlasHandle();
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    cublasSetStream(handle, stream);

    const float alpha = 1.0f;
    const float beta0 = 0.0f;

    // Extract exact native data types
    cudaDataType_t type_w = get_cublas_datatype(weight_cast);
    cudaDataType_t type_x = get_cublas_datatype(x_safe);
    cudaDataType_t type_go = get_cublas_datatype(grad_out); // Use grad_out directly
    cudaDataType_t type_gx = get_cublas_datatype(grad_x_safe);
    cudaDataType_t type_gw = get_cublas_datatype(grad_weight);

    // Determine layout properties of grad_out dynamically
    bool go_is_contiguous = grad_out.is_contiguous();

    // For grad_x calculation
    cublasOperation_t op_b_x = go_is_contiguous ? CUBLAS_OP_T : CUBLAS_OP_N;
    int ldb_x = go_is_contiguous ? (int)grad_out.stride(1) : (int)grad_out.stride(2);

    // For grad_weight calculation
    cublasOperation_t op_b_w = go_is_contiguous ? CUBLAS_OP_N : CUBLAS_OP_T;
    int ldb_w = ldb_x;

    // Compute byte offsets using the actual input grad_out tensor
    long long byte_stride_go = grad_out.stride(0) * grad_out.element_size();
    long long byte_stride_gx = grad_x_safe.stride(0) * grad_x_safe.element_size();
    long long byte_stride_x = x_safe.stride(0) * x_safe.element_size();

    // ------------------------------------------------------------------------
    // 1. Compute grad_x -> grad_x[b] = weight^T @ grad_out[b]
    // ------------------------------------------------------------------------
    int m_x = (int)K;
    int n_x = (int)T;
    int k_x = (int)F;

    int lda_x = (int)weight_cast.stride(0); // lda >= m_x
    int ldc_x = (int)grad_x_safe.stride(1); // ldc >= m_x

    for (int b = 0; b < B; ++b)
    {
        void *A_ptr = weight_cast.data_ptr();
        void *B_ptr = (char *)grad_out.data_ptr() + b * byte_stride_go;
        void *C_ptr = (char *)grad_x_safe.data_ptr() + b * byte_stride_gx;

        cublasStatus_t stat = cublasGemmEx(
            handle,
            CUBLAS_OP_N, op_b_x, // Dynamically toggled
            m_x, n_x, k_x,
            &alpha,
            A_ptr, type_w, lda_x,
            B_ptr, type_go, ldb_x, // Dynamically set leading dimension
            &beta0,
            C_ptr, type_gx, ldc_x,
            CUBLAS_COMPUTE_32F,
            CUBLAS_GEMM_DEFAULT);
        TORCH_CHECK(stat == CUBLAS_STATUS_SUCCESS, "cublasGemmEx failed for grad_x at batch element ", b);
    }

    if (!grad_x_is_contiguous)
    {
        grad_x.copy_(grad_x_safe);
    }

    // ------------------------------------------------------------------------
    // 2. Compute grad_weight -> grad_weight = sum_b(grad_out[b] @ x[b]^T)
    // ------------------------------------------------------------------------
    int m_w = (int)K;
    int n_w = (int)F;
    int k_w = (int)T;

    int lda_w = (int)x_safe.stride(1);      // lda >= m_w
    int ldc_w = (int)grad_weight.stride(0); // ldc >= m_w

    for (int b = 0; b < B; ++b)
    {
        void *A_ptr = (char *)x_safe.data_ptr() + b * byte_stride_x;
        void *B_ptr = (char *)grad_out.data_ptr() + b * byte_stride_go;
        void *C_ptr = grad_weight.data_ptr();

        float current_beta = (b == 0) ? 0.0f : 1.0f;

        cublasStatus_t stat = cublasGemmEx(
            handle,
            CUBLAS_OP_N, op_b_w, // Dynamically toggled
            m_w, n_w, k_w,
            &alpha,
            A_ptr, type_x, lda_w,
            B_ptr, type_go, ldb_w, // Dynamically set leading dimension
            &current_beta,
            C_ptr, type_gw, ldc_w,
            CUBLAS_COMPUTE_32F,
            CUBLAS_GEMM_DEFAULT);
        TORCH_CHECK(stat == CUBLAS_STATUS_SUCCESS, "cublasGemmEx failed for grad_weight at batch element ", b);
    }
}
