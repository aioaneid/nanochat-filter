#include <cublas_v2.h>
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>

// Helper to reliably map PyTorch ScalarTypes directly to cuBLAS data types
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
        TORCH_CHECK(false, "Unsupported tensor data type for fused_linear_transpose: ", tensor.scalar_type());
    }
}

void fused_linear_transpose_impl(torch::Tensor x, torch::Tensor weight,
                                 torch::Tensor out)
{
    // x: (B, T, K), weight: (F, K), out: (B, F, T) allocated in Python
    const auto B = x.size(0),
               T = x.size(1), K = x.size(2), F = weight.size(0);

    // 1. Resolve types dynamically
    cudaDataType_t typeX = get_cublas_datatype(x);
    cudaDataType_t typeW = get_cublas_datatype(weight);
    cudaDataType_t typeOut = get_cublas_datatype(out);

    // 2. Fetch the global context handles and stream
    cublasHandle_t handle = at::cuda::getCurrentCUDABlasHandle();
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    cublasSetStream(handle, stream);

    const float alpha = 1.0f, beta = 0.0f;

    // 3. Strided Batched GEMM Execution (Row-Major to Column-Major Mapping)
    cublasGemmStridedBatchedEx(
        handle,
        CUBLAS_OP_T, CUBLAS_OP_N, // Swapped flags to handle row-major memory mapping
        T, F, K,                  // m=T, n=F, k=K
        &alpha,
        x.data_ptr(), typeX, K,      // A = x      | lda = K (Valid because transa=T requires lda >= K)
        T * K,                       // strideA = T * K
        weight.data_ptr(), typeW, K, // B = weight | ldb = K (Valid because transb=N requires ldb >= K)
        0,                           // strideB = 0 (broadcast weight across all batches)
        &beta,
        out.data_ptr(), typeOut, T, // C = out    | ldc = T (Valid because ldc >= m, and T >= T)
        F * T,                      // strideC = F * T
        B,                          // batch count
        CUBLAS_COMPUTE_32F,         // internal math accumulation precision
        CUBLAS_GEMM_DEFAULT         // default execution algorithm
    );
}
