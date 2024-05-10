#ifndef CUDA_TBC_MULTIBIT_PBS_CUH
#define CUDA_TBC_MULTIBIT_PBS_CUH

#include "cooperative_groups.h"
#include "crypto/gadget.cuh"
#include "crypto/ggsw.cuh"
#include "crypto/torus.cuh"
#include "device.h"
#include "fft/bnsmfft.cuh"
#include "fft/twiddles.cuh"
#include "polynomial/functions.cuh"
#include "polynomial/parameters.cuh"
#include "polynomial/polynomial_math.cuh"
#include "programmable_bootstrap.cuh"
#include "programmable_bootstrap.h"
#include "programmable_bootstrap_multibit.cuh"
#include "types/complex/operations.cuh"
#include <vector>

template <typename Torus, class params, sharedMemDegree SMD>
__global__ void
compute_keybundle(double2 *keybundle_out, Torus *block_lwe_array_in,
                  Torus *bootstrapping_key, uint32_t lwe_dimension,
                  uint32_t glwe_dimension, uint32_t polynomial_size,
                  uint32_t grouping_factor, uint32_t base_log,
                  uint32_t level_count, uint32_t glwe_id, uint32_t level_id,
                  uint32_t lwe_iteration, Torus *device_mem, uint64_t device_memory_size_per_block) {

  extern __shared__ int8_t sharedmem[];
  int8_t *selected_memory = sharedmem;

  if constexpr (SMD == FULLSM) {
    selected_memory = sharedmem;
  } else {
    int block_index = blockIdx.x + blockIdx.y * gridDim.x +
                      blockIdx.z * gridDim.x * gridDim.y;
    selected_memory = &device_mem[block_index * device_memory_size_per_block];
  }

  // Ids
  uint32_t poly_id = blockIdx.x;

    //
    Torus *accumulator = (Torus *)selected_memory;

    ////////////////////////////////////////////////////////////
    // Computes all keybundles
    uint32_t rev_lwe_iteration =
        ((lwe_dimension / grouping_factor) - lwe_iteration - 1);

    // ////////////////////////////////
    // Keygen guarantees the first term is a constant term of the polynomial, no
    // polynomial multiplication required
    Torus *bsk_slice = get_multi_bit_ith_lwe_gth_group_kth_block(
        bootstrapping_key, 0, rev_lwe_iteration, glwe_id, level_id,
        grouping_factor, 2 * polynomial_size, glwe_dimension, level_count);
    Torus *bsk_poly = bsk_slice + poly_id * params::degree;

    copy_polynomial<Torus, params::opt, params::degree / params::opt>(
        bsk_poly, accumulator);

    // Accumulate the other terms
    for (int g = 1; g < (1 << grouping_factor); g++) {

      Torus *bsk_slice = get_multi_bit_ith_lwe_gth_group_kth_block(
          bootstrapping_key, g, rev_lwe_iteration, glwe_id, level_id,
          grouping_factor, 2 * polynomial_size, glwe_dimension, level_count);
      Torus *bsk_poly = bsk_slice + poly_id * params::degree;

      // Calculates the monomial degree
      Torus *lwe_array_group =
          block_lwe_array_in + rev_lwe_iteration * grouping_factor;
      uint32_t monomial_degree = calculates_monomial_degree<Torus, params>(
          lwe_array_group, g, grouping_factor);

      synchronize_threads_in_block();
      // Multiply by the bsk element
      polynomial_product_accumulate_by_monomial<Torus, params>(
          accumulator, bsk_poly, monomial_degree, false);
    }

    synchronize_threads_in_block();

    double2 *fft = (double2 *)selected_memory;

    // Move accumulator to local memory
    double2 temp[params::opt / 2];
    int tid = threadIdx.x;
#pragma unroll
    for (int i = 0; i < params::opt / 2; i++) {
      temp[i].x = __ll2double_rn((int64_t)accumulator[tid]);
      temp[i].y =
          __ll2double_rn((int64_t)accumulator[tid + params::degree / 2]);
      temp[i].x /= (double)std::numeric_limits<Torus>::max();
      temp[i].y /= (double)std::numeric_limits<Torus>::max();
      tid += params::degree / params::opt;
    }

    synchronize_threads_in_block();
    // Move from local memory back to shared memory but as complex
    tid = threadIdx.x;
#pragma unroll
    for (int i = 0; i < params::opt / 2; i++) {
      fft[tid] = temp[i];
      tid += params::degree / params::opt;
    }
    synchronize_threads_in_block();
    NSMFFT_direct<HalfDegree<params>>(fft);

    // lwe iteration
    auto keybundle_poly = keybundle_out + poly_id * params::degree / 2;

    copy_polynomial<double2, params::opt / 2, params::degree / params::opt>(
        fft, keybundle_poly);
}

template <typename Torus, class params, sharedMemDegree SMD>
__global__ void device_multi_bit_programmable_bootstrap(
    Torus *lwe_array_out, Torus *lwe_output_indexes, Torus *lut_vector,
    Torus *lut_vector_indexes, Torus *lwe_array_in, Torus *lwe_input_indexes,
    Torus *bootstrapping_key, double2 *keybundle_array, double2 *join_buffer,
    uint32_t lwe_dimension, uint32_t glwe_dimension, uint32_t polynomial_size,
    uint32_t grouping_factor, uint32_t base_log, uint32_t level_count,
    int8_t *device_mem, uint64_t device_memory_size_per_block,
    bool support_dsm) {

  cluster_group cluster = this_cluster();

  extern __shared__ int8_t sharedmem[];
  int8_t *selected_memory = sharedmem;

  if constexpr (SMD == FULLSM) {
    // The first (polynomial_size/2) * sizeof(double2) bytes are reserved for
    // external product using distributed shared memory
    selected_memory = sharedmem;
    if (support_dsm)
      selected_memory += sizeof(Torus) * polynomial_size;
  } else {
    int block_index = blockIdx.x + blockIdx.y * gridDim.x +
                      blockIdx.z * gridDim.x * gridDim.y;
    selected_memory = &device_mem[block_index * device_memory_size_per_block];
  }

  //
  Torus *accumulator = (Torus *)selected_memory;
  double2 *accumulator_fft =
      (double2 *)accumulator +
      (ptrdiff_t)(sizeof(Torus) * polynomial_size / sizeof(double2));

  if constexpr (SMD == PARTIALSM) {
    accumulator_fft = (double2 *)sharedmem;
    if (support_dsm)
      accumulator_fft += sizeof(double2) * (polynomial_size / 2);
  }

  Torus *block_lwe_array_in =
      &lwe_array_in[lwe_input_indexes[blockIdx.z] * (lwe_dimension + 1)];

  Torus *block_lut_vector = &lut_vector[lut_vector_indexes[blockIdx.z] *
                                        params::degree * (glwe_dimension + 1)];

  double2 *block_join_buffer =
      &join_buffer[blockIdx.z * level_count * (glwe_dimension + 1) *
                   params::degree / 2];

  // Put "b" in [0, 2N[
  Torus b_hat = 0;
  rescale_torus_element(block_lwe_array_in[lwe_dimension], b_hat,
                        2 * params::degree);

  divide_by_monomial_negacyclic_inplace<Torus, params::opt,
                                        params::degree / params::opt>(
      accumulator, &block_lut_vector[blockIdx.y * params::degree], b_hat,
      false);

  ////////////////////////////////////////////////////////////
  for (int lwe_iteration = 0; lwe_iteration < lwe_dimension / grouping_factor;
       lwe_iteration++) {

    uint32_t keybundle_size_per_input = level_count * (glwe_dimension + 1) *
                                        (glwe_dimension + 1) *
                                        (polynomial_size / 2);

    double2 *keybundle = keybundle_array +
                         // select the input
                         blockIdx.z * keybundle_size_per_input;
    auto keybundle_out =
        get_ith_mask_kth_block(keybundle, 0, blockIdx.y, blockIdx.x,
                               polynomial_size, glwe_dimension, level_count);

    if(threadIdx.x == 0)
        compute_keybundle<Torus, params, FULLSM><<<glwe_dimension+1, blockDim, sizeof(Torus) * polynomial_size>>>(
            keybundle_out, block_lwe_array_in, bootstrapping_key, lwe_dimension,
            glwe_dimension, polynomial_size, grouping_factor, base_log, level_count,
            blockIdx.y, blockIdx.x, lwe_iteration, NULL, sizeof(Torus) * polynomial_size);

    cluster.sync();

    // Perform a rounding to increase the accuracy of the
    // bootstrapped ciphertext
    round_to_closest_multiple_inplace<Torus, params::opt,
                                      params::degree / params::opt>(
        accumulator, base_log, level_count);

    // Decompose the accumulator. Each block gets one level of the
    // decomposition, for the mask and the body (so block 0 will have the
    // accumulator decomposed at level 0, 1 at 1, etc.)
    GadgetMatrix<Torus, params> gadget_acc(base_log, level_count, accumulator);
    gadget_acc.decompose_and_compress_level(accumulator_fft, blockIdx.x);

    // We are using the same memory space for accumulator_fft and
    // accumulator_rotated, so we need to synchronize here to make sure they
    // don't modify the same memory space at the same time
    synchronize_threads_in_block();

    // Perform G^-1(ACC) * GGSW -> GLWE
    mul_ggsw_glwe<Torus, cluster_group, params>(
        accumulator, accumulator_fft, block_join_buffer, keybundle,
        polynomial_size, glwe_dimension, level_count, 0, cluster, support_dsm);

    synchronize_threads_in_block();
  }

  auto block_lwe_array_out =
      &lwe_array_out[lwe_output_indexes[blockIdx.z] *
                         (glwe_dimension * polynomial_size + 1) +
                     blockIdx.y * polynomial_size];

  if (blockIdx.y < glwe_dimension) {
    // Perform a sample extract. At this point, all blocks have the result,
    // but we do the computation at block 0 to avoid waiting for extra blocks,
    // in case they're not synchronized
    sample_extract_mask<Torus, params>(block_lwe_array_out, accumulator);
  } else {
    sample_extract_body<Torus, params>(block_lwe_array_out, accumulator, 0);
  }
}

template <typename Torus>
__host__ __device__ uint64_t
get_buffer_size_sm_dsm_plus_tbc_multibit_programmable_bootstrap(
    uint32_t polynomial_size) {
  return sizeof(Torus) * polynomial_size; // distributed shared memory
}

template <typename Torus>
__host__ __device__ uint64_t
get_buffer_size_partial_sm_tbc_multibit_programmable_bootstrap(
    uint32_t polynomial_size) {
  return sizeof(double2) * polynomial_size / 2; // accumulator
}

template <typename Torus>
__host__ __device__ uint64_t
get_buffer_size_full_sm_tbc_multibit_programmable_bootstrap(
    uint32_t polynomial_size) {
  return sizeof(Torus) * polynomial_size +     // accumulator
         sizeof(double2) * polynomial_size / 2;
}

template <typename Torus, typename STorus, typename params>
__host__ void scratch_tbc_multi_bit_programmable_bootstrap(
    cudaStream_t stream, uint32_t gpu_index,
    pbs_buffer<uint64_t, MULTI_BIT> **buffer, uint32_t lwe_dimension,
    uint32_t glwe_dimension, uint32_t polynomial_size, uint32_t level_count,
    uint32_t input_lwe_ciphertext_count, uint32_t grouping_factor,
    uint32_t max_shared_memory, bool allocate_gpu_memory,
    uint32_t lwe_chunk_size = 0) {

  cudaSetDevice(gpu_index);

  bool supports_dsm =
      supports_distributed_shared_memory_on_multibit_programmable_bootstrap<
          Torus>(polynomial_size, max_shared_memory);

  uint64_t full_sm_tbc =
      get_buffer_size_full_sm_tbc_multibit_programmable_bootstrap<Torus>(
          polynomial_size);
  uint64_t partial_sm_tbc =
      get_buffer_size_partial_sm_tbc_multibit_programmable_bootstrap<Torus>(
          polynomial_size);
  uint64_t minimum_sm_tbc = 0;
  if (supports_dsm)
    minimum_sm_tbc =
        get_buffer_size_sm_dsm_plus_tbc_multibit_programmable_bootstrap<Torus>(
            polynomial_size);

  // We know max_shared_memory >= minimum_sm_tbc otherwise we wouldn't be here
  if (max_shared_memory < partial_sm_tbc + minimum_sm_tbc) {
    // We only have the minium amount of shared memory available
    check_cuda_error(cudaFuncSetAttribute(
        device_multi_bit_programmable_bootstrap<Torus, params, NOSM>,
        cudaFuncAttributeMaxDynamicSharedMemorySize, minimum_sm_tbc));
    cudaFuncSetCacheConfig(
        device_multi_bit_programmable_bootstrap<Torus, params, NOSM>,
        cudaFuncCachePreferShared);
    check_cuda_error(cudaGetLastError());
  } else if (max_shared_memory < full_sm_tbc + minimum_sm_tbc) {
    // We can partially run on shared memory
    check_cuda_error(cudaFuncSetAttribute(
        device_multi_bit_programmable_bootstrap<Torus, params, PARTIALSM>,
        cudaFuncAttributeMaxDynamicSharedMemorySize,
        partial_sm_tbc + minimum_sm_tbc));
    cudaFuncSetCacheConfig(
        device_multi_bit_programmable_bootstrap<Torus, params, PARTIALSM>,
        cudaFuncCachePreferShared);
    check_cuda_error(cudaGetLastError());
  } else {
    // We can fully exploit shared memory
    check_cuda_error(cudaFuncSetAttribute(
        device_multi_bit_programmable_bootstrap<Torus, params, FULLSM>,
        cudaFuncAttributeMaxDynamicSharedMemorySize,
        full_sm_tbc + minimum_sm_tbc));
    cudaFuncSetCacheConfig(
        device_multi_bit_programmable_bootstrap<Torus, params, FULLSM>,
        cudaFuncCachePreferShared);
    check_cuda_error(cudaGetLastError());
  }
  if (!lwe_chunk_size)
    lwe_chunk_size =
        get_lwe_chunk_size<Torus, params>(gpu_index, input_lwe_ciphertext_count,
                                          polynomial_size, max_shared_memory);
  *buffer = new pbs_buffer<uint64_t, MULTI_BIT>(
      stream, gpu_index, glwe_dimension, polynomial_size, level_count,
      input_lwe_ciphertext_count, lwe_chunk_size, PBS_VARIANT::TBC,
      allocate_gpu_memory);
}

template <typename Torus, typename STorus, class params>
__host__ void host_tbc_multi_bit_programmable_bootstrap(
    cudaStream_t stream, uint32_t gpu_index, Torus *lwe_array_out,
    Torus *lwe_output_indexes, Torus *lut_vector, Torus *lut_vector_indexes,
    Torus *lwe_array_in, Torus *lwe_input_indexes, uint64_t *bootstrapping_key,
    pbs_buffer<Torus, MULTI_BIT> *buffer, uint32_t glwe_dimension,
    uint32_t lwe_dimension, uint32_t polynomial_size, uint32_t grouping_factor,
    uint32_t base_log, uint32_t level_count, uint32_t num_samples,
    uint32_t num_luts, uint32_t lwe_idx, uint32_t max_shared_memory,
    uint32_t lwe_chunk_size = 0) {
  cudaSetDevice(gpu_index);

  auto supports_dsm =
      supports_distributed_shared_memory_on_multibit_programmable_bootstrap<
          Torus>(polynomial_size, max_shared_memory);

  uint64_t full_sm_tbc =
      get_buffer_size_full_sm_tbc_multibit_programmable_bootstrap<Torus>(
          polynomial_size);
  uint64_t partial_sm_tbc =
      get_buffer_size_partial_sm_tbc_multibit_programmable_bootstrap<Torus>(
          polynomial_size);
  uint64_t minimum_sm_tbc = 0;
  if (supports_dsm)
    minimum_sm_tbc =
        get_buffer_size_sm_dsm_plus_tbc_multibit_programmable_bootstrap<Torus>(
            polynomial_size);

  uint64_t full_dm = full_sm_tbc;
  uint64_t partial_dm = full_sm_tbc - partial_sm_tbc;

  auto d_mem = buffer->d_mem_acc_tbc;
  auto buffer_fft = buffer->global_accumulator_fft;
  auto keybundle_buffer = buffer->keybundle_fft;

  dim3 grid(level_count, glwe_dimension + 1, num_samples);
  dim3 thds(polynomial_size / params::opt, 1, 1);

  cudaLaunchConfig_t config = {0};
  // The grid dimension is not affected by cluster launch, and is still
  // enumerated using number of blocks. The grid dimension should be a multiple
  // of cluster size.
  config.gridDim = grid;
  config.blockDim = thds;

  cudaLaunchAttribute attribute[1];
  attribute[0].id = cudaLaunchAttributeClusterDimension;
  attribute[0].val.clusterDim.x = level_count; // Cluster size in X-dimension
  attribute[0].val.clusterDim.y = (glwe_dimension + 1);
  attribute[0].val.clusterDim.z = 1;
  config.attrs = attribute;
  config.numAttrs = 1;
  config.stream = stream;

  if (max_shared_memory < partial_dm + minimum_sm_tbc) {
    config.dynamicSmemBytes = minimum_sm_tbc;
    check_cuda_error(cudaLaunchKernelEx(
        &config, device_multi_bit_programmable_bootstrap<Torus, params, NOSM>,
        lwe_array_out, lwe_output_indexes, lut_vector, lut_vector_indexes,
        lwe_array_in, lwe_input_indexes, bootstrapping_key, keybundle_buffer,
        buffer_fft, lwe_dimension, glwe_dimension, polynomial_size,
        grouping_factor, base_log, level_count, d_mem, full_dm, supports_dsm));
  } else if (max_shared_memory < full_dm + minimum_sm_tbc) {
    config.dynamicSmemBytes = partial_sm_tbc + minimum_sm_tbc;
    check_cuda_error(cudaLaunchKernelEx(
        &config,
        device_multi_bit_programmable_bootstrap<Torus, params, PARTIALSM>,
        lwe_array_out, lwe_output_indexes, lut_vector, lut_vector_indexes,
        lwe_array_in, lwe_input_indexes, bootstrapping_key, keybundle_buffer,
        buffer_fft, lwe_dimension, glwe_dimension, polynomial_size,
        grouping_factor, base_log, level_count, d_mem, partial_dm,
        supports_dsm));
  } else {
    config.dynamicSmemBytes = full_sm_tbc + minimum_sm_tbc;
    check_cuda_error(cudaLaunchKernelEx(
        &config, device_multi_bit_programmable_bootstrap<Torus, params, FULLSM>,
        lwe_array_out, lwe_output_indexes, lut_vector, lut_vector_indexes,
        lwe_array_in, lwe_input_indexes, bootstrapping_key, keybundle_buffer,
        buffer_fft, lwe_dimension, glwe_dimension, polynomial_size,
        grouping_factor, base_log, level_count, d_mem, 0, supports_dsm));
  }
}

template <typename Torus>
__host__ bool
supports_distributed_shared_memory_on_multibit_programmable_bootstrap(
    uint32_t polynomial_size, uint32_t max_shared_memory) {
  uint64_t minimum_sm =
      get_buffer_size_sm_dsm_plus_tbc_multibit_programmable_bootstrap<Torus>(
          polynomial_size);

  if (max_shared_memory <= minimum_sm) {
    // If we cannot store a single polynomial in a block shared memory we
    // cannot use TBC
    return false;
  } else {
    return cuda_check_support_thread_block_clusters();
  }
}

template <typename Torus, class params>
__host__ bool supports_thread_block_clusters_on_multibit_programmable_bootstrap(
    uint32_t num_samples, uint32_t glwe_dimension, uint32_t polynomial_size,
    uint32_t level_count, uint32_t max_shared_memory) {

  if (!cuda_check_support_thread_block_clusters())
    return false;

  uint64_t full_sm_tbc =
      get_buffer_size_full_sm_tbc_multibit_programmable_bootstrap<Torus>(
          polynomial_size);
  uint64_t partial_sm_tbc =
      get_buffer_size_partial_sm_tbc_multibit_programmable_bootstrap<Torus>(
          polynomial_size);
  uint64_t minimum_sm_tbc = 0;
  if (supports_distributed_shared_memory_on_multibit_programmable_bootstrap<
          Torus>(polynomial_size, max_shared_memory))
    minimum_sm_tbc =
        get_buffer_size_sm_dsm_plus_tbc_multibit_programmable_bootstrap<Torus>(
            polynomial_size);

  int cluster_size;

  dim3 grid(level_count, glwe_dimension + 1, num_samples);
  dim3 thds(polynomial_size / params::opt, 1, 1);

  cudaLaunchConfig_t config = {0};
  // The grid dimension is not affected by cluster launch, and is still
  // enumerated using number of blocks. The grid dimension should be a multiple
  // of cluster size.
  config.gridDim = grid;
  config.blockDim = thds;
  config.numAttrs = 0;

  if (max_shared_memory < partial_sm_tbc + minimum_sm_tbc) {
    check_cuda_error(cudaFuncSetAttribute(
        device_multi_bit_programmable_bootstrap<Torus, params, NOSM>,
        cudaFuncAttributeNonPortableClusterSizeAllowed, true));
    check_cuda_error(cudaOccupancyMaxPotentialClusterSize(
        &cluster_size,
        device_multi_bit_programmable_bootstrap<Torus, params, NOSM>, &config));
  } else if (max_shared_memory < full_sm_tbc + minimum_sm_tbc) {
    check_cuda_error(cudaFuncSetAttribute(
        device_multi_bit_programmable_bootstrap<Torus, params, PARTIALSM>,
        cudaFuncAttributeNonPortableClusterSizeAllowed, true));
    check_cuda_error(cudaOccupancyMaxPotentialClusterSize(
        &cluster_size,
        device_multi_bit_programmable_bootstrap<Torus, params, PARTIALSM>,
        &config));
  } else {
    check_cuda_error(cudaFuncSetAttribute(
        device_multi_bit_programmable_bootstrap<Torus, params, FULLSM>,
        cudaFuncAttributeNonPortableClusterSizeAllowed, true));
    check_cuda_error(cudaOccupancyMaxPotentialClusterSize(
        &cluster_size,
        device_multi_bit_programmable_bootstrap<Torus, params, FULLSM>,
        &config));
  }

  return cluster_size >= level_count * (glwe_dimension + 1);
}

template __host__ bool
supports_distributed_shared_memory_on_multibit_programmable_bootstrap<uint64_t>(
    uint32_t polynomial_size, uint32_t max_shared_memory);
#endif // FASTMULTIBIT_PBS_H
