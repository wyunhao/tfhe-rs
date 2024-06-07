#ifndef HELPER_MULTI_GPU_H
#define HELPER_MULTI_GPU_H
#include <mutex>

extern std::mutex m;
extern bool p2p_enabled;

extern "C" {
int cuda_setup_multi_gpu();
}

int get_active_gpu_count(int num_inputs, int gpu_count);

int get_num_inputs_on_gpu(int total_num_inputs, int gpu_index, int gpu_count);

int get_gpu_offset(int total_num_inputs, int gpu_index, int gpu_count);

template <typename Torus>
void multi_gpu_dispatch(cudaStream_t *streams, uint32_t *gpu_indexes,
                        uint32_t gpu_count, std::vector<Torus *> &dest,
                        Torus *src, uint32_t num_inputs, uint32_t elements_per_input);

template <typename Torus>
void multi_gpu_gather(cudaStream_t *streams, uint32_t *gpu_indexes,
                      uint32_t gpu_count, Torus *dest, std::vector<Torus *> src,
                      uint32_t num_inputs, uint32_t elements_per_input);

template <typename Torus>
void multi_gpu_release(cudaStream_t *streams, uint32_t *gpu_indexes,
                       uint32_t gpu_count, std::vector<Torus *> vec,
                       uint32_t num_inputs);
#endif
