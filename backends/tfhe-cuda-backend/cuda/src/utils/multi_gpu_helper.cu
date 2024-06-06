#include "device.h"
#include "helper.h"
#include <mutex>

bool p2p_enabled = false;

int cuda_setup_multi_gpu() {
  int num_gpus = cuda_get_number_of_gpus();
  if (num_gpus == 0)
    PANIC("GPU error: the number of GPUs should be > 0.")
  int num_used_gpus = 1;
  if (num_gpus > 1) {
    if (!p2p_enabled) {
      p2p_enabled = true;
      int has_peer_access_to_device_0;
      for (int i = 1; i < num_gpus; i++) {
        check_cuda_error(
            cudaDeviceCanAccessPeer(&has_peer_access_to_device_0, i, 0));
        if (has_peer_access_to_device_0) {
          cudaMemPool_t mempool;
          cudaMemAccessDesc desc = {};
          // Enable P2P Access and mempool access
          check_cuda_error(cudaSetDevice(i));
          check_cuda_error_ignore_specific(cudaDeviceEnablePeerAccess(0, 0),
                                           cudaErrorPeerAccessAlreadyEnabled);

          check_cuda_error(cudaDeviceGetDefaultMemPool(&mempool, 0));
          desc.location.type = cudaMemLocationTypeDevice;
          desc.location.id = i;
          desc.flags = cudaMemAccessFlagsProtReadWrite;
          check_cuda_error(
              cudaMemPoolSetAccess(mempool, &desc, 1 /* numDescs */));
          num_used_gpus += 1;
        } else {
          break;
        }
      }
    } else {
      int has_peer_access_to_device_0;
      for (int i = 1; i < num_gpus; i++) {
        check_cuda_error(
            cudaDeviceCanAccessPeer(&has_peer_access_to_device_0, i, 0));
        if (has_peer_access_to_device_0) {
          num_used_gpus += 1;
        } else {
          break;
        }
      }
    }
  }
  printf("num used gpus: %d, p2p enabled: %d", num_used_gpus, p2p_enabled);
  return num_used_gpus;
}

void cuda_cleanup_multi_gpu() {
  int num_gpus = cuda_get_number_of_gpus();
  if (num_gpus == 0)
    PANIC("GPU error: the number of GPUs should be > 0.")
  if (p2p_enabled) {
    p2p_enabled = false;
    if (num_gpus > 1) {
      int has_peer_access_to_device_0;
      for (int i = 1; i < num_gpus; i++) {
        check_cuda_error(
            cudaDeviceCanAccessPeer(&has_peer_access_to_device_0, i, 0));
        if (has_peer_access_to_device_0) {
          //// Disable access to memory pool
          cudaMemPool_t mempool;
          cudaMemAccessDesc desc = {};
          cudaDeviceGetDefaultMemPool(&mempool, 0);
          desc.location.type = cudaMemLocationTypeDevice;
          desc.location.id = i;
          desc.flags = cudaMemAccessFlagsProtNone;
          cudaMemPoolSetAccess(mempool, &desc, 1 /* numDescs */);
          //  Disable P2P Access
          cudaSetDevice(i);
          cudaDeviceDisablePeerAccess(0);
        } else {
          break;
        }
      }
    }
  }
}

int get_active_gpu_count(int num_inputs, int gpu_count) {
  int active_gpu_count = gpu_count;
  if (gpu_count > num_inputs) {
    active_gpu_count = num_inputs;
  }
  return active_gpu_count;
}

int get_gpu_offset(int total_num_inputs, int gpu_index, int gpu_count) {
  int gpu_offset = 0;
  for (uint i = 0; i < gpu_index; i++)
    gpu_offset += get_num_inputs_on_gpu(total_num_inputs, i, gpu_count);
  return gpu_offset;
}

int get_num_inputs_on_gpu(int total_num_inputs, int gpu_index, int gpu_count) {

  int num_inputs = 0;
  // If there are fewer inputs than GPUs, not all GPUs are active and GPU 0
  // handles everything
  if (gpu_count > total_num_inputs) {
    if (gpu_index < total_num_inputs) {
      num_inputs = 1;
    }
  } else {
    // If there are more inputs than GPUs, all GPUs are active and compute over
    // a chunk of the total inputs. The chunk size is smaller on the last GPUs.
    int small_input_num, large_input_num, cutoff;
    if (total_num_inputs % gpu_count == 0) {
      small_input_num = total_num_inputs / gpu_count;
      large_input_num = small_input_num;
      cutoff = 0;
    } else {
      int y = ceil((double)total_num_inputs / (double)gpu_count) * gpu_count -
              total_num_inputs;
      cutoff = gpu_count - y;
      small_input_num = total_num_inputs / gpu_count;
      large_input_num = (int)ceil((double)total_num_inputs / (double)gpu_count);
    }
    if (gpu_index < cutoff)
      num_inputs = large_input_num;
    else
      num_inputs = small_input_num;
  }
  return num_inputs;
}
