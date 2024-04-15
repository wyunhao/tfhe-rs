#ifndef TFHE_RS_DIV_REM_CUH
#define TFHE_RS_DIV_REM_CUH

#include "crypto/keyswitch.cuh"
#include "device.h"
#include "integer.h"
#include "integer/comparison.cuh"
#include "integer/integer.cuh"
#include "integer/negation.cuh"
#include "integer/scalar_shifts.cuh"
#include "linear_algebra.h"
#include "programmable_bootstrap.h"
#include "utils/helper.cuh"
#include "utils/kernel_dimensions.cuh"
#include <fstream>
#include <iostream>
#include <omp.h>
#include <sstream>
#include <string>
#include <vector>

template <typename Torus> struct ciphertext_list {
  Torus *data;
  size_t max_blocks;
  size_t len;
  int_radix_params params;

  size_t big_lwe_size;
  size_t radix_size;
  size_t big_lwe_size_bytes;
  size_t radix_size_bytes;
  size_t big_lwe_dimension;

  ciphertext_list(int_radix_params params, size_t max_blocks,
                  cuda_stream_t *stream)
      : params(params), max_blocks(max_blocks) {
    big_lwe_size = params.big_lwe_dimension + 1;
    big_lwe_size_bytes = big_lwe_size * sizeof(Torus);
    radix_size = max_blocks * big_lwe_size;
    radix_size_bytes = radix_size * sizeof(Torus);
    big_lwe_dimension = params.big_lwe_dimension;
    data = (Torus *)cuda_malloc_async(radix_size_bytes, stream);
  }

  void clone_from(Torus *src, size_t start_block, size_t finish_block,
                  cuda_stream_t *stream) {
    len = finish_block - start_block + 1;

    cuda_memcpy_async_gpu_to_gpu(data, &src[start_block * big_lwe_size],
                                 len * big_lwe_size_bytes, stream);
  }

  void clone_from(ciphertext_list src, size_t start_block, size_t finish_block,
                  cuda_stream_t *stream) {
    clone_from(src.data, start_block, finish_block, stream);
  }

  void assign_zero(size_t start_block, size_t finish_block,
                   cuda_stream_t *stream) {
    auto size = finish_block - start_block + 1;
    cuda_memset_async(&data[start_block * big_lwe_size], 0,
                      size * big_lwe_size_bytes, stream);
  }

  Torus *last_block() { return &data[(len - 1) * big_lwe_size]; }
  Torus *first_block() { return data; }

  bool is_empty() { return len == 0; }

  void pop() {
    if (len > 0)
      len--;
    else
      assert(len > 0);
  }

  void insert(size_t ind, Torus *ciphertext_block, cuda_stream_t *stream) {
    assert(ind <= len);
    assert(len < max_blocks);

    size_t insert_offset = ind * big_lwe_size;

    for (size_t i = len; i > ind; i--) {
      Torus *src = &data[(i - 1) * big_lwe_size];
      Torus *dst = &data[i * big_lwe_size];
      cuda_memcpy_async_gpu_to_gpu(dst, src, big_lwe_size_bytes, stream);
    }

    cuda_memcpy_async_gpu_to_gpu(&data[insert_offset], ciphertext_block,
                                 big_lwe_size_bytes, stream);
    len++;
  }

  void push(Torus *ciphertext_block, cuda_stream_t *stream) {
    assert(len < max_blocks);

    size_t offset = len * big_lwe_size;
    cuda_memcpy_async_gpu_to_gpu(&data[offset], ciphertext_block,
                                 big_lwe_size_bytes, stream);
    len++;
  }

  void print_blocks_body(const char *name) {
    for (int i = 0; i < len; i++) {
      print_debug(name, &data[i * big_lwe_size + big_lwe_dimension], 1);
    }
  }
};

template <typename Torus>
__host__ void scratch_cuda_integer_div_rem_kb(
    cuda_stream_t *stream, int_div_rem_memory<Torus> **mem_ptr,
    uint32_t num_blocks, int_radix_params params, bool allocate_gpu_memory) {

  cudaSetDevice(stream->gpu_index);
  *mem_ptr = new int_div_rem_memory<Torus>(stream, params, num_blocks,
                                           allocate_gpu_memory);
}

template <typename Torus, class params>
__host__ void host_integer_div_rem_kb(cuda_stream_t *stream, Torus *quotient,
                                      Torus *remainder, Torus *numerator,
                                      Torus *divisor, void *bsk, uint64_t *ksk,
                                      int_div_rem_memory<uint64_t> *mem_ptr,
                                      uint32_t num_blocks) {

  auto radix_params = mem_ptr->params;

  auto big_lwe_dimension = radix_params.big_lwe_dimension;
  auto big_lwe_size = big_lwe_dimension + 1;
  auto big_lwe_size_bytes = big_lwe_size * sizeof(Torus);
  auto radix_size_bytes = big_lwe_size_bytes * num_blocks;

  uint32_t message_modulus = radix_params.message_modulus;
  uint32_t num_bits_in_message = 31 - __builtin_clz(message_modulus);
  uint32_t total_bits = num_bits_in_message * num_blocks;

  // TODO move in scratch
  cuda_stream_t *sub_stream_1 = new cuda_stream_t(stream->gpu_index);
  cuda_stream_t *sub_stream_2 = new cuda_stream_t(stream->gpu_index);
  cuda_stream_t *sub_stream_3 = new cuda_stream_t(stream->gpu_index);
  cuda_stream_t *sub_stream_4 = new cuda_stream_t(stream->gpu_index);

  cudaEvent_t eventMain;
  cudaEvent_t event1, event2, event3, event4;
  cudaEventCreate(&eventMain);
  cudaEventCreate(&event1);
  cudaEventCreate(&event2);
  cudaEventCreate(&event3);
  cudaEventCreate(&event4);


  ciphertext_list<Torus> remainder1(radix_params, num_blocks, stream);
  ciphertext_list<Torus> remainder2(radix_params, num_blocks, stream);
  ciphertext_list<Torus> numerator_block_stack(radix_params, num_blocks,
                                               stream);
  ciphertext_list<Torus> numerator_block(radix_params, 1, stream);

  ciphertext_list<Torus> interesting_remainder1(radix_params, num_blocks,
                                                stream);
  ciphertext_list<Torus> interesting_remainder2(radix_params, num_blocks,
                                                stream);
  ciphertext_list<Torus> interesting_divisor(radix_params, num_blocks, stream);
  ciphertext_list<Torus> divisor_ms_blocks(radix_params, num_blocks, stream);

  ciphertext_list<Torus> merged_interesting_remainder(radix_params, num_blocks,
                                                      stream);

  ciphertext_list<Torus> cur_quotient(radix_params, num_blocks, stream);
  ciphertext_list<Torus> overflowed(radix_params, 1, stream);

  ciphertext_list<Torus> check_divisor_upper_blocks(radix_params, 1, stream);
  ciphertext_list<Torus> compare_with_zero_equality(radix_params, num_blocks,
                                                    stream);

  numerator_block_stack.clone_from(numerator, 0, num_blocks - 1, stream);

  remainder1.assign_zero(0, num_blocks - 1, stream);
  remainder2.assign_zero(0, num_blocks - 1, stream);

  // luts
  int_radix_lut<Torus> *merge_overflow_flags_luts =
      new int_radix_lut<Torus>(stream, radix_params, num_bits_in_message,
                               num_bits_in_message * num_blocks, true);
  int_radix_lut<Torus> *masking_lut =
      new int_radix_lut<Torus>(stream, radix_params, 1, num_blocks, true);

  uint32_t numerator_block_stack_size = num_blocks;
  uint32_t interesting_remainder1_size = 0;
  for (int i = 0; i < num_bits_in_message; i++) {
    auto lut_f_bit = [i](Torus x, Torus y) -> Torus {
      return (x == 0 && y == 0) << i;
    };
    auto cur_lut = merge_overflow_flags_luts->get_lut(i);
    generate_device_accumulator_bivariate<Torus>(
        stream, cur_lut, radix_params.glwe_dimension,
        radix_params.polynomial_size, radix_params.message_modulus,
        radix_params.carry_modulus, lut_f_bit);
  }

  // end of move in scratch

  { // debug

    for (int i = 0; i < num_bits_in_message; i++) {
      auto cur_lut = merge_overflow_flags_luts->get_lut(i);
      print_debug("cuda_merge_lut:", cur_lut,
                  radix_params.polynomial_size *
                      (radix_params.glwe_dimension + 1));
    }

    for (int i = 0; i < num_blocks; i++) {
      print_debug(
          "numerator: ", &numerator[i * big_lwe_size + big_lwe_dimension], 1);
    }

    for (int i = 0; i < num_blocks; i++) {
      print_debug("divisor: ", &divisor[i * big_lwe_size + big_lwe_dimension],
                  1);
    }
  }

  int iter = 0; // debug
  for (int i = total_bits - 1; i >= 0; i--) {
    { // debug
      printf("cuda i = %u\n", i);
    }
    uint32_t block_of_bit = i / num_bits_in_message;
    uint32_t pos_in_block = i % num_bits_in_message;

    uint32_t msb_bit_set = total_bits - 1 - i;

    uint32_t last_non_trivial_block = msb_bit_set / num_bits_in_message;

    // Index to the first block of the remainder that is fully trivial 0
    // and all blocks after it are also trivial zeros
    // This number is in range 1..=num_bocks -1
    uint32_t first_trivial_block = last_non_trivial_block + 1;

    interesting_remainder1.clone_from(remainder1, 0, last_non_trivial_block,
                                      stream);
    interesting_remainder2.clone_from(remainder2, 0, last_non_trivial_block,
                                      stream);
    interesting_divisor.clone_from(divisor, 0, last_non_trivial_block, stream);
    divisor_ms_blocks.clone_from(divisor,
                                 (msb_bit_set + 1) / num_bits_in_message,
                                 num_blocks - 1, stream);

    //    { // debug
    //
    //      for (int i = 0; i < num_blocks; i++) {
    //        print_debug("numerator: ", &numerator[i * big_lwe_size],
    //        big_lwe_size);
    //      }
    //
    //    }

    interesting_remainder1_size = last_non_trivial_block + 1;
    // We split the divisor at a block position, when in reality the split
    // should be at a bit position meaning that potentially (depending on
    // msb_bit_set) the split versions share some bits they should not. So we do
    // one PBS on the last block of the interesting_divisor, and first block of
    // divisor_ms_blocks to trim out bits which should not be there

    // TODO following 3 apply_lookup_table can be called in one batch

    { // debug
      //      printf("initialize mid buffers iter %u\n", iter);
      printf("cuda_last_non_trivial_block %u\n", last_non_trivial_block);
      printf("cuda_(msb_bit_set + 1) / num_bits_in_message %u\n",
             (msb_bit_set + 1) / num_bits_in_message);

      interesting_remainder1.print_blocks_body("cuda_interesting_remainder1");
      interesting_remainder2.print_blocks_body("cuda_interesting_remainder2");
      interesting_divisor.print_blocks_body("cuda_interesting_divisor");
      divisor_ms_blocks.print_blocks_body("cuda_divisor_ms_blocks");
    }

    auto trim_last_interesting_divisor_bits = [&](cuda_stream_t *stream) {
      if ((msb_bit_set + 1) % num_bits_in_message == 0) {
        return;
      }
      // The last block of the interesting part of the remainder
      // can contain bits which we should not account for
      // we have to zero them out.

      // Where the msb is set in the block
      uint32_t pos_in_block = msb_bit_set % num_bits_in_message;

      // e.g 2 bits in message:
      // if pos_in_block is 0, then we want to keep only first bit (right
      // shift
      // mask by 1) if pos_in_block is 1, then we want to keep the two
      // bits
      // (right shift mask by 0)
      uint32_t shift_amount = num_bits_in_message - (pos_in_block + 1);

      // Create mask of 1s on the message part, 0s in the carries
      uint32_t full_message_mask = message_modulus - 1;

      // Shift the mask so that we will only keep bits we should
      uint32_t shifted_mask = full_message_mask >> shift_amount;

      // TODO move in scratch
      std::function<Torus(Torus)> lut_f_masking;
      lut_f_masking = [shifted_mask](Torus x) -> Torus {
        return x & shifted_mask;
      };
      generate_device_accumulator<Torus>(
          stream, masking_lut->lut, radix_params.glwe_dimension,
          radix_params.polynomial_size, radix_params.message_modulus,
          radix_params.carry_modulus, lut_f_masking);

      // end of move in scratch

      integer_radix_apply_univariate_lookup_table_kb(
          stream, interesting_divisor.last_block(),
          interesting_divisor.last_block(), bsk, ksk, 1, masking_lut);

      cudaEventRecord(event1, stream->stream);
    }; // trim_last_interesting_divisor_bits

    auto trim_first_divisor_ms_bits = [&](cuda_stream_t *stream) {
      if (divisor_ms_blocks.is_empty() ||
          ((msb_bit_set + 1) % num_bits_in_message) == 0) {
        return;
      }
      // Where the msb is set in the block
      uint32_t pos_in_block = msb_bit_set % num_bits_in_message;

      // e.g 2 bits in message:
      // if pos_in_block is 0, then we want to discard the first bit (left shift
      // mask by 1) if pos_in_block is 1, then we want to discard the two bits
      // (left shift mask by 2) let shift_amount = num_bits_in_message -
      // pos_in_block
      uint32_t shift_amount = pos_in_block + 1;
      uint32_t full_message_mask = message_modulus - 1;
      uint32_t shifted_mask = full_message_mask << shift_amount;

      // Keep the mask within the range of message bits, so that
      // the estimated degree of the output is < msg_modulus
      shifted_mask = shifted_mask & full_message_mask;

      // TODO movie in scratch
      std::function<Torus(Torus)> lut_f_masking;
      lut_f_masking = [shifted_mask](Torus x) -> Torus {
        return x & shifted_mask;
      };
      generate_device_accumulator<Torus>(
          stream, masking_lut->lut, radix_params.glwe_dimension,
          radix_params.polynomial_size, radix_params.message_modulus,
          radix_params.carry_modulus, lut_f_masking);
      // end of move in scratch

      integer_radix_apply_univariate_lookup_table_kb(
          stream, divisor_ms_blocks.first_block(),
          divisor_ms_blocks.first_block(), bsk, ksk, 1, masking_lut);
      cudaEventRecord(event2, stream->stream);

    }; // trim_first_divisor_ms_bits

    // This does
    //  R := R << 1; R(0) := N(i)
    //
    // We could to that by left shifting, R by one, then unchecked_add the
    // correct numerator bit.
    //
    // However, to keep the remainder clean (noise wise), what we do is that we
    // put the remainder block from which we need to extract the bit, as the LSB
    // of the Remainder, so that left shifting will pull the bit we need.
    auto left_shift_interesting_remainder1 = [&](cuda_stream_t *stream) {
      numerator_block.clone_from(numerator_block_stack,
                                 numerator_block_stack.len - 1,
                                 numerator_block_stack.len - 1, stream);
      numerator_block_stack.pop();
      interesting_remainder1.insert(0, numerator_block.first_block(), stream);
      host_integer_radix_logical_scalar_shift_kb_inplace(
          stream, interesting_remainder1.data, 1, mem_ptr->shift_mem, bsk, ksk,
          num_blocks);

      radix_blocks_rotate_left<<<num_blocks, 256, 0, stream->stream>>>(
          interesting_remainder1.data, interesting_remainder1.data, 1,
          num_blocks, big_lwe_size);

      numerator_block.clone_from(interesting_remainder1,
                                 interesting_remainder1.len - 1,
                                 interesting_remainder1.len - 1, stream);
      interesting_remainder1.pop();

      if (pos_in_block != 0) {
        // We have not yet extracted all the bits from this numerator
        // so, we put it back on the front so that it gets taken next iteration
        numerator_block_stack.push(numerator_block.first_block(), stream);
      }
      cudaEventRecord(event3, stream->stream);

    };  // left_shift_interesting_remainder1

    auto left_shift_interesting_remainder2 = [&](cuda_stream_t *stream) {
      host_integer_radix_logical_scalar_shift_kb_inplace(
          stream, interesting_remainder2.data, 1, mem_ptr->shift_mem, bsk, ksk,
          num_blocks);
      cudaEventRecord(event4, stream->stream);


    };  // left_shift_interesting_remainder2

    cudaEventRecord(eventMain, stream->stream);

    cudaStreamWaitEvent(sub_stream_1->stream, eventMain, 0);
    cudaStreamWaitEvent(sub_stream_2->stream, eventMain, 0);
    cudaStreamWaitEvent(sub_stream_3->stream, eventMain, 0);
    cudaStreamWaitEvent(sub_stream_4->stream, eventMain, 0);

    #pragma omp parallel sections
    {
      #pragma omp section
      {
        trim_last_interesting_divisor_bits(sub_stream_1);
      }
      #pragma omp section
      {
        trim_first_divisor_ms_bits(sub_stream_2);
      }
      #pragma omp section
      {
        left_shift_interesting_remainder1(sub_stream_3);
      }
      #pragma omp section
      {
        left_shift_interesting_remainder2(sub_stream_4);
      }
    }

    cudaStreamWaitEvent(stream->stream, event1, 0);
    cudaStreamWaitEvent(stream->stream, event2, 0);
    cudaStreamWaitEvent(stream->stream, event3, 0);
    cudaStreamWaitEvent(stream->stream, event4, 0);
    //
    //    // left_shift_interesting_remainder2
    //    host_integer_radix_logical_scalar_shift_kb_inplace(
    //        stream, interesting_remainder2, 1, mem_ptr->shift_mem, bsk, ksk,
    //        num_blocks);
    //
    //    cuda_memcpy_async_gpu_to_gpu(merged_interesting_remainder,
    //                                 interesting_remainder1, radix_size_bytes,
    //                                 stream);
    //
    //    host_addition(stream, merged_interesting_remainder,
    //                  merged_interesting_remainder, interesting_remainder2,
    //                  radix_params.big_lwe_dimension, num_blocks);
    //
    //    // TODO there is a way to parallelize following 3 calls
    //    // do_overflowing_sub
    //    //    host_integer_overflowing_sub_kb(stream,
    //    //                                    cur_quotient,
    //    //                                    overflowed,
    //    //                                    merged_interesting_remainder,
    //    //                                    interesting_divisor,
    //    //                                    bsk, ksk,
    //    mem_ptr->overflow_sub_mem,
    //    //                                    num_blocks);
    //
    //    // check_divisor_upper_blocks
    //    auto trivial_blocks = divisor_ms_blocks;
    //
    //    if (ms_start_index >= num_blocks - 1) {
    //      cuda_memset_async(check_divisor_upper_blocks, 0, big_lwe_size_bytes,
    //                        stream);
    //    } else {
    //      host_compare_with_zero_equality(
    //          stream, compare_with_zero_equality, trivial_blocks,
    //          mem_ptr->comparison_buffer, bsk, ksk, num_blocks,
    //          mem_ptr->comparison_buffer->eq_buffer->is_non_zero_lut);
    //
    //      is_at_least_one_comparisons_block_true(
    //          stream, check_divisor_upper_blocks, compare_with_zero_equality,
    //          mem_ptr->comparison_buffer, bsk, ksk, num_blocks);
    //    }
    //
    //    // Creates a cleaned version (noise wise) of the merged remainder
    //    // so that it can be safely used in bivariate PBSes
    //
    //    iter++; // debug
    //    break;
  }

  cuda_memcpy_async_gpu_to_gpu(quotient, cur_quotient.data, radix_size_bytes,
                               stream);
}

#endif // TFHE_RS_DIV_REM_CUH
