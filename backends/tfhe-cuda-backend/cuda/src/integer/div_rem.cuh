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

int ceil_div(int a, int b) { return (a + b - 1) / b; }
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
    len = max_blocks;
  }

  void copy_from(Torus *src, size_t start_block, size_t finish_block,
                 cuda_stream_t *stream) {
    cuda_memcpy_async_gpu_to_gpu(data, &src[start_block * big_lwe_size],
                                 len * big_lwe_size_bytes, stream);
  }
  void copy_from(ciphertext_list src, size_t start_block, size_t finish_block,
                 cuda_stream_t *stream) {
    copy_from(src.data, start_block, finish_block, stream);
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
  Torus *get_block(size_t index) {
    assert(index < len);
    return &data[index * big_lwe_size];
  }
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

  void fill_with_same_ciphertext(Torus *ciphertext, size_t number_of_blocks,
                                 cuda_stream_t *stream) {
    assert(number_of_blocks <= max_blocks);

    for (size_t i = 0; i < number_of_blocks; i++) {
      Torus *dest = &data[i * big_lwe_size];
      cuda_memcpy_async_gpu_to_gpu(dest, ciphertext, big_lwe_size_bytes,
                                   stream);
    }

    len = number_of_blocks;
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
  uint32_t carry_modulus = radix_params.carry_modulus;
  uint32_t num_bits_in_message = 31 - __builtin_clz(message_modulus);
  uint32_t total_bits = num_bits_in_message * num_blocks;

  // TODO move in scratch
  cuda_stream_t *sub_stream_1 = new cuda_stream_t(stream->gpu_index);
  cuda_stream_t *sub_stream_2 = new cuda_stream_t(stream->gpu_index);
  cuda_stream_t *sub_stream_3 = new cuda_stream_t(stream->gpu_index);
  cuda_stream_t *sub_stream_4 = new cuda_stream_t(stream->gpu_index);

  ciphertext_list<Torus> remainder1(radix_params, num_blocks, stream);
  ciphertext_list<Torus> remainder2(radix_params, num_blocks, stream);
  ciphertext_list<Torus> numerator_block_stack(radix_params, num_blocks,
                                               stream);
  ciphertext_list<Torus> numerator_block(radix_params, 1, stream);
  ciphertext_list<Torus> numerator_block2(radix_params, 1, stream);

  ciphertext_list<Torus> interesting_remainder1(radix_params, num_blocks + 1,
                                                stream);
  ciphertext_list<Torus> interesting_remainder2(radix_params, num_blocks,
                                                stream);
  ciphertext_list<Torus> interesting_divisor(radix_params, num_blocks, stream);
  ciphertext_list<Torus> divisor_ms_blocks(radix_params, num_blocks, stream);

  ciphertext_list<Torus> new_remainder(radix_params, num_blocks, stream);
  ciphertext_list<Torus> subtraction_overflowed(radix_params, 1, stream);
  ciphertext_list<Torus> did_not_overflow(radix_params, 1, stream);
  ciphertext_list<Torus> overflow_sum(radix_params, 1, stream);
  ciphertext_list<Torus> overflow_sum_radix(radix_params, num_blocks, stream);

  ciphertext_list<Torus> tmp1(radix_params, num_blocks, stream);
  ciphertext_list<Torus> at_least_one_upper_block_is_non_zero(radix_params, 1,
                                                              stream);
  ciphertext_list<Torus> cleaned_merged_interesting_remainder(
      radix_params, num_blocks, stream);

  numerator_block_stack.clone_from(numerator, 0, num_blocks - 1, stream);

  remainder1.assign_zero(0, num_blocks - 1, stream);
  remainder2.assign_zero(0, num_blocks - 1, stream);

  // luts
  int_radix_lut<Torus> **merge_overflow_flags_luts =
      new int_radix_lut<Torus> *[num_bits_in_message];
  int_radix_lut<Torus> *masking_lut =
      new int_radix_lut<Torus>(stream, radix_params, 1, num_blocks, true);
  int_radix_lut<Torus> *masking_lut2 =
      new int_radix_lut<Torus>(stream, radix_params, 1, num_blocks, true);
  int_radix_lut<Torus> *message_extract_lut =
      new int_radix_lut<Torus>(stream, radix_params, 1, num_blocks, true);
  int_radix_lut<Torus> *zero_out_if_overflow_did_not_happen =
      new int_radix_lut<Torus>(stream, radix_params, 1, num_blocks, true);
  int_radix_lut<Torus> *zero_out_if_overflow_happened =
      new int_radix_lut<Torus>(stream, radix_params, 1, num_blocks, true);

  uint32_t numerator_block_stack_size = num_blocks;
  uint32_t interesting_remainder1_size = 0;
  for (int i = 0; i < num_bits_in_message; i++) {
    auto lut_f_bit = [i](Torus x, Torus y) -> Torus {
      return (x == 0 && y == 0) << i;
    };

    merge_overflow_flags_luts[i] =
        new int_radix_lut<Torus>(stream, radix_params, 1, num_blocks, true);

    generate_device_accumulator_bivariate<Torus>(
        stream, merge_overflow_flags_luts[i]->lut, radix_params.glwe_dimension,
        radix_params.polynomial_size, radix_params.message_modulus,
        radix_params.carry_modulus, lut_f_bit);
  }

  // end of move in scratch

  { // debug

    for (int i = 0; i < num_bits_in_message; i++) {
      auto cur_lut = merge_overflow_flags_luts[i]->get_lut(0);
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
        { // debug
          printf("cuda trim_last_interesting_divisor_bits dabrunda\n");
        }
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

      printf("shifted_mask: %u \n", shifted_mask);
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
      { // debug
        print_debug("masking_lut#1", masking_lut->lut,
                    (radix_params.glwe_dimension + 1) *
                        radix_params.polynomial_size);
      }
      integer_radix_apply_univariate_lookup_table_kb(
          stream, interesting_divisor.last_block(),
          interesting_divisor.last_block(), bsk, ksk, 1, masking_lut);
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

      // TODO move in scratch
      std::function<Torus(Torus)> lut_f_masking;
      lut_f_masking = [shifted_mask](Torus x) -> Torus {
        return x & shifted_mask;
      };
      generate_device_accumulator<Torus>(
          stream, masking_lut2->lut, radix_params.glwe_dimension,
          radix_params.polynomial_size, radix_params.message_modulus,
          radix_params.carry_modulus, lut_f_masking);
      // end of move in scratch

      integer_radix_apply_univariate_lookup_table_kb(
          stream, divisor_ms_blocks.first_block(),
          divisor_ms_blocks.first_block(), bsk, ksk, 1, masking_lut2);
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
      { // debug
        numerator_block.print_blocks_body("numerator_block");
        interesting_remainder1.print_blocks_body("interesting_remainder1");
      }
      host_integer_radix_logical_scalar_shift_kb_inplace(
          stream, interesting_remainder1.data, 1, mem_ptr->shift_mem, bsk, ksk,
          interesting_remainder1.len);
      { // debug
        interesting_remainder1.print_blocks_body("interesting_remainder1");
        printf("interesting_remainder1.len: %u \n", interesting_remainder1.len);
      }

      radix_blocks_rotate_left<<<interesting_remainder1.len, 256, 0,
                                 stream->stream>>>(
          interesting_remainder1.data, interesting_remainder1.data, 1,
          interesting_remainder1.len, big_lwe_size);

      { // debug
        interesting_remainder1.print_blocks_body("interesting_remainder1");
      }
      numerator_block.clone_from(interesting_remainder1,
                                 interesting_remainder1.len - 1,
                                 interesting_remainder1.len - 1, stream);
      interesting_remainder1.pop();

      if (pos_in_block != 0) {
        // We have not yet extracted all the bits from this numerator
        // so, we put it back on the front so that it gets taken next iteration
        { // debug
          numerator_block.print_blocks_body("numerator_block");
        }
        numerator_block_stack.push(numerator_block.first_block(), stream);
      }
    }; // left_shift_interesting_remainder1

    auto left_shift_interesting_remainder2 = [&](cuda_stream_t *stream) {
      host_integer_radix_logical_scalar_shift_kb_inplace(
          stream, interesting_remainder2.data, 1, mem_ptr->shift_mem, bsk, ksk,
          num_blocks);
    }; // left_shift_interesting_remainder2

#pragma omp parallel sections
    {
#pragma omp section
      {
        // interesting_divisor
        trim_last_interesting_divisor_bits(sub_stream_1);
      }
#pragma omp section
      {
        // divisor_ms_blocks
        trim_first_divisor_ms_bits(sub_stream_2);
      }
#pragma omp section
      {
        // interesting_remainder1
        // numerator_block_stack
        left_shift_interesting_remainder1(sub_stream_3);
      }
#pragma omp section
      {
        // interesting_remainder2
        left_shift_interesting_remainder2(sub_stream_4);
      }
    }
    cuda_synchronize_stream(sub_stream_1);
    cuda_synchronize_stream(sub_stream_2);
    cuda_synchronize_stream(sub_stream_3);
    cuda_synchronize_stream(sub_stream_4);

    { // debug
      printf("cuda chunk #1-----------------\n");
      interesting_divisor.print_blocks_body("cuda_interesting_divisor");
      divisor_ms_blocks.print_blocks_body("cuda_divisor_ms_blocks");
      interesting_remainder1.print_blocks_body("cuda_interesting_remainder1");
      numerator_block_stack.print_blocks_body("cuda_numerator_block_stack");
      interesting_remainder2.print_blocks_body("cuda_interesting_remainder2");
    }

    // if interesting_remainder1 != 0 -> interesting_remainder2 == 0
    // if interesting_remainder1 == 0 -> interesting_remainder2 != 0
    // In practice interesting_remainder1 contains the numerator bit,
    // but in that position, interesting_remainder2 always has a 0
    auto &merged_interesting_remainder = interesting_remainder1;

    host_addition(stream, merged_interesting_remainder.data,
                  merged_interesting_remainder.data,
                  interesting_remainder2.data, radix_params.big_lwe_dimension,
                  num_blocks);

    // after create_clean_version_of_merged_remainder
    // `merged_interesting_remainder` will be reused as
    // `cleaned_merged_interesting_remainder`
    cleaned_merged_interesting_remainder.clone_from(
        merged_interesting_remainder, 0, merged_interesting_remainder.len - 1,
        stream);

    { // debug
      merged_interesting_remainder.print_blocks_body(
          "merged_interesting_remainder_after_add");
    }

    assert(merged_interesting_remainder.len == interesting_divisor.len);

    // `new_remainder` is not initialized yet, so need to set length
    new_remainder.len = merged_interesting_remainder.len;
    // fills:
    //  `new_remainder` - radix ciphertext
    //  `subtraction_overflowed` - single ciphertext
    auto do_overflowing_sub = [&](cuda_stream_t *stream) {
      host_integer_overflowing_sub_kb<Torus, params>(
          stream, new_remainder.data, subtraction_overflowed.data,
          merged_interesting_remainder.data, interesting_divisor.data, bsk, ksk,
          mem_ptr->overflow_sub_mem, merged_interesting_remainder.len);
    };

    // fills:
    //  `at_least_one_upper_block_is_non_zero` - single ciphertext
    auto check_divisor_upper_blocks = [&](cuda_stream_t *stream) {
      auto &trivial_blocks = divisor_ms_blocks;
      if (trivial_blocks.is_empty()) {
        cuda_memset_async(at_least_one_upper_block_is_non_zero.first_block(), 0,
                          big_lwe_size_bytes, stream);
      } else {

        printf("trivial_blocks.len: %u\n", trivial_blocks.len); // debug
        // We could call unchecked_scalar_ne
        // But we are in the special case where scalar == 0
        // So we can skip some stuff
        host_compare_with_zero_equality(
            stream, tmp1.data, trivial_blocks.data, mem_ptr->comparison_buffer,
            bsk, ksk, trivial_blocks.len,
            mem_ptr->comparison_buffer->eq_buffer->is_non_zero_lut);
        tmp1.len =
            ceil_div(trivial_blocks.len, message_modulus * carry_modulus - 1);

        { tmp1.print_blocks_body("tmp1"); }
        is_at_least_one_comparisons_block_true(
            stream, at_least_one_upper_block_is_non_zero.data, tmp1.data,
            mem_ptr->comparison_buffer, bsk, ksk, tmp1.len);
      }
    };

    // Creates a cleaned version (noise wise) of the merged remainder
    // so that it can be safely used in bivariate PBSes
    // fills:
    //  `cleaned_merged_interesting_remainder` - radix ciphertext
    auto create_clean_version_of_merged_remainder = [&](cuda_stream_t *stream) {
      auto lut_f_message_extract = [message_modulus](Torus x) -> Torus {
        return x % message_modulus;
      };
      auto cur_lut = message_extract_lut->get_lut(0);
      generate_device_accumulator<Torus>(
          stream, cur_lut, radix_params.glwe_dimension,
          radix_params.polynomial_size, radix_params.message_modulus,
          radix_params.carry_modulus, lut_f_message_extract);
      integer_radix_apply_univariate_lookup_table_kb(
          stream, cleaned_merged_interesting_remainder.data,
          cleaned_merged_interesting_remainder.data, bsk, ksk,
          cleaned_merged_interesting_remainder.len, message_extract_lut);
    };

    printf("new_remainder.len: %u\n", new_remainder.len);

    // phase 2
#pragma omp parallel sections
    {
#pragma omp section
      {
        // new_remainder
        // subtraction_overflowed
        do_overflowing_sub(sub_stream_1);
      }
#pragma omp section
      {
        // at_least_one_upper_block_is_non_zero
        check_divisor_upper_blocks(sub_stream_2);
      }
#pragma omp section
      {
        // cleaned_merged_interesting_remainder
        create_clean_version_of_merged_remainder(sub_stream_3);
      }
    }
    cuda_synchronize_stream(sub_stream_1);
    cuda_synchronize_stream(sub_stream_2);
    cuda_synchronize_stream(sub_stream_3);

    { // debug
      printf("new_remainder.len: %u\n", new_remainder.len);
      new_remainder.print_blocks_body("new_remainder");
      subtraction_overflowed.print_blocks_body("subtraction_overflowed");
      at_least_one_upper_block_is_non_zero.print_blocks_body(
          "at_least_one_upper_block_is_non_zero");
      cleaned_merged_interesting_remainder.print_blocks_body(
          "cleaned_merged_interesting_remainder");
    }

    // phase 3

    // Give name to closures to improve readability
    auto overflow_happened = [](uint64_t overflow_sum) {
      return overflow_sum != 0;
    };
    auto overflow_did_not_happen = [&overflow_happened](uint64_t overflow_sum) {
      return !overflow_happened(overflow_sum);
    };

    host_addition(stream, overflow_sum.data, subtraction_overflowed.data,
                  at_least_one_upper_block_is_non_zero.data,
                  radix_params.big_lwe_dimension, 1);

    { // debug
      overflow_sum.print_blocks_body("overflow_sum");
    }

    int factor = (i) ? 3 : 2;
    overflow_sum_radix.fill_with_same_ciphertext(
        overflow_sum.first_block(), cleaned_merged_interesting_remainder.len,
        stream);

    auto conditionally_zero_out_merged_interesting_remainder =
        [&](cuda_stream_t *stream) {
          auto cur_lut_f = [&](Torus block, Torus overflow_sum) -> Torus {
            if (overflow_did_not_happen(overflow_sum)) {
              return 0;
            } else {
              return block;
            }
          };
          auto cur_lut = zero_out_if_overflow_did_not_happen->lut;
          generate_device_accumulator_bivariate_with_factor<Torus>(
              stream, cur_lut, radix_params.glwe_dimension,
              radix_params.polynomial_size, radix_params.message_modulus,
              radix_params.carry_modulus, cur_lut_f, factor);

          integer_radix_apply_bivariate_lookup_table_kb<Torus>(
              stream, cleaned_merged_interesting_remainder.data,
              cleaned_merged_interesting_remainder.data,
              overflow_sum_radix.data, bsk, ksk,
              cleaned_merged_interesting_remainder.len,
              zero_out_if_overflow_did_not_happen);
        };

    auto conditionally_zero_out_merged_new_remainder =
        [&](cuda_stream_t *stream) {
          auto cur_lut_f = [&](Torus block, Torus overflow_sum) -> Torus {
            if (overflow_happened(overflow_sum)) {
              return 0;
            } else {
              return block;
            }
          };
          auto cur_lut = zero_out_if_overflow_happened->lut;
          generate_device_accumulator_bivariate_with_factor<Torus>(
              stream, cur_lut, radix_params.glwe_dimension,
              radix_params.polynomial_size, radix_params.message_modulus,
              radix_params.carry_modulus, cur_lut_f, factor);

          integer_radix_apply_bivariate_lookup_table_kb<Torus>(
              stream, new_remainder.data, new_remainder.data,
              overflow_sum_radix.data, bsk, ksk, new_remainder.len,
              zero_out_if_overflow_happened);
        };

    auto set_quotient_bit = [&](cuda_stream_t *stream) {
      integer_radix_apply_bivariate_lookup_table_kb<Torus>(
          stream, did_not_overflow.data, subtraction_overflowed.data,
          at_least_one_upper_block_is_non_zero.data, bsk, ksk, 1,
          merge_overflow_flags_luts[pos_in_block]);

      host_addition(stream, &quotient[block_of_bit * big_lwe_size],
                    &quotient[block_of_bit * big_lwe_size],
                    did_not_overflow.data, radix_params.big_lwe_dimension, 1);
    };

#pragma omp parallel sections
    {
#pragma omp section
      {
        // cleaned_merged_interesting_remainder
        conditionally_zero_out_merged_interesting_remainder(sub_stream_1);
      }
#pragma omp section
      {
        // new_remainder
        conditionally_zero_out_merged_new_remainder(sub_stream_2);
      }
#pragma omp section
      {
        // quotient
        set_quotient_bit(sub_stream_3);
      }
    }
    cuda_synchronize_stream(sub_stream_1);
    cuda_synchronize_stream(sub_stream_2);
    cuda_synchronize_stream(sub_stream_3);

    assert(first_trivial_block - 1 == cleaned_merged_interesting_remainder.len);
    assert(first_trivial_block - 1 == new_remainder.len);

    remainder1.copy_from(cleaned_merged_interesting_remainder, 0,
                         first_trivial_block - 1, stream);
    remainder2.copy_from(new_remainder, 0, first_trivial_block - 1, stream);
  }

  cuda_memcpy_async_gpu_to_gpu(quotient, new_remainder.data, radix_size_bytes,
                               stream);
}

#endif // TFHE_RS_DIV_REM_CUH
