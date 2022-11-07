use crate::c_api::utils::*;
use std::os::raw::c_int;

use crate::shortint::ciphertext::Degree;

use super::{ShortintCiphertext, ShortintServerKey};

// This is the accepted way to declare a pointer to a C function/callback in cbindgen
pub type AccumulatorCallback = Option<extern "C" fn(u64) -> u64>;

pub struct ShortintPBSAccumulator(
    pub(in crate::c_api) crate::core_crypto::prelude::GlweCiphertext64,
    Degree,
);

#[no_mangle]
pub unsafe extern "C" fn shortints_shortints_server_key_generate_pbs_accumulator(
    server_key: *const ShortintServerKey,
    accumulator_callback: AccumulatorCallback,
    result: *mut *mut ShortintPBSAccumulator,
) -> c_int {
    catch_panic(|| {
        check_ptr_is_non_null_and_aligned(result).unwrap();

        // First fill the result with a null ptr so that if we fail and the return code is not
        // checked, then any access to the result pointer will segfault (mimics malloc on failure)
        *result = std::ptr::null_mut();

        let accumulator_callback = accumulator_callback.unwrap();

        let server_key = get_ref_checked(server_key).unwrap();

        // Closure is required as extern "C" fn does not implement the Fn trait
        #[allow(clippy::redundant_closure)]
        let heap_allocated_accumulator = Box::new(ShortintPBSAccumulator(
            server_key
                .0
                .generate_accumulator(|x: u64| accumulator_callback(x)),
            Degree(
                (0u64..(server_key.0.carry_modulus.0 as u64
                    * server_key.0.message_modulus.0 as u64))
                    .into_iter()
                    .map(|x: u64| accumulator_callback(x))
                    .fold(std::u64::MIN, |a, b| a.max(b)) as usize,
            ),
        ));

        *result = Box::into_raw(heap_allocated_accumulator);
    })
}

#[no_mangle]
pub unsafe extern "C" fn shortints_server_key_programmable_bootstrap(
    server_key: *const ShortintServerKey,
    accumulator: *const ShortintPBSAccumulator,
    ct_in: *const ShortintCiphertext,
    result: *mut *mut ShortintCiphertext,
) -> c_int {
    catch_panic(|| {
        check_ptr_is_non_null_and_aligned(result).unwrap();

        // First fill the result with a null ptr so that if we fail and the return code is not
        // checked, then any access to the result pointer will segfault (mimics malloc on failure)
        *result = std::ptr::null_mut();

        let server_key = get_ref_checked(server_key).unwrap();
        let accumulator = get_ref_checked(accumulator).unwrap();
        let ct_in = get_ref_checked(ct_in).unwrap();

        let mut heap_allocated_result = Box::new(ShortintCiphertext(
            server_key
                .0
                .keyswitch_programmable_bootstrap(&ct_in.0, &accumulator.0),
        ));

        heap_allocated_result.as_mut().0.degree = accumulator.1;

        *result = Box::into_raw(heap_allocated_result);
    })
}

#[no_mangle]
pub unsafe extern "C" fn shortints_server_key_programmable_bootstrap_assign(
    server_key: *const ShortintServerKey,
    accumulator: *const ShortintPBSAccumulator,
    ct_in_and_result: *mut ShortintCiphertext,
) -> c_int {
    catch_panic(|| {
        let server_key = get_ref_checked(server_key).unwrap();
        let accumulator = get_ref_checked(accumulator).unwrap();
        let ct_in_and_result = get_mut_checked(ct_in_and_result).unwrap();

        server_key
            .0
            .keyswitch_programmable_bootstrap_assign(&mut ct_in_and_result.0, &accumulator.0);

        ct_in_and_result.0.degree = accumulator.1;
    })
}
