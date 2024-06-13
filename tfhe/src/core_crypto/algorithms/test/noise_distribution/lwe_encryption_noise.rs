use super::*;
use crate::core_crypto::algorithms::misc::check_clear_content_respects_mod;
use crate::core_crypto::commons::test_tools::{
    modular_distance, modular_distance_custom_mod, torus_modular_diff, variance,
};
use rayon::prelude::*;

// This is 1 / 16 which is exactly representable in an f64 (even an f32)
// 1 / 32 is too strict and fails the tests
const RELATIVE_TOLERANCE: f64 = 0.0625;

const NB_TESTS: usize = 1000;

fn lwe_encrypt_decrypt_noise_distribution_custom_mod<Scalar: UnsignedTorus + CastInto<usize>>(
    params: ClassicTestParams<Scalar>,
) {
    let lwe_dimension = params.lwe_dimension;
    let lwe_noise_distribution = params.lwe_noise_distribution;
    let ciphertext_modulus = params.ciphertext_modulus;
    let message_modulus_log = params.message_modulus_log;
    let encoding_with_padding = get_encoding_with_padding(ciphertext_modulus);

    let expected_variance = Variance(lwe_noise_distribution.gaussian_std_dev().get_variance());

    let mut rsc = TestResources::new();

    let msg_modulus = Scalar::ONE.shl(message_modulus_log.0);
    let mut msg = msg_modulus;
    let delta: Scalar = encoding_with_padding / msg_modulus;

    let num_samples = NB_TESTS * <Scalar as CastInto<usize>>::cast_into(msg);
    let mut noise_samples = Vec::with_capacity(num_samples);

    while msg != Scalar::ZERO {
        msg = msg.wrapping_sub(Scalar::ONE);
        for _ in 0..NB_TESTS {
            let lwe_sk = allocate_and_generate_new_binary_lwe_secret_key(
                lwe_dimension,
                &mut rsc.secret_random_generator,
            );

            let mut ct = LweCiphertext::new(
                Scalar::ZERO,
                lwe_dimension.to_lwe_size(),
                ciphertext_modulus,
            );

            let plaintext = Plaintext(msg * delta);

            encrypt_lwe_ciphertext(
                &lwe_sk,
                &mut ct,
                plaintext,
                lwe_noise_distribution,
                &mut rsc.encryption_random_generator,
            );

            assert!(check_encrypted_content_respects_mod(
                &ct,
                ciphertext_modulus
            ));

            let decrypted = decrypt_lwe_ciphertext(&lwe_sk, &ct);

            let decoded = round_decode(decrypted.0, delta) % msg_modulus;

            assert_eq!(msg, decoded);

            let torus_diff = torus_modular_diff(plaintext.0, decrypted.0, ciphertext_modulus);
            noise_samples.push(torus_diff);
        }
    }

    let measured_variance = variance(&noise_samples);
    let var_abs_diff = (expected_variance.0 - measured_variance.0).abs();
    let tolerance_threshold = RELATIVE_TOLERANCE * expected_variance.0;
    assert!(
        var_abs_diff < tolerance_threshold,
        "Absolute difference for variance: {var_abs_diff}, \
        tolerance threshold: {tolerance_threshold}, \
        got variance: {measured_variance:?}, \
        expected variance: {expected_variance:?}"
    );
}

create_parametrized_test!(lwe_encrypt_decrypt_noise_distribution_custom_mod {
    TEST_PARAMS_4_BITS_NATIVE_U64,
    TEST_PARAMS_3_BITS_SOLINAS_U64,
    TEST_PARAMS_3_BITS_63_U64
});

fn lwe_compact_public_key_encryption_expected_variance(
    input_noise: impl DispersionParameter,
    lwe_dimension: LweDimension,
) -> Variance {
    let input_variance = input_noise.get_variance();
    Variance(input_variance * (lwe_dimension.to_lwe_size().0 as f64))
}

#[test]
fn test_variance_increase_cpk_formula() {
    let predicted_variance = lwe_compact_public_key_encryption_expected_variance(
        StandardDev(2.0_f64.powi(39)),
        LweDimension(1024),
    );

    assert!(
        (predicted_variance.get_standard_dev().log2() - 44.000704097196405f64).abs() < f64::EPSILON
    );
}

fn lwe_compact_public_encrypt_noise_distribution_custom_mod<
    Scalar: UnsignedTorus + CastInto<usize>,
>(
    params: ClassicTestParams<Scalar>,
) {
    let lwe_dimension = LweDimension(params.polynomial_size.0);
    let glwe_noise_distribution = params.glwe_noise_distribution;
    let ciphertext_modulus = params.ciphertext_modulus;
    let message_modulus_log = params.message_modulus_log;
    let encoding_with_padding = get_encoding_with_padding(ciphertext_modulus);

    let glwe_variance = Variance(glwe_noise_distribution.gaussian_std_dev().get_variance());

    let expected_variance =
        lwe_compact_public_key_encryption_expected_variance(glwe_variance, lwe_dimension);

    let mut rsc = TestResources::new();

    let msg_modulus = Scalar::ONE.shl(message_modulus_log.0);
    let mut msg = msg_modulus;
    let delta: Scalar = encoding_with_padding / msg_modulus;

    let num_samples = NB_TESTS * <Scalar as CastInto<usize>>::cast_into(msg);
    let mut noise_samples = Vec::with_capacity(num_samples);

    while msg != Scalar::ZERO {
        msg = msg.wrapping_sub(Scalar::ONE);
        for _ in 0..NB_TESTS {
            let lwe_sk = allocate_and_generate_new_binary_lwe_secret_key(
                lwe_dimension,
                &mut rsc.secret_random_generator,
            );

            let pk = allocate_and_generate_new_lwe_compact_public_key(
                &lwe_sk,
                glwe_noise_distribution,
                ciphertext_modulus,
                &mut rsc.encryption_random_generator,
            );

            let mut ct = LweCiphertext::new(
                Scalar::ZERO,
                lwe_dimension.to_lwe_size(),
                ciphertext_modulus,
            );

            let plaintext = Plaintext(msg * delta);

            encrypt_lwe_ciphertext_with_compact_public_key(
                &pk,
                &mut ct,
                plaintext,
                glwe_noise_distribution,
                glwe_noise_distribution,
                &mut rsc.secret_random_generator,
                &mut rsc.encryption_random_generator,
            );

            assert!(check_encrypted_content_respects_mod(
                &ct,
                ciphertext_modulus
            ));

            let decrypted = decrypt_lwe_ciphertext(&lwe_sk, &ct);

            let decoded = round_decode(decrypted.0, delta) % msg_modulus;

            assert_eq!(msg, decoded);

            let torus_diff = torus_modular_diff(plaintext.0, decrypted.0, ciphertext_modulus);
            noise_samples.push(torus_diff);
        }
    }

    let measured_variance = variance(&noise_samples);
    let var_abs_diff = (expected_variance.0 - measured_variance.0).abs();
    let tolerance_threshold = RELATIVE_TOLERANCE * expected_variance.0;
    assert!(
        var_abs_diff < tolerance_threshold,
        "Absolute difference for variance: {var_abs_diff}, \
        tolerance threshold: {tolerance_threshold}, \
        got variance: {measured_variance:?}, \
        expected variance: {expected_variance:?}"
    );
}

create_parametrized_test!(lwe_compact_public_encrypt_noise_distribution_custom_mod {
    TEST_PARAMS_4_BITS_NATIVE_U64
});

fn random_noise_roundtrip<Scalar: UnsignedTorus + CastInto<usize>>(
    params: ClassicTestParams<Scalar>,
) {
    let mut rsc = TestResources::new();
    let noise = params.glwe_noise_distribution;
    let ciphertext_modulus = params.ciphertext_modulus;
    let encryption_rng = &mut rsc.encryption_random_generator;

    assert!(matches!(noise, DynamicDistribution::Gaussian(_)));

    let expected_variance = Variance(noise.gaussian_std_dev().get_variance());

    let num_ouptuts = 100_000;

    let mut output: Vec<_> = vec![Scalar::ZERO; num_ouptuts];

    encryption_rng.fill_slice_with_random_noise_from_distribution_custom_mod(
        &mut output,
        noise,
        ciphertext_modulus,
    );

    assert!(check_clear_content_respects_mod(
        &output,
        ciphertext_modulus
    ));

    for val in output.iter().copied() {
        if ciphertext_modulus.is_native_modulus() {
            let float_torus = val.into_torus();
            let from_torus = Scalar::from_torus(float_torus);
            assert!(
                modular_distance(val, from_torus)
                    < (Scalar::ONE << (Scalar::BITS.saturating_sub(f64::MANTISSA_DIGITS as usize))),
                "val={val}, from_torus={from_torus}, float_torus={float_torus}"
            );
        } else {
            let custom_modulus_as_scalar: Scalar =
                ciphertext_modulus.get_custom_modulus().cast_into();

            let float_torus = val.into_torus_custom_mod(custom_modulus_as_scalar);
            let from_torus = Scalar::from_torus_custom_mod(float_torus, custom_modulus_as_scalar);
            assert!(from_torus < custom_modulus_as_scalar);
            assert!(
                modular_distance_custom_mod(val, from_torus, custom_modulus_as_scalar)
                    < (Scalar::ONE << (Scalar::BITS.saturating_sub(f64::MANTISSA_DIGITS as usize))),
                "val={val}, from_torus={from_torus}, float_torus={float_torus}"
            );
        }
    }

    let output: Vec<_> = output
        .into_iter()
        .map(|x| torus_modular_diff(Scalar::ZERO, x, ciphertext_modulus))
        .collect();

    let measured_variance = variance(&output);
    let var_abs_diff = (expected_variance.0 - measured_variance.0).abs();
    let tolerance_threshold = RELATIVE_TOLERANCE * expected_variance.0;
    assert!(
        var_abs_diff < tolerance_threshold,
        "Absolute difference for variance: {var_abs_diff}, \
            tolerance threshold: {tolerance_threshold}, \
            got variance: {measured_variance:?}, \
            expected variance: {expected_variance:?}"
    );
}

create_parametrized_test!(random_noise_roundtrip {
    TEST_PARAMS_4_BITS_NATIVE_U64,
    TEST_PARAMS_3_BITS_SOLINAS_U64,
    TEST_PARAMS_3_BITS_63_U64
});

use crate::shortint::parameters::multi_bit::p_fail_2_minus_64::ks_pbs_gpu::*;

#[test]
fn lwe_encrypt_multi_bit_pbs_decrypt_noise_distribution_custom_mod() {
    use concrete_cpu_noise_model::gaussian_noise::noise::blind_rotate::variance_blind_rotate;
    use concrete_cpu_noise_model::gaussian_noise::noise::multi_bit_blind_rotate::variance_multi_bit_blind_rotate;
    let param_array = [
        (
            "PARAM_GPU_MULTI_BIT_GROUP_2_MESSAGE_1_CARRY_1_KS_PBS_GAUSSIAN_2M64",
            PARAM_GPU_MULTI_BIT_GROUP_2_MESSAGE_1_CARRY_1_KS_PBS_GAUSSIAN_2M64,
        ),
        (
            "PARAM_GPU_MULTI_BIT_GROUP_2_MESSAGE_2_CARRY_2_KS_PBS_GAUSSIAN_2M64",
            PARAM_GPU_MULTI_BIT_GROUP_2_MESSAGE_2_CARRY_2_KS_PBS_GAUSSIAN_2M64,
        ),
        (
            "PARAM_GPU_MULTI_BIT_GROUP_2_MESSAGE_3_CARRY_3_KS_PBS_GAUSSIAN_2M64",
            PARAM_GPU_MULTI_BIT_GROUP_2_MESSAGE_3_CARRY_3_KS_PBS_GAUSSIAN_2M64,
        ),
        (
            "PARAM_GPU_MULTI_BIT_GROUP_3_MESSAGE_1_CARRY_1_KS_PBS_GAUSSIAN_2M64",
            PARAM_GPU_MULTI_BIT_GROUP_3_MESSAGE_1_CARRY_1_KS_PBS_GAUSSIAN_2M64,
        ),
        (
            "PARAM_GPU_MULTI_BIT_GROUP_3_MESSAGE_2_CARRY_2_KS_PBS_GAUSSIAN_2M64",
            PARAM_GPU_MULTI_BIT_GROUP_3_MESSAGE_2_CARRY_2_KS_PBS_GAUSSIAN_2M64,
        ),
        (
            "PARAM_GPU_MULTI_BIT_GROUP_3_MESSAGE_3_CARRY_3_KS_PBS_GAUSSIAN_2M64",
            PARAM_GPU_MULTI_BIT_GROUP_3_MESSAGE_3_CARRY_3_KS_PBS_GAUSSIAN_2M64,
        ),
        (
            "PARAM_MULTI_BIT_GROUP_4_MESSAGE_1_CARRY_1_KS_PBS_GAUSSIAN_2M64",
            PARAM_MULTI_BIT_GROUP_4_MESSAGE_1_CARRY_1_KS_PBS_GAUSSIAN_2M64,
        ),
        (
            "PARAM_MULTI_BIT_GROUP_4_MESSAGE_2_CARRY_2_KS_PBS_GAUSSIAN_2M64",
            PARAM_MULTI_BIT_GROUP_4_MESSAGE_2_CARRY_2_KS_PBS_GAUSSIAN_2M64,
        ),
        (
            "PARAM_MULTI_BIT_GROUP_4_MESSAGE_3_CARRY_3_KS_PBS_GAUSSIAN_2M64",
            PARAM_MULTI_BIT_GROUP_4_MESSAGE_3_CARRY_3_KS_PBS_GAUSSIAN_2M64,
        ),
    ];

    for (name, params) in param_array {
        println!("{name}");
        let mut rsc = TestResources::new();

        let lwe_dimension = params.lwe_dimension;
        let glwe_dimension = params.glwe_dimension;
        let polynomial_size = params.polynomial_size;
        let ciphertext_modulus = params.ciphertext_modulus;
        let decomp_base_log = params.pbs_base_log;
        let decomp_level_count = params.pbs_level;
        let grouping_factor = params.grouping_factor;
        let glwe_noise_distribution = params.glwe_noise_distribution;
        let encoding_with_padding = get_encoding_with_padding(ciphertext_modulus);
        let msg_modulus = (params.carry_modulus.0 * params.message_modulus.0) as u64;
        let mut msg = msg_modulus;
        let delta = encoding_with_padding / msg_modulus;

        let num_samples = NB_TESTS * msg as usize;
        let mut noise_samples = Vec::with_capacity(num_samples);

        let small_lwe_sk = allocate_and_generate_new_binary_lwe_secret_key(
            lwe_dimension,
            &mut rsc.secret_random_generator,
        );

        let glwe_sk = allocate_and_generate_new_binary_glwe_secret_key(
            glwe_dimension,
            polynomial_size,
            &mut rsc.secret_random_generator,
        );

        let big_lwe_sk = glwe_sk.as_lwe_secret_key();

        let bsk = par_allocate_and_generate_new_lwe_multi_bit_bootstrap_key(
            &small_lwe_sk,
            &glwe_sk,
            decomp_base_log,
            decomp_level_count,
            grouping_factor,
            glwe_noise_distribution,
            ciphertext_modulus,
            &mut rsc.encryption_random_generator,
        );

        let mut fbsk = FourierLweMultiBitBootstrapKey::new(
            bsk.input_lwe_dimension(),
            bsk.glwe_size(),
            bsk.polynomial_size(),
            bsk.decomposition_base_log(),
            bsk.decomposition_level_count(),
            bsk.grouping_factor(),
        );

        par_convert_standard_lwe_multi_bit_bootstrap_key_to_fourier(&bsk, &mut fbsk);

        let fft_once_at_the_end = false;

        let expected_variance = Variance(variance_multi_bit_blind_rotate(
            fbsk.input_lwe_dimension().0 as u64,
            fbsk.glwe_size().to_glwe_dimension().0 as u64,
            fbsk.polynomial_size().0 as u64,
            decomp_base_log.0 as u64,
            decomp_level_count.0 as u64,
            64,
            53,
            glwe_noise_distribution.gaussian_variance().0,
            grouping_factor.0 as u32,
            fft_once_at_the_end,
        ));

        // let bsk = par_allocate_and_generate_new_lwe_bootstrap_key(
        //     &small_lwe_sk,
        //     &glwe_sk,
        //     decomp_base_log,
        //     decomp_level_count,
        //     glwe_noise_distribution,
        //     ciphertext_modulus,
        //     &mut rsc.encryption_random_generator,
        // );

        // let mut fbsk = FourierLweBootstrapKey::new(
        //     bsk.input_lwe_dimension(),
        //     bsk.glwe_size(),
        //     bsk.polynomial_size(),
        //     bsk.decomposition_base_log(),
        //     bsk.decomposition_level_count(),
        // );

        // par_convert_standard_lwe_bootstrap_key_to_fourier(&bsk, &mut fbsk);

        // let expected_variance = Variance(variance_blind_rotate(
        //     fbsk.input_lwe_dimension().0 as u64,
        //     fbsk.glwe_size().to_glwe_dimension().0 as u64,
        //     fbsk.polynomial_size().0 as u64,
        //     decomp_base_log.0 as u64,
        //     decomp_level_count.0 as u64,
        //     64,
        //     53,
        //     glwe_noise_distribution.gaussian_variance().0,
        // ));

        let accumulator = generate_programmable_bootstrap_glwe_lut(
            polynomial_size,
            glwe_dimension.to_glwe_size(),
            msg_modulus as usize,
            ciphertext_modulus,
            delta,
            |x| x,
        );

        while msg != 0 {
            msg -= 1;
            println!("msg: {msg}");
            let curr_samples = (0..NB_TESTS)
                .into_par_iter()
                .map(|idx| {
                    let mut rsc = TestResources::new();

                    let mut ct =
                        LweCiphertext::new(0u64, lwe_dimension.to_lwe_size(), ciphertext_modulus);

                    let plaintext = Plaintext(msg * delta);

                    encrypt_lwe_ciphertext(
                        &small_lwe_sk,
                        &mut ct,
                        plaintext,
                        DynamicDistribution::new_gaussian_from_std_dev(StandardDev(0.0)),
                        &mut rsc.encryption_random_generator,
                    );

                    assert!(check_encrypted_content_respects_mod(
                        &ct,
                        ciphertext_modulus
                    ));

                    let mut output_ct = LweCiphertext::new(
                        0u64,
                        fbsk.output_lwe_dimension().to_lwe_size(),
                        ciphertext_modulus,
                    );

                    if fft_once_at_the_end {
                        std_multi_bit_programmable_bootstrap_lwe_ciphertext(
                            &ct,
                            &mut output_ct,
                            &accumulator,
                            &bsk,
                            ThreadCount(1 << grouping_factor.0),
                            false,
                        );
                    } else {
                        multi_bit_programmable_bootstrap_lwe_ciphertext(
                            &ct,
                            &mut output_ct,
                            &accumulator,
                            &fbsk,
                            ThreadCount(1 << grouping_factor.0),
                            false,
                        );
                    }
                    // programmable_bootstrap_lwe_ciphertext(
                    //     &ct,
                    //     &mut output_ct,
                    //     &accumulator,
                    //     &fbsk,
                    // );

                    let decrypted = decrypt_lwe_ciphertext(&big_lwe_sk, &output_ct);

                    let decoded = round_decode(decrypted.0, delta) % msg_modulus;

                    // println!("#{idx} done");
                    assert_eq!(msg, decoded);

                    let torus_diff =
                        torus_modular_diff(plaintext.0, decrypted.0, ciphertext_modulus);
                    torus_diff
                })
                .collect::<Vec<_>>();

            noise_samples.extend(curr_samples.into_iter());
        }

        let measured_variance = variance(&noise_samples);
        let var_abs_diff = (expected_variance.0 - measured_variance.0).abs();
        let tolerance_threshold = RELATIVE_TOLERANCE * expected_variance.0;
        if measured_variance.0 > expected_variance.0 {
            // assert!(
            //     var_abs_diff < tolerance_threshold,
            //     "Absolute difference for variance: {var_abs_diff}, \
            //     tolerance threshold: {tolerance_threshold}, \
            //     got variance: {measured_variance:?}, \
            //     expected variance: {expected_variance:?}"
            // );
            if var_abs_diff > tolerance_threshold {
                println!(
                    "Absolute difference for variance: {var_abs_diff}, \
                    tolerance threshold: {tolerance_threshold}, \n\
                    got variance: {measured_variance:?}, \n\
                    expected variance: {expected_variance:?}"
                );

                println!("FAIL ===========================================================")
            } else {
                println!("Measured variance is in the expected range : OK");
                println!("Measured: {measured_variance:?}");
                println!("Expected: {expected_variance:?}");
            }
        } else {
            println!("Measured variance is smaller than what would be expected : OK");
            println!("Measured: {measured_variance:?}");
            println!("Expected: {expected_variance:?}");
        }

        println!("{name} OK");
    }
}
