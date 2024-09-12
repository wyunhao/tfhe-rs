use super::*;

fn glwe_encrypt_assign_decrypt_custom_mod<Scalar: UnsignedTorus>(params: TestParams<Scalar>) {
    let glwe_dimension = params.glwe_dimension;
    let polynomial_size = params.polynomial_size;
    let glwe_modular_std_dev = params.glwe_modular_std_dev;
    let ciphertext_modulus = params.ciphertext_modulus;
    let message_modulus_log = params.message_modulus_log;
    let encoding_with_padding = get_encoding_with_padding(ciphertext_modulus);

    let mut rsc = TestResources::new();

    const NB_TESTS: usize = 10;
    let msg_modulus = Scalar::ONE.shl(message_modulus_log.0);
    let mut msg = msg_modulus;
    let delta: Scalar = encoding_with_padding / msg_modulus;

    while msg != Scalar::ZERO {
        msg = msg.wrapping_sub(Scalar::ONE);

        println!("{}", msg);
        for _ in 0..NB_TESTS {
            let glwe_sk = allocate_and_generate_new_binary_glwe_secret_key(
                glwe_dimension,
                polynomial_size,
                &mut rsc.secret_random_generator,
            );

            let mut glwe = GlweCiphertext::new(
                msg * delta,
                glwe_dimension.to_glwe_size(),
                polynomial_size,
                ciphertext_modulus,
            );

            encrypt_glwe_ciphertext_assign(
                &glwe_sk,
                &mut glwe,
                glwe_modular_std_dev,
                &mut rsc.encryption_random_generator,
            );

            assert!(check_content_respects_mod(&glwe, ciphertext_modulus));

            let mut plaintext_list =
                PlaintextList::new(Scalar::ZERO, PlaintextCount(glwe.polynomial_size().0));

            decrypt_glwe_ciphertext(&glwe_sk, &glwe, &mut plaintext_list);

            let mut decoded = vec![Scalar::ZERO; plaintext_list.plaintext_count().0];

            decoded
                .iter_mut()
                .zip(plaintext_list.iter())
                .for_each(|(dst, src)| *dst = round_decode(*src.0, delta) % msg_modulus);

            assert!(decoded.iter().all(|&x| x == msg));
        }
    }
}

create_parametrized_test!(glwe_encrypt_assign_decrypt_custom_mod);

fn glwe_encrypt_decrypt_custom_mod<Scalar: UnsignedTorus>(params: TestParams<Scalar>) {
    let glwe_dimension = params.glwe_dimension;
    let polynomial_size = params.polynomial_size;
    let glwe_modular_std_dev = params.glwe_modular_std_dev;
    let ciphertext_modulus = params.ciphertext_modulus;
    let message_modulus_log = params.message_modulus_log;
    let encoding_with_padding = get_encoding_with_padding(ciphertext_modulus);

    let mut rsc = TestResources::new();

    const NB_TESTS: usize = 10;
    let msg_modulus = Scalar::ONE.shl(message_modulus_log.0);
    let mut msg = msg_modulus;
    let delta: Scalar = encoding_with_padding / msg_modulus;

    while msg != Scalar::ZERO {
        msg = msg.wrapping_sub(Scalar::ONE);
        for _ in 0..NB_TESTS {
            let glwe_sk = allocate_and_generate_new_binary_glwe_secret_key(
                glwe_dimension,
                polynomial_size,
                &mut rsc.secret_random_generator,
            );

            let mut glwe = GlweCiphertext::new(
                Scalar::ZERO,
                glwe_dimension.to_glwe_size(),
                polynomial_size,
                ciphertext_modulus,
            );

            let plaintext_list =
                PlaintextList::new(msg * delta, PlaintextCount(glwe.polynomial_size().0));

            encrypt_glwe_ciphertext(
                &glwe_sk,
                &mut glwe,
                &plaintext_list,
                glwe_modular_std_dev,
                &mut rsc.encryption_random_generator,
            );

            assert!(check_content_respects_mod(&glwe, ciphertext_modulus));

            let mut plaintext_list =
                PlaintextList::new(Scalar::ZERO, PlaintextCount(glwe.polynomial_size().0));

            decrypt_glwe_ciphertext(&glwe_sk, &glwe, &mut plaintext_list);

            let mut decoded = vec![Scalar::ZERO; plaintext_list.plaintext_count().0];

            decoded
                .iter_mut()
                .zip(plaintext_list.iter())
                .for_each(|(dst, src)| *dst = round_decode(*src.0, delta) % msg_modulus);

            assert!(decoded.iter().all(|&x| x == msg));
        }
    }
}


fn glwe_allocate_trivial_encrypt_decrypt_custom_mod<Scalar: UnsignedTorus>(
    params: TestParams<Scalar>,
) {
    let glwe_dimension = params.glwe_dimension;
    let polynomial_size = params.polynomial_size;
    let ciphertext_modulus = params.ciphertext_modulus;
    let message_modulus_log = params.message_modulus_log;
    let encoding_with_padding = get_encoding_with_padding(ciphertext_modulus);

    let mut rsc = TestResources::new();

    const NB_TESTS: usize = 10;
    let msg_modulus = Scalar::ONE.shl(message_modulus_log.0);
    let mut msg = msg_modulus;
    let delta: Scalar = encoding_with_padding / msg_modulus;

    while msg != Scalar::ZERO {
        msg = msg.wrapping_sub(Scalar::ONE);
        for _ in 0..NB_TESTS {
            let lwe_sk = allocate_and_generate_new_binary_glwe_secret_key(
                glwe_dimension,
                polynomial_size,
                &mut rsc.secret_random_generator,
            );

            let plaintext_list = PlaintextList::new(msg * delta, PlaintextCount(polynomial_size.0));

            let ct = allocate_and_trivially_encrypt_new_glwe_ciphertext(
                glwe_dimension.to_glwe_size(),
                &plaintext_list,
                ciphertext_modulus,
            );

            assert!(check_content_respects_mod(&ct, ciphertext_modulus));

            let mut output_plaintext_list =
                PlaintextList::new(Scalar::ZERO, PlaintextCount(polynomial_size.0));

            decrypt_glwe_ciphertext(&lwe_sk, &ct, &mut output_plaintext_list);

            output_plaintext_list
                .iter_mut()
                .for_each(|x| *x.0 = round_decode(*x.0, delta) % msg_modulus);

            assert!(output_plaintext_list.iter().all(|x| *x.0 == msg));
        }
    }
}
