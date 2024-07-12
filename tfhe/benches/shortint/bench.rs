#[path = "../utilities.rs"]
mod utilities;

// use tfhe::shortint::parameters::*;
// use tfhe::shortint::{gen_keys};

use tfhe::core_crypto::algorithms::*;
use tfhe::core_crypto::fft_impl::fft128::crypto::bootstrap::bootstrap_scratch;
use tfhe::core_crypto::prelude::*;


pub fn round_decode<Scalar: UnsignedInteger>(decrypted: Scalar, delta: Scalar) -> Scalar {
    // Get half interval on the discretized torus
    let rounding_margin = delta.wrapping_div(Scalar::TWO);

    // Add the half interval mapping
    // [delta * (m - 1/2); delta * (m + 1/2)[ to [delta * m; delta * (m + 1)[
    // Dividing by delta gives m which is what we want
    (decrypted.wrapping_add(rounding_margin)).wrapping_div(delta)
}

fn main() {

    type Scalar = u128;

    let glwe_dimension = GlweDimension(1);
    let polynomial_size = PolynomialSize(2048);
    let ciphertext_modulus = CiphertextModulus::new_native();
    let glwe_modular_std_dev = StandardDev(0.00000000000000029403601535432533);

    let mut boxed_seeder = new_seeder();
    let seeder = boxed_seeder.as_mut();

    let mut secret_generator =
        SecretRandomGenerator::<ActivatedRandomGenerator>::new(seeder.seed());
    let mut encryption_generator =
        EncryptionRandomGenerator::<ActivatedRandomGenerator>::new(seeder.seed(), seeder);

    let glwe_sk = GlweSecretKey::<Vec<Scalar>>::generate_new_binary(
        glwe_dimension,
        polynomial_size,
        &mut secret_generator,
    );

    let mut glwe = GlweCiphertext::new(
        Scalar::ZERO,
        glwe_dimension.to_glwe_size(),
        polynomial_size,
        ciphertext_modulus,
    );

    let msg_modulus = 1024;
    let mut msg = Scalar::TWO;
    let delta = 4;

    let plaintext_list =
        PlaintextList::new(msg * delta, PlaintextCount(glwe.polynomial_size().0));

    encrypt_glwe_ciphertext(
        &glwe_sk,
        &mut glwe,
        &plaintext_list,
        glwe_modular_std_dev,
        &mut encryption_generator,
    );

    let mut output_plaintext_list =
        PlaintextList::new(Scalar::ZERO, PlaintextCount(glwe.polynomial_size().0));

    decrypt_glwe_ciphertext(&glwe_sk, &glwe, &mut output_plaintext_list);

    let mut decoded = vec![Scalar::ZERO; output_plaintext_list.plaintext_count().0];

    decoded
        .iter_mut()
        .zip(output_plaintext_list.iter())
        .for_each(|(dst, src)| *dst = round_decode(*src.0, delta) % msg_modulus);

    // assert!(decoded.iter().all(|&x| x == (msg) % msg_modulus));

    println!("Expected: {}, but with random noise: {:?}", msg, decoded);
    // decoded.iter().all(|&x| println!(x));





    //////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    //////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    ///////////////////////////// LWE simple test for encryption with random noise ///////////////////////////////////
    //////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    //////////////////////////////////////////////////////////////////////////////////////////////////////////////////


    // let (client_key, server_key) = gen_keys(PARAM_MESSAGE_2_CARRY_2_KS_PBS);

    // let msg1 = 8;
    // let msg2 = 18;

    // let modulus = client_key.parameters.message_modulus().0;

    // // We use the client key to encrypt two messages:
    // let ct_1 = client_key.encrypt(msg1);
    // let ct_2 = client_key.encrypt(msg2);

    // // We use the server public key to execute an integer circuit:
    // let ct_3 = server_key.unchecked_add(&ct_1, &ct_2);

    // // We use the client key to decrypt the output of the circuit:
    // let output = client_key.decrypt(&ct_3);
    // // assert_eq!(output, (msg1 + msg2) % modulus as u64);
    

    // println!("expected {}, found {}", (msg1 + msg2) % modulus as u64, output);

}
