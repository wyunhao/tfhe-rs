//! Module containing primitives pertaining to [`LWE trace packing keyswitch key
//! generation`](`LweTracePackingKeyswitchKey`).

use crate::core_crypto::algorithms::*;
use crate::core_crypto::commons::generators::EncryptionRandomGenerator;
use crate::core_crypto::commons::math::random::{Distribution, Uniform};
use crate::core_crypto::commons::traits::*;
use crate::core_crypto::entities::*;
use crate::core_crypto::prelude::polynomial_algorithms::apply_automorphism_wrapping_add_assign;

/// Fill a [`GLWE secret key`](`GlweSecretKey`) with an actual key derived from an
/// [`LWE secret key`](`LweSecretKey`) for use in the [`LWE trace packing keyswitch key`]
/// (`LweTracePackingKeyswitchKey`)
/// # Example
///
/// ```
/// use tfhe::core_crypto::prelude::*;
///
/// // DISCLAIMER: these toy example parameters are not guaranteed to be secure or yield correct
/// // computations
/// // Define parameters for GlweCiphertext creation
/// let glwe_size = GlweSize(3);
/// let polynomial_size = PolynomialSize(1024);
/// let lwe_dimension = LweDimension(900);
/// let ciphertext_modulus = CiphertextModulus::new_native();
///
/// let mut seeder = new_seeder();
/// let mut secret_generator =
///     SecretRandomGenerator::<ActivatedRandomGenerator>::new(seeder.seed());
/// let lwe_secret_key =
///     allocate_and_generate_new_binary_lwe_secret_key(lwe_dimension, &mut secret_generator);
///
/// let mut glwe_secret_key =
///     GlweSecretKey::new_empty_key(0u64, glwe_size.to_glwe_dimension(), polynomial_size);
///
/// generate_tpksk_output_glwe_secret_key(&lwe_secret_key, &mut glwe_secret_key);
///
/// let decomp_base_log = DecompositionBaseLog(2);
/// let decomp_level_count = DecompositionLevelCount(8);
/// let glwe_noise_distribution =
///     DynamicDistribution::new_gaussian(Variance::from_variance(2f64.powf(-80.0)));
/// let mut seeder = new_seeder();
/// let seeder = seeder.as_mut();
/// let mut encryption_generator =
///     EncryptionRandomGenerator::<ActivatedRandomGenerator>::new(seeder.seed(), seeder);
///
/// let mut lwe_tpksk = LweTracePackingKeyswitchKey::new(
///     0u64,
///     decomp_base_log,
///     decomp_level_count,
///     lwe_dimension.to_lwe_size(),
///     glwe_size,
///     polynomial_size,
///     ciphertext_modulus,
/// );
///
/// generate_lwe_trace_packing_keyswitch_key(
///     &glwe_secret_key,
///     &mut lwe_tpksk,
///     glwe_noise_distribution,
///     &mut encryption_generator,
/// );
/// ```
pub fn generate_tpksk_output_glwe_secret_key<Scalar, InputKeyCont, OutputKeyCont>(
    input_lwe_secret_key: &LweSecretKey<InputKeyCont>,
    output_glwe_secret_key: &mut GlweSecretKey<OutputKeyCont>,
) where
    Scalar: UnsignedTorus,
    InputKeyCont: Container<Element = Scalar>,
    OutputKeyCont: ContainerMut<Element = Scalar>,
{
    let lwe_dimension = input_lwe_secret_key.lwe_dimension();
    let glwe_dimension = output_glwe_secret_key.glwe_dimension();
    let glwe_poly_size = output_glwe_secret_key.polynomial_size();

    assert!(
        lwe_dimension.0 <= glwe_dimension.0 * glwe_poly_size.0,
        "Mismatched between input_lwe_secret_key dimension {:?} and number of coefficients of \
        output_glwe_secret_key {:?}.",
        lwe_dimension.0,
        glwe_dimension.0 * glwe_poly_size.0
    );

    let glwe_key_container = output_glwe_secret_key.as_mut();

    for (index, lwe_key_bit) in input_lwe_secret_key.as_ref().iter().enumerate() {
        if index % glwe_poly_size.0 == 0 {
            glwe_key_container[index] = *lwe_key_bit;
        } else {
            let rem = index % glwe_poly_size.0;
            let quo = index / glwe_poly_size.0;
            let new_index = (quo + 1) * glwe_poly_size.0 - rem;
            glwe_key_container[new_index] = Scalar::ZERO.wrapping_sub(*lwe_key_bit);
        }
    }
}

/// Fill an [`LWE trace packing keyswitch key`](`LweTracePackingKeyswitchKey`)
/// with an actual key.
pub fn generate_lwe_trace_packing_keyswitch_key<
    Scalar,
    NoiseDistribution,
    InputKeyCont,
    KSKeyCont,
    Gen,
>(
    input_glwe_secret_key: &GlweSecretKey<InputKeyCont>,
    lwe_tpksk: &mut LweTracePackingKeyswitchKey<KSKeyCont>,
    noise_distribution: NoiseDistribution,
    generator: &mut EncryptionRandomGenerator<Gen>,
) where
    Scalar: Encryptable<Uniform, NoiseDistribution>,
    NoiseDistribution: Distribution,
    InputKeyCont: Container<Element = Scalar>,
    KSKeyCont: ContainerMut<Element = Scalar>,
    Gen: ByteRandomGenerator,
{
    assert_eq!(
        input_glwe_secret_key.glwe_dimension(),
        lwe_tpksk.output_glwe_key_dimension()
    );
    assert_eq!(
        input_glwe_secret_key.polynomial_size(),
        lwe_tpksk.polynomial_size()
    );

    let glwe_dimension = lwe_tpksk.output_glwe_key_dimension();
    let decomp_level_count = lwe_tpksk.decomposition_level_count();
    let decomp_base_log = lwe_tpksk.decomposition_base_log();
    let polynomial_size = lwe_tpksk.polynomial_size();
    let ciphertext_modulus = lwe_tpksk.ciphertext_modulus();

    // let gen_iter = generator
    //     .fork_tpksk_to_tpksk_chunks::<Scalar>(
    //         decomp_level_count,
    //         glwe_dimension.to_glwe_size(),
    //         polynomial_size,
    //     )
    //     .unwrap();

    // loop over the before key blocks

    // for (automorphism_index, (glwe_keyswitch_block, mut loop_generator)) in
    //     lwe_tpksk.iter_mut().zip(gen_iter).enumerate()
    // {

    assert_eq!(lwe_tpksk.entity_count(), polynomial_size.log2().0);

    for (automorphism_index, glwe_keyswitch_block) in lwe_tpksk.iter_mut().enumerate() {
        let automorphism_exponent = 2usize.pow(automorphism_index as u32 + 1) + 1;
        let mut auto_glwe_sk = GlweSecretKey::new_empty_key(
            Scalar::ZERO,
            input_glwe_secret_key.glwe_dimension(),
            input_glwe_secret_key.polynomial_size(),
        );

        for (mut output_poly, input_poly) in auto_glwe_sk
            .as_mut_polynomial_list()
            .iter_mut()
            .zip(input_glwe_secret_key.as_polynomial_list().iter())
        {
            apply_automorphism_wrapping_add_assign(
                &mut output_poly,
                &input_poly,
                automorphism_exponent,
            );
        }
        let mut glwe_ksk = GlweKeyswitchKey::from_container(
            glwe_keyswitch_block.into_container(),
            decomp_base_log,
            decomp_level_count,
            glwe_dimension.to_glwe_size(),
            polynomial_size,
            ciphertext_modulus,
        );
        generate_glwe_keyswitch_key(
            &auto_glwe_sk,
            input_glwe_secret_key,
            &mut glwe_ksk,
            noise_distribution,
            generator,
        );
    }
}
