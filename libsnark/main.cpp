#include <cassert>
#include <cstdio>
#include <fstream>
#include <fstream>
#include <libff/common/rng.hpp>
#include <libff/common/profiling.hpp>
#include <libff/common/utils.hpp>
#include <libsnark/serialization.hpp>
#include <libff/algebra/curves/mnt753/mnt4753/mnt4753_pp.hpp>
#include <libff/algebra/curves/mnt753/mnt6753/mnt6753_pp.hpp>
#include <omp.h>
#include <libff/algebra/scalar_multiplication/multiexp.hpp>
#include <libsnark/knowledge_commitment/kc_multiexp.hpp>
#include <libsnark/knowledge_commitment/knowledge_commitment.hpp>
#include <libsnark/reductions/r1cs_to_qap/r1cs_to_qap.hpp>
#include <time.h>

#include <libsnark/zk_proof_systems/ppzksnark/r1cs_gg_ppzksnark/r1cs_gg_ppzksnark.hpp>

#include <libfqfft/evaluation_domain/domains/basic_radix2_domain.hpp>

#include "cuda-fixnum/main.cuh"

using namespace libff;
using namespace libsnark;

//#undef MULTICORE
//const multi_exp_method method = multi_exp_method_BDLO12;
const multi_exp_method method = multi_exp_method_bos_coster;
//const multi_exp_method method = multi_exp_method_naive_plain;

template<typename ppT>
class groth16_parameters {
  public:
    size_t d;
    size_t m;
    std::vector<G1<ppT>> A, B1, L, H;
    std::vector<G2<ppT>> B2;

  groth16_parameters(const char* path) {
    FILE* params = fopen(path, "r");
    d = read_size_t(params);
    m = read_size_t(params);
    for (size_t i = 0; i <= m; ++i) { A.emplace_back(read_g1<ppT>(params)); }
    for (size_t i = 0; i <= m; ++i) { B1.emplace_back(read_g1<ppT>(params)); }
    for (size_t i = 0; i <= m; ++i) { B2.emplace_back(read_g2<ppT>(params)); }
    for (size_t i = 0; i < m-1; ++i) { L.emplace_back(read_g1<ppT>(params)); }
    for (size_t i = 0; i < d; ++i) { H.emplace_back(read_g1<ppT>(params)); }
    fclose(params);
  }
};

template<typename ppT>
class groth16_input {
  public:
    std::vector<Fr<ppT>> w;
    std::vector<Fr<ppT>> ca, cb, cc;
    Fr<ppT> r;

  groth16_input(const char* path, size_t d, size_t m) {
    FILE* inputs = fopen(path, "r");

    for (size_t i = 0; i < m + 1; ++i) { w.emplace_back(read_fr<ppT>(inputs)); }

    for (size_t i = 0; i < d + 1; ++i) { ca.emplace_back(read_fr<ppT>(inputs)); }
    for (size_t i = 0; i < d + 1; ++i) { cb.emplace_back(read_fr<ppT>(inputs)); }
    for (size_t i = 0; i < d + 1; ++i) { cc.emplace_back(read_fr<ppT>(inputs)); }

    r = read_fr<ppT>(inputs);

    fclose(inputs);
  }
};

template<typename ppT>
class groth16_output {
  public:
    G1<ppT> A, C;
    G2<ppT> B;

  groth16_output(G1<ppT> &&A, G2<ppT> &&B, G1<ppT> &&C) :
    A(std::move(A)), B(std::move(B)), C(std::move(C)) {}

    groth16_output(G1<ppT> &&A) :
        A(std::move(A)) {}

  void write(const char* path) {
    FILE* out = fopen(path, "w");
    write_g1<ppT>(out, A);
    write_g2<ppT>(out, B);
    write_g1<ppT>(out, C);
    fclose(out);
  }
};

// Here is where all the FFTs happen.
template<typename ppT>
std::vector<Fr<ppT>> compute_H(
    size_t d,
    std::vector<Fr<ppT>> &ca,
    std::vector<Fr<ppT>> &cb,
    std::vector<Fr<ppT>> &cc)
{
    // Begin witness map
    libff::enter_block("Compute the polynomial H");

    const std::shared_ptr<libfqfft::evaluation_domain<Fr<ppT>> > domain = libfqfft::get_evaluation_domain<Fr<ppT>>(d + 1);

    domain->iFFT(ca);
    domain->iFFT(cb);

    domain->cosetFFT(ca, Fr<ppT>::multiplicative_generator);
    domain->cosetFFT(cb, Fr<ppT>::multiplicative_generator);

    libff::enter_block("Compute evaluation of polynomial H on set T");
    std::vector<Fr<ppT>> &H_tmp = ca; // can overwrite ca because it is not used later
#ifdef MULTICORE
#pragma omp parallel for
#endif
    for (size_t i = 0; i < domain->m; ++i)
    {
        H_tmp[i] = ca[i]*cb[i];
    }
    std::vector<Fr<ppT>>().swap(cb); // destroy cb

    domain->iFFT(cc);

    domain->cosetFFT(cc, Fr<ppT>::multiplicative_generator);

#ifdef MULTICORE
#pragma omp parallel for
#endif
    for (size_t i = 0; i < domain->m; ++i)
    {
        H_tmp[i] = (H_tmp[i]-cc[i]);
    }

    domain->divide_by_Z_on_coset(H_tmp);

    libff::leave_block("Compute evaluation of polynomial H on set T");

    domain->icosetFFT(H_tmp, Fr<ppT>::multiplicative_generator);

    std::vector<Fr<ppT>> coefficients_for_H(domain->m+1, Fr<ppT>::zero());
#ifdef MULTICORE
#pragma omp parallel for
#endif
    for (size_t i = 0; i < domain->m; ++i)
    {
        coefficients_for_H[i] = H_tmp[i];
    }

    libff::leave_block("Compute the polynomial H");

    return coefficients_for_H;
}

template<typename G, typename Fr>
G multiexp(typename std::vector<Fr>::const_iterator scalar_start,
           typename std::vector<G>::const_iterator g_start,
           size_t length)
{
#ifdef MULTICORE
    const size_t chunks = omp_get_max_threads(); // to override, set OMP_NUM_THREADS env var or call omp_set_num_threads()
#else
    const size_t chunks = 1;
#endif

    return libff::multi_exp_with_mixed_addition<G,
                                                Fr,
                                                method>(
        g_start,
        g_start + length,
        scalar_start,
        scalar_start + length,
        chunks);

}

template<typename G, typename Fr>
G multiexpGPU(typename std::vector<Fr>::const_iterator scalar_start,
           typename std::vector<G>::const_iterator g_start,
           size_t length)
{
#ifdef MULTICORE
    const size_t chunks = omp_get_max_threads(); // to override, set OMP_NUM_THREADS env var or call omp_set_num_threads()
#else
    const size_t chunks = 1;
#endif

    return libff::multi_exp_with_mixed_addition_gpu<G,
                                                    Fr,
                                                    method>(
        g_start,
        g_start + length,
        scalar_start,
        scalar_start + length,
        chunks);

}

template<typename ppT>
int run_prover(
    const char* params_path,
    const char* input_path,
    const char* output_path)
{
    time_t start_time, finish_time;
    time(&start_time);
    ppT::init_public_params();

    const size_t primary_input_size = 1;

    const groth16_parameters<ppT> parameters(params_path);
    const groth16_input<ppT> input(input_path, parameters.d, parameters.m);

    std::vector<Fr<ppT>> w  = std::move(input.w);
    std::vector<Fr<ppT>> ca = std::move(input.ca);
    std::vector<Fr<ppT>> cb = std::move(input.cb);
    std::vector<Fr<ppT>> cc = std::move(input.cc);

    // End reading of parameters and input

    libff::enter_block("Call to r1cs_gg_ppzksnark_prover");
    std::vector<Fr<ppT>> coefficients_for_H = compute_H<ppT>(
        parameters.d,
        ca, cb, cc);

    libff::enter_block("Compute the proof");
    libff::enter_block("Multi-exponentiations");

    // Now the 5 multi-exponentiations
    G1<ppT> evaluation_At = multiexp<G1<ppT>, Fr<ppT>>(
        w.begin(), parameters.A.begin(), parameters.m + 1);

#if 0
    printf("evaluation_At\n");
    evaluation_At.X().mont_repr.print_hex();
    evaluation_At.Y().mont_repr.print_hex();
    evaluation_At.Z().mont_repr.print_hex();
#endif

    G1<ppT> evaluation_Bt1 = multiexp<G1<ppT>, Fr<ppT>>(
        w.begin(), parameters.B1.begin(), parameters.m + 1);
#if 0
    printf("evaluation_Bt1t\n");
    evaluation_Bt1.X().mont_repr.print_hex();
    evaluation_Bt1.Y().mont_repr.print_hex();
    evaluation_Bt1.Z().mont_repr.print_hex();
#endif

    G2<ppT> evaluation_Bt2 = multiexp<G2<ppT>, Fr<ppT>>(
        w.begin(), parameters.B2.begin(), parameters.m + 1);
#if 0
    printf("evaluation_Bt2\n");
    evaluation_Bt2.X().c0.mont_repr.print_hex();
    evaluation_Bt2.X().c1.mont_repr.print_hex();
    evaluation_Bt2.Y().c0.mont_repr.print_hex();
    evaluation_Bt2.Y().c1.mont_repr.print_hex();
    evaluation_Bt2.Z().c0.mont_repr.print_hex();
    evaluation_Bt2.Z().c1.mont_repr.print_hex();
#endif
    G1<ppT> evaluation_Ht = multiexp<G1<ppT>, Fr<ppT>>(
        coefficients_for_H.begin(), parameters.H.begin(), parameters.d);
#if 0
    printf("evaluation_Ht\n");
    evaluation_Ht.X().mont_repr.print_hex();
    evaluation_Ht.Y().mont_repr.print_hex();
    evaluation_Ht.Z().mont_repr.print_hex();
#endif

    G1<ppT> evaluation_Lt = multiexp<G1<ppT>, Fr<ppT>>(
        w.begin() + primary_input_size + 1,
        parameters.L.begin(),
        parameters.m - 1);
#if 0
    printf("evaluation_Lt\n");
    evaluation_Lt.X().mont_repr.print_hex();
    evaluation_Lt.Y().mont_repr.print_hex();
    evaluation_Lt.Z().mont_repr.print_hex();
#endif

    libff::G1<ppT> C = evaluation_Ht + evaluation_Lt + input.r * evaluation_Bt1; /*+ s *  g1_A  - (r * s) * pk.delta_g1; */

    libff::leave_block("Multi-exponentiations");
    libff::leave_block("Compute the proof");
    libff::leave_block("Call to r1cs_gg_ppzksnark_prover");

    groth16_output<ppT> output(std::move(evaluation_At),
                               std::move(evaluation_Bt2),
                               std::move(C));

    output.write(output_path);

    time(&finish_time);
    int duration = finish_time - start_time;
    printf("CPU all cost %d seconds\n", duration);
    return 0;
}

void setBigData(bigint<mnt4753_q_limbs> *bigintV, uint8_t *val, int size) {
    for (int i = 0; i < size; i+=8) {
        unsigned long x0, x1, x2, x3, x4, x5, x6, x7;
        x0 = val[i];
        x1 = val[i+1];
        x2 = val[i+2];
        x3 = val[i+3];
        x4 = val[i+4];
        x5 = val[i+5];
        x6 = val[i+6];
        x7 = val[i+7];
        bigintV->data[i/8] = x7<<56 | x6<<48 | x5<<40 | x4<<32 | x3<<24 | x2<<16 | x1<<8 | x0;
    }
}

template<typename G, typename Fr>
G multiexpG1(typename std::vector<Fr>::const_iterator scalar_start,
           typename std::vector<G>::const_iterator g_start,
           size_t length) {
    typename std::vector<G>::const_iterator vec_start = g_start;
    typename std::vector<G>::const_iterator vec_end = g_start + length;
    typename std::vector<Fr>::const_iterator scalar_end = scalar_start + length;
    typename std::vector<G>::const_iterator vec_it;
    typename std::vector<Fr>::const_iterator scalar_it;
    G acc = G::zero();
    int size = length;//vec_end - vec_start;
    int esize = 96;
    int total_size = size*esize;
    uint8_t *x_val = new uint8_t[total_size];
    uint8_t *y_val = new uint8_t[total_size];
    uint8_t *z_val = new uint8_t[total_size];
    uint8_t *scalar_val = new uint8_t[total_size];
    memset(x_val, 0x0, total_size);
    memset(y_val, 0x0, total_size);
    memset(z_val, 0x0, total_size);
    memset(scalar_val, 0x0, total_size);
    uint8_t x3[esize];
    uint8_t y3[esize];
    uint8_t z3[esize];
    uint8_t *val = new uint8_t[esize];;
    int i = 0;
    for (vec_it = vec_start, scalar_it = scalar_start; vec_it != vec_end; ++vec_it, ++scalar_it)
    {
        memset(val, 0x0, esize);
        ((*vec_it).X()).mont_repr.as_bytes(val);
        memcpy(x_val + i*esize, val, esize);
        memset(val, 0x0, esize);
        ((*vec_it).Y()).mont_repr.as_bytes(val);
        memcpy(y_val + i*esize, val, esize);
        memset(val, 0x0, esize);
        ((*vec_it).Z()).mont_repr.as_bytes(val);
        memcpy(z_val + i*esize, val, esize);
        memset(val, 0x0, esize);
        (*scalar_it).as_bigint().as_bytes(val);
        memcpy(scalar_val + i*esize, val, esize);
        i ++;
    }
    mnt4_g1_do_calc_np_sigma(size, scalar_val, x_val, y_val, z_val, x3, y3, z3);
    delete x_val;
    delete y_val;
    delete z_val;
    delete scalar_val;
    delete val;
    bigint<mnt4753_q_limbs> bigint_x, bigint_y, bigint_z;
    setBigData(&bigint_x, x3, esize);
    setBigData(&bigint_y, y3, esize);
    setBigData(&bigint_z, z3, esize);
#if 0
    bigint_x.print_hex();
    bigint_y.print_hex();
    bigint_z.print_hex();
#endif
    G result = G(bigint_x, bigint_y, bigint_z);
#if 0
    result.X().mont_repr.print_hex();
    result.Y().mont_repr.print_hex();
    result.Z().mont_repr.print_hex();
#endif
    return result;
}

template<typename G, typename Fr>
G multiexpG2(typename std::vector<Fr>::const_iterator scalar_start,
           typename std::vector<G>::const_iterator g_start,
           size_t length) {
    typename std::vector<G>::const_iterator vec_start = g_start;
    typename std::vector<G>::const_iterator vec_end = g_start + length;
    typename std::vector<Fr>::const_iterator scalar_end = scalar_start + length;
    typename std::vector<G>::const_iterator vec_it;
    typename std::vector<Fr>::const_iterator scalar_it;
    G acc = G::zero();
    int size = length;//vec_end - vec_start;
    int esize = 96;
    int total_size = size*esize;
    uint8_t *x0_val = new uint8_t[total_size];
    uint8_t *x1_val = new uint8_t[total_size];
    uint8_t *y0_val = new uint8_t[total_size];
    uint8_t *y1_val = new uint8_t[total_size];
    uint8_t *z0_val = new uint8_t[total_size];
    uint8_t *z1_val = new uint8_t[total_size];
    uint8_t *scalar_val = new uint8_t[total_size];
    memset(x0_val, 0x0, total_size);
    memset(x1_val, 0x0, total_size);
    memset(y0_val, 0x0, total_size);
    memset(y1_val, 0x0, total_size);
    memset(z0_val, 0x0, total_size);
    memset(z1_val, 0x0, total_size);
    memset(scalar_val, 0x0, total_size);
    uint8_t x30[esize];
    uint8_t x31[esize];
    uint8_t y30[esize];
    uint8_t y31[esize];
    uint8_t z30[esize];
    uint8_t z31[esize];
    uint8_t *val = new uint8_t[esize];

    int i = 0;
    for (vec_it = vec_start, scalar_it = scalar_start; vec_it != vec_end; ++vec_it, ++scalar_it)
    {
        memset(val, 0x0, esize);
        ((*vec_it).X()).c0.mont_repr.as_bytes(val);
        memcpy(x0_val + i*esize, val, esize);
        memset(val, 0x0, esize);
        ((*vec_it).X()).c1.mont_repr.as_bytes(val);
        memcpy(x1_val + i*esize, val, esize);
        memset(val, 0x0, esize);
        ((*vec_it).Y()).c0.mont_repr.as_bytes(val);
        memcpy(y0_val + i*esize, val, esize);
        memset(val, 0x0, esize);
        ((*vec_it).Y()).c1.mont_repr.as_bytes(val);
        memcpy(y1_val + i*esize, val, esize);
        memset(val, 0x0, esize);
        ((*vec_it).Z()).c0.mont_repr.as_bytes(val);
        memcpy(z0_val + i*esize, val, esize);
        memset(val, 0x0, esize);
        ((*vec_it).Z()).c1.mont_repr.as_bytes(val);
        memcpy(z1_val + i*esize, val, esize);
        memset(val, 0x0, esize);
        (*scalar_it).as_bigint().as_bytes(val);
        memcpy(scalar_val + i*esize, val, esize);
        i ++;
    }

    mnt4_g2_do_calc_np_sigma(size, scalar_val, x0_val, x1_val, y0_val, y1_val, z0_val, z1_val, x30, x31, y30, y31, z30, z31);
    delete x0_val;
    delete x1_val;
    delete y0_val;
    delete y1_val;
    delete z0_val;
    delete z1_val;
    delete val;
    bigint<mnt4753_q_limbs> bigint_x0, bigint_x1, bigint_y0, bigint_y1, bigint_z0, bigint_z1;
    setBigData(&bigint_x0, x30, esize);
    setBigData(&bigint_x1, x31, esize);
    setBigData(&bigint_y0, y30, esize);
    setBigData(&bigint_y1, y31, esize);
    setBigData(&bigint_z0, z30, esize);
    setBigData(&bigint_z1, z31, esize);
#if 0
    bigint_x0.print_hex();
    bigint_x1.print_hex();
    bigint_y0.print_hex();
    bigint_y1.print_hex();
    bigint_z0.print_hex();
    bigint_z1.print_hex();
#endif
    mnt4753_Fq2 bigint_x = mnt4753_Fq2(bigint_x0, bigint_x1);
    mnt4753_Fq2 bigint_y = mnt4753_Fq2(bigint_y0, bigint_y1);
    mnt4753_Fq2 bigint_z = mnt4753_Fq2(bigint_z0, bigint_z1);
    G result = G(bigint_x, bigint_y, bigint_z);
#if 0
    result.X().c0.mont_repr.print_hex();
    result.X().c1.mont_repr.print_hex();
    result.Y().c0.mont_repr.print_hex();
    result.Y().c1.mont_repr.print_hex();
    result.Z().c0.mont_repr.print_hex();
    result.Z().c1.mont_repr.print_hex();
#endif
    return result;
}

template<typename ppT>
int mnt4_prove(
    const char* params_path,
    const char* input_path,
    const char* output_path)
{
    time_t start_time, finish_time;
    time(&start_time);
    ppT::init_public_params();

    const size_t primary_input_size = 1;

    const groth16_parameters<ppT> parameters(params_path);
    const groth16_input<ppT> input(input_path, parameters.d, parameters.m);

    std::vector<Fr<ppT>> w  = std::move(input.w);
    std::vector<Fr<ppT>> ca = std::move(input.ca);
    std::vector<Fr<ppT>> cb = std::move(input.cb);
    std::vector<Fr<ppT>> cc = std::move(input.cc);

    // End reading of parameters and input

    libff::enter_block("Call to r1cs_gg_ppzksnark_prover");

    std::vector<Fr<ppT>> coefficients_for_H = compute_H<ppT>(
        parameters.d,
        ca, cb, cc);

    libff::enter_block("Compute the proof");
    libff::enter_block("Multi-exponentiations");

// Now the 5 multi-exponentiations
    printf("compute evaluation_At\n");
    G1<ppT> evaluation_At = multiexpGPU<G1<ppT>, Fr<ppT>>(
        w.begin(), parameters.A.begin(), parameters.m + 1);
#if 1
    printf("compute evaluation_Bt1\n");
    G1<ppT> evaluation_Bt1 = multiexpGPU<G1<ppT>, Fr<ppT>>(
        w.begin(), parameters.B1.begin(), parameters.m + 1);

    printf("compute evaluation_Bt2\n");
    G2<ppT> evaluation_Bt2 = multiexpG2<G2<ppT>, Fr<ppT>>(
        w.begin(), parameters.B2.begin(), parameters.m + 1);

    printf("compute evaluation_Ht\n");
    G1<ppT> evaluation_Ht = multiexpGPU<G1<ppT>, Fr<ppT>>(
        coefficients_for_H.begin(), parameters.H.begin(), parameters.d);

    printf("compute evaluation_Lt\n");
    G1<ppT> evaluation_Lt = multiexpGPU<G1<ppT>, Fr<ppT>>(
        w.begin() + primary_input_size + 1,
        parameters.L.begin(),
        parameters.m - 1);

    libff::G1<ppT> C = evaluation_Ht + evaluation_Lt + input.r * evaluation_Bt1; /*+ s *  g1_A  - (r * s) * pk.delta_g1; */
#endif
    libff::leave_block("Multi-exponentiations");
    libff::leave_block("Compute the proof");
    libff::leave_block("Call to r1cs_gg_ppzksnark_prover");

    groth16_output<ppT> output(
      std::move(evaluation_At),
      std::move(evaluation_Bt2),
      std::move(C));

    output.write(output_path);

    time(&finish_time);
    int duration = finish_time - start_time;
    printf("GPU all cost %d seconds\n", duration);
    return 0;
}

template<typename G, typename Fr>
G multiexp6G1(typename std::vector<Fr>::const_iterator scalar_start,
           typename std::vector<G>::const_iterator g_start,
           size_t length) {
    typename std::vector<G>::const_iterator vec_start = g_start;
    typename std::vector<G>::const_iterator vec_end = g_start + length;
    typename std::vector<Fr>::const_iterator scalar_end = scalar_start + length;
    typename std::vector<G>::const_iterator vec_it;
    typename std::vector<Fr>::const_iterator scalar_it;
    G acc = G::zero();
    int size = length;//vec_end - vec_start;
    int esize = 96;
    int total_size = size*esize;
    uint8_t *x_val = new uint8_t[total_size];
    uint8_t *y_val = new uint8_t[total_size];
    uint8_t *z_val = new uint8_t[total_size];
    uint8_t *scalar_val = new uint8_t[total_size];
    memset(x_val, 0x0, total_size);
    memset(y_val, 0x0, total_size);
    memset(z_val, 0x0, total_size);
    memset(scalar_val, 0x0, total_size);
    uint8_t x3[esize];
    uint8_t y3[esize];
    uint8_t z3[esize];
    uint8_t *val = new uint8_t[esize];;
    int i = 0;
    for (vec_it = vec_start, scalar_it = scalar_start; vec_it != vec_end; ++vec_it, ++scalar_it)
    {
        memset(val, 0x0, esize);
        ((*vec_it).X()).mont_repr.as_bytes(val);
        memcpy(x_val + i*esize, val, esize);
        memset(val, 0x0, esize);
        ((*vec_it).Y()).mont_repr.as_bytes(val);
        memcpy(y_val + i*esize, val, esize);
        memset(val, 0x0, esize);
        ((*vec_it).Z()).mont_repr.as_bytes(val);
        memcpy(z_val + i*esize, val, esize);
        memset(val, 0x0, esize);
        (*scalar_it).as_bigint().as_bytes(val);
        memcpy(scalar_val + i*esize, val, esize);
        i ++;
    }
    mnt6_g1_do_calc_np_sigma(size, scalar_val, x_val, y_val, z_val, x3, y3, z3);
    delete x_val;
    delete y_val;
    delete z_val;
    delete scalar_val;
    delete val;
    bigint<mnt6753_q_limbs> bigint_x, bigint_y, bigint_z;
    setBigData(&bigint_x, x3, esize);
    setBigData(&bigint_y, y3, esize);
    setBigData(&bigint_z, z3, esize);
#if 0
    bigint_x.print_hex();
    bigint_y.print_hex();
    bigint_z.print_hex();
#endif
    G result = G(bigint_x, bigint_y, bigint_z);
#if 0
    result.X().mont_repr.print_hex();
    result.Y().mont_repr.print_hex();
    result.Z().mont_repr.print_hex();
#endif
    return result;
}

template<typename G, typename Fr>
G multiexp6G2(typename std::vector<Fr>::const_iterator scalar_start,
           typename std::vector<G>::const_iterator g_start,
           size_t length) {
    typename std::vector<G>::const_iterator vec_start = g_start;
    typename std::vector<G>::const_iterator vec_end = g_start + length;
    typename std::vector<Fr>::const_iterator scalar_end = scalar_start + length;
    typename std::vector<G>::const_iterator vec_it;
    typename std::vector<Fr>::const_iterator scalar_it;
    G acc = G::zero();
    int size = length;//vec_end - vec_start;
    int esize = 96;
    int total_size = size*esize;
    int thr_total_size = 3 * total_size;
    uint8_t *x_val = new uint8_t[thr_total_size];
    uint8_t *y_val = new uint8_t[thr_total_size];
    uint8_t *z_val = new uint8_t[thr_total_size];
    memset(x_val, 0x0, thr_total_size);
    memset(y_val, 0x0, thr_total_size);
    memset(z_val, 0x0, thr_total_size);
    uint8_t *x0_val = x_val;
    uint8_t *x1_val = x_val + total_size;
    uint8_t *x2_val = x_val + 2*total_size;
    uint8_t *y0_val = y_val;
    uint8_t *y1_val = y_val + total_size;
    uint8_t *y2_val = y_val + 2*total_size;
    uint8_t *z0_val = z_val;
    uint8_t *z1_val = z_val + total_size;
    uint8_t *z2_val = z_val + 2*total_size;
    uint8_t *scalar_val = new uint8_t[total_size];
    memset(scalar_val, 0x0, total_size);
    uint8_t x3[3*esize];
    uint8_t y3[3*esize];
    uint8_t z3[3*esize];
    uint8_t *val = new uint8_t[esize];
    int i = 0;
    for (vec_it = vec_start, scalar_it = scalar_start; vec_it != vec_end; ++vec_it, ++scalar_it)
    {
        memset(val, 0x0, esize);
        ((*vec_it).X()).c0.mont_repr.as_bytes(val);
        memcpy(x0_val + i*esize, val, esize);
        memset(val, 0x0, esize);
        ((*vec_it).X()).c1.mont_repr.as_bytes(val);
        memcpy(x1_val + i*esize, val, esize);
        memset(val, 0x0, esize);
        ((*vec_it).X()).c2.mont_repr.as_bytes(val);
        memcpy(x2_val + i*esize, val, esize);
        memset(val, 0x0, esize);
        ((*vec_it).Y()).c0.mont_repr.as_bytes(val);
        memcpy(y0_val + i*esize, val, esize);
        memset(val, 0x0, esize);
        ((*vec_it).Y()).c1.mont_repr.as_bytes(val);
        memcpy(y1_val + i*esize, val, esize);
        memset(val, 0x0, esize);
        ((*vec_it).Y()).c2.mont_repr.as_bytes(val);
        memcpy(y2_val + i*esize, val, esize);
        memset(val, 0x0, esize);
        ((*vec_it).Z()).c0.mont_repr.as_bytes(val);
        memcpy(z0_val + i*esize, val, esize);
        memset(val, 0x0, esize);
        ((*vec_it).Z()).c1.mont_repr.as_bytes(val);
        memcpy(z1_val + i*esize, val, esize);
        memset(val, 0x0, esize);
        ((*vec_it).Z()).c2.mont_repr.as_bytes(val);
        memcpy(z2_val + i*esize, val, esize);
        memset(val, 0x0, esize);
        (*scalar_it).as_bigint().as_bytes(val);
        memcpy(scalar_val + i*esize, val, esize);
        i ++;
    }
#if 0
    printf("x input:");
    for (int i = 0 ; i < thr_total_size; i ++) {
        printf("%x", x_val[i]);
        if ((i+1)%esize == 0)  printf("\t");
        if ((i+1)%total_size == 0)  printf("\n");
    }
    printf("\ny input:");
    for (int i = 0 ; i < thr_total_size; i ++) {
        printf("%x", y_val[i]);
        if ((i+1)%esize == 0)  printf("\t");
        if ((i+1)%total_size == 0)  printf("\n");
    }
    printf("\nz input:");
    for (int i = 0 ; i < thr_total_size; i ++) {
        printf("%x", z_val[i]);
        if ((i+1)%esize == 0)  printf("\t");
        if ((i+1)%total_size == 0)  printf("\n");
    }
    printf("\n");
#endif

    mnt6_g2_do_calc_np_sigma(size, scalar_val, x_val, y_val, z_val, x3, y3, z3);
    delete x_val;
    delete y_val;
    delete z_val;
    delete val;
    bigint<mnt6753_q_limbs> bigint_x0, bigint_x1, bigint_x2, bigint_y0, bigint_y1, bigint_y2, bigint_z0, bigint_z1, bigint_z2;
    setBigData(&bigint_x0, x3, esize);
    setBigData(&bigint_x1, x3+esize, esize);
    setBigData(&bigint_x2, x3+2*esize, esize);
    setBigData(&bigint_y0, y3, esize);
    setBigData(&bigint_y1, y3+esize, esize);
    setBigData(&bigint_y2, y3+2*esize, esize);
    setBigData(&bigint_z0, z3, esize);
    setBigData(&bigint_z1, z3+esize, esize);
    setBigData(&bigint_z2, z3+2*esize, esize);
#if 0
    bigint_x0.print_hex();
    bigint_x1.print_hex();
    bigint_x2.print_hex();
    bigint_y0.print_hex();
    bigint_y1.print_hex();
    bigint_y2.print_hex();
    bigint_z0.print_hex();
    bigint_z1.print_hex();
    bigint_z2.print_hex();
#endif
    mnt6753_Fq3 bigint_x = mnt6753_Fq3(bigint_x0, bigint_x1, bigint_x2);
    mnt6753_Fq3 bigint_y = mnt6753_Fq3(bigint_y0, bigint_y1, bigint_y2);
    mnt6753_Fq3 bigint_z = mnt6753_Fq3(bigint_z0, bigint_z1, bigint_z2);
    G result = G(bigint_x, bigint_y, bigint_z);
#if 0
    result.X().c0.mont_repr.print_hex();
    result.X().c1.mont_repr.print_hex();
    result.X().c2.mont_repr.print_hex();
    result.Y().c0.mont_repr.print_hex();
    result.Y().c1.mont_repr.print_hex();
    result.Y().c2.mont_repr.print_hex();
    result.Z().c0.mont_repr.print_hex();
    result.Z().c1.mont_repr.print_hex();
    result.Z().c2.mont_repr.print_hex();
#endif
    return result;
}

template<typename ppT>
int mnt6_prove(
    const char* params_path,
    const char* input_path,
    const char* output_path)
{
    time_t start_time, finish_time;
    time(&start_time);
    ppT::init_public_params();

    const size_t primary_input_size = 1;

    const groth16_parameters<ppT> parameters(params_path);
    const groth16_input<ppT> input(input_path, parameters.d, parameters.m);

    std::vector<Fr<ppT>> w  = std::move(input.w);
    std::vector<Fr<ppT>> ca = std::move(input.ca);
    std::vector<Fr<ppT>> cb = std::move(input.cb);
    std::vector<Fr<ppT>> cc = std::move(input.cc);

    // End reading of parameters and input

    libff::enter_block("Call to r1cs_gg_ppzksnark_prover");

    std::vector<Fr<ppT>> coefficients_for_H = compute_H<ppT>(
        parameters.d,
        ca, cb, cc);

    libff::enter_block("Compute the proof");
    libff::enter_block("Multi-exponentiations");

// Now the 5 multi-exponentiations
    printf("compute evaluation_At\n");
    G1<ppT> evaluation_At = multiexp6G1<G1<ppT>, Fr<ppT>>(
        w.begin(), parameters.A.begin(), parameters.m + 1);

    printf("compute evaluation_Bt1\n");
    G1<ppT> evaluation_Bt1 = multiexp6G1<G1<ppT>, Fr<ppT>>(
        w.begin(), parameters.B1.begin(), parameters.m + 1);

    printf("compute evaluation_Bt2\n");
    G2<ppT> evaluation_Bt2 = multiexp6G2<G2<ppT>, Fr<ppT>>(
        w.begin(), parameters.B2.begin(), parameters.m + 1);

    printf("compute evaluation_Ht\n");
    G1<ppT> evaluation_Ht = multiexp6G1<G1<ppT>, Fr<ppT>>(
        coefficients_for_H.begin(), parameters.H.begin(), parameters.d);

    printf("compute evaluation_Lt\n");
    G1<ppT> evaluation_Lt = multiexp6G1<G1<ppT>, Fr<ppT>>(
        w.begin() + primary_input_size + 1,
        parameters.L.begin(),
        parameters.m - 1);

    libff::G1<ppT> C = evaluation_Ht + evaluation_Lt + input.r * evaluation_Bt1; /*+ s *  g1_A  - (r * s) * pk.delta_g1; */

    libff::leave_block("Multi-exponentiations");
    libff::leave_block("Compute the proof");
    libff::leave_block("Call to r1cs_gg_ppzksnark_prover");
    groth16_output<ppT> output(
      std::move(evaluation_At),
      std::move(evaluation_Bt2),
      std::move(C));

    output.write(output_path);

    time(&finish_time);
    int duration = finish_time - start_time;
    printf("GPU all cost %d seconds\n", duration);
    return 0;
}

int main(int argc, const char * argv[])
{
#if 0
  cudaDo(argc, argv);
  return 0;
#endif

  setbuf(stdout, NULL);
  std::string curve(argv[1]);
  std::string mode(argv[2]);
  std::string device(argv[3]);

  const char* params_path = argv[4];
  const char* input_path = argv[5];
  const char* output_path = argv[6];

  if (device == "GPU") {
    if (curve == "MNT4753") {
      if (mode == "compute") {
        return mnt4_prove<mnt4753_pp>(params_path, input_path, output_path);
      }
    } else if (curve == "MNT6753") {
      if (mode == "compute") {
        return mnt6_prove<mnt6753_pp>(params_path, input_path, output_path);
      }
    }
  } else if (device == "CPU") {
    if (curve == "MNT4753") {
        return run_prover<mnt4753_pp>(params_path, input_path, output_path);
    } else if (curve == "MNT6753") {
      if (mode == "compute") {
        return run_prover<mnt6753_pp>(params_path, input_path, output_path);
      }
    }
  } else {
    fprintf(stderr, "device code isn't part of this reference!");
  }
}

template<typename ppT>
void debug(
    Fr<ppT>& r,
    groth16_output<ppT>& output,
    std::vector<Fr<ppT>>& w) {

    const size_t primary_input_size = 1;

    std::vector<Fr<ppT>> primary_input(w.begin() + 1, w.begin() + 1 + primary_input_size);
    std::vector<Fr<ppT>> auxiliary_input(w.begin() + 1 + primary_input_size, w.end() );

    const libff::Fr<ppT> s = libff::Fr<ppT>::random_element();

    r1cs_gg_ppzksnark_proving_key<ppT> pk;
    std::ifstream pk_debug;
    pk_debug.open("proving-key.debug");
    pk_debug >> pk;

    /* A = alpha + sum_i(a_i*A_i(t)) + r*delta */
    libff::G1<ppT> g1_A = pk.alpha_g1 + output.A + r * pk.delta_g1;

    /* B = beta + sum_i(a_i*B_i(t)) + s*delta */
    libff::G2<ppT> g2_B = pk.beta_g2 + output.B + s * pk.delta_g2;

    /* C = sum_i(a_i*((beta*A_i(t) + alpha*B_i(t) + C_i(t)) + H(t)*Z(t))/delta) + A*s + r*b - r*s*delta */
    libff::G1<ppT> g1_C = output.C + s * g1_A + r * pk.beta_g1;

    libff::leave_block("Compute the proof");

    libff::leave_block("Call to r1cs_gg_ppzksnark_prover");

    r1cs_gg_ppzksnark_proof<ppT> proof = r1cs_gg_ppzksnark_proof<ppT>(std::move(g1_A), std::move(g2_B), std::move(g1_C));
    proof.print_size();

    r1cs_gg_ppzksnark_verification_key<ppT> vk;
    std::ifstream vk_debug;
    vk_debug.open("verification-key.debug");
    vk_debug >> vk;
    vk_debug.close();

    assert (r1cs_gg_ppzksnark_verifier_strong_IC<ppT>(vk, primary_input, proof) );

    r1cs_gg_ppzksnark_proof<ppT> proof1=
      r1cs_gg_ppzksnark_prover<ppT>(
          pk,
          primary_input,
          auxiliary_input);
    assert (r1cs_gg_ppzksnark_verifier_strong_IC<ppT>(vk, primary_input, proof1) );
}
