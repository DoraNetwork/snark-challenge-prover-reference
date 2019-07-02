/** @file
 *****************************************************************************

 Implementation of interfaces for multi-exponentiation routines.

 See multiexp.hpp .

 *****************************************************************************
 * @author     This file is part of libff, developed by SCIPR Lab
 *             and contributors (see AUTHORS).
 * @copyright  MIT license (see LICENSE file)
 *****************************************************************************/

#ifndef MULTIEXP_TCC_
#define MULTIEXP_TCC_

#include <algorithm>
#include <cassert>
#include <type_traits>

#include <libff/algebra/fields/bigint.hpp>
#include <libff/algebra/fields/fp_aux.tcc>
#include <libff/algebra/curves/mnt753/mnt4753/mnt4753_pp.hpp>
#include <libff/algebra/curves/mnt753/mnt6753/mnt6753_pp.hpp>
#include <libff/algebra/scalar_multiplication/multiexp.hpp>
#include <libff/algebra/scalar_multiplication/wnaf.hpp>
#include <libff/common/profiling.hpp>
#include <libff/common/utils.hpp>
#include "cuda-fixnum/main.cuh"

namespace libff {

void setBigDataG1(bigint<mnt4753_q_limbs> *bigintV, uint8_t *val, int size) {
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
void calcMultiPQPlusExt(typename std::vector<G>::iterator pq_start,
                        typename std::vector<int>::const_iterator p_idx_start,
                        typename std::vector<int>::const_iterator q_idx_start,
                        size_t size)
{
    if (size == 0) {
        return;
    }

//    size = 1;
//#define CPU_DO_CALC
#ifdef CPU_DO_CALC
    typename std::vector<G> plusPQsRef;
    plusPQsRef.reserve(size);

    typename std::vector<int>::const_iterator p_cpu_idx_it = p_idx_start;
    typename std::vector<int>::const_iterator q_cpu_idx_it = q_idx_start;

    for (; p_cpu_idx_it != p_idx_start + size && q_cpu_idx_it != q_idx_start + size; p_cpu_idx_it++, q_cpu_idx_it++) {

//#define DEBUG_PQ
#ifdef DEBUG_PQ
        printf("------------------ P ---------------------\n");
        (*(pq_start + *p_cpu_idx_it)).print();
        printf("------------------ Q ---------------------\n");
        (*(pq_start + *q_cpu_idx_it)).print();
#endif
        //*(pq_start + *p_cpu_idx_it) = *(pq_start + *p_cpu_idx_it) + *(pq_start + *q_cpu_idx_it);

#if 1
        plusPQsRef.push_back(*(pq_start + *p_cpu_idx_it) + *(pq_start + *q_cpu_idx_it));
#endif
    }

//#define DEBUG_CPU_OUTPUT
#ifdef DEBUG_CPU_OUTPUT
    G tmpRRef = G::zero();
    for(typename std::vector<G>::iterator pqPlusRef_it = plusPQsRef.begin(); pqPlusRef_it != plusPQsRef.end(); ++pqPlusRef_it){
        tmpRRef = tmpRRef + *pqPlusRef_it;
    }
    printf("------------------ CPU sum %d P+Q ---------------------\n", size);
    tmpRRef.print();
    printf("--------------------------------------------------\n");
#endif

    return;
#else //GPU do calc

    typename std::vector<G*> plusPQs;
    plusPQs.reserve(size);

    typename std::vector<int>::const_iterator p_idx_it = p_idx_start;
    typename std::vector<int>::const_iterator q_idx_it = q_idx_start;

    int ptBufCnt = size*6;//include x1/x2/x3/y1/y2/y3/ pointers
    uint8_t** xyz = new uint8_t*[ptBufCnt];
    uint8_t **x1 = xyz,       **y1 = x1 + size, **z1 = y1 + size;
    uint8_t **x2 = z1 + size, **y2 = x2 + size, **z2 = y2 + size;

    int i = 0;
    std::vector<int>::const_iterator p_idx_end = p_idx_start + size;
    std::vector<int>::const_iterator q_idx_end = q_idx_start + size;
    for (; p_idx_it != p_idx_end && q_idx_it != q_idx_end; ++p_idx_it, ++q_idx_it, ++i)
    {
        typename std::vector<G>::const_iterator p_it = pq_start + *p_idx_it;
        typename std::vector<G>::const_iterator q_it = pq_start + *q_idx_it;
        x1[i] = (uint8_t*)((p_it->X()).mont_repr.data);
        y1[i] = (uint8_t*)((p_it->Y()).mont_repr.data);
        z1[i] = (uint8_t*)((p_it->Z()).mont_repr.data);
        //setup Q
        x2[i] = (uint8_t*)((q_it->X()).mont_repr.data);
        y2[i] = (uint8_t*)((q_it->Y()).mont_repr.data);
        z2[i] = (uint8_t*)((q_it->Z()).mont_repr.data);
    }

#if 0
    {
        int esize = 96;
        bigint<mnt4753_q_limbs> bigint_x, bigint_y, bigint_z;
        size_t offset = 0;
        setBigDataG1(&bigint_x, x1[0], esize);
        setBigDataG1(&bigint_y, y1[0], esize);
        setBigDataG1(&bigint_z, z1[0], esize);
        printf("******************************* GPU P ********************************\n");
        bigint_x.print_hex();
        bigint_y.print_hex();
        bigint_z.print_hex();
        printf("******************************* GPU Q ********************************\n");
        typename std::vector<G>::const_iterator q_it = pq_start + *q_idx_start;
        setBigDataG1(&bigint_x, x2[0], esize);
        setBigDataG1(&bigint_y, y2[0], esize);
        setBigDataG1(&bigint_z, z2[0], esize);
        bigint_x.print_hex();
        bigint_y.print_hex();
        bigint_z.print_hex();
        printf("**********************************************************************\n");
    }
#endif
    int esize = 96;
    //uint8_t* dbg = new uint8_t[size*esize*3];
    uint8_t* dbg = nullptr;
    mnt4_g1_pq_plus_ext_adv(size, x1, y1, z1, x2, y2, z2, dbg);

#if 0
    for (i=0;i<size;i++){
        bigint<mnt4753_q_limbs> bigint_x, bigint_y, bigint_z;
        size_t offset = i*esize;
        setBigDataG1(&bigint_x, x1[i], esize);
        setBigDataG1(&bigint_y, y1[i], esize);
        setBigDataG1(&bigint_z, z1[i], esize);
        G* pR = new G(bigint_x, bigint_y, bigint_z);
        plusPQs.emplace_back(pR);
//        bigint_x.print_hex();
//        bigint_y.print_hex();
//        bigint_z.print_hex();
    }

    uint8_t* x3 = dbg, *y3=x3+size*esize, *z3=y3+size*esize;
    for (i=0;i<size;i++){
        bigint<mnt4753_q_limbs> bigint_x, bigint_y, bigint_z;
        size_t offset = i*esize;
        setBigDataG1(&bigint_x, x3, esize);
        setBigDataG1(&bigint_y, y3, esize);
        setBigDataG1(&bigint_z, z3, esize);
        G* pR = new G(bigint_x, bigint_y, bigint_z);
        printf("%%%%%%%%%%%%%%%%%%%%%%%%%%%% retieve all %%%%%%%%%%%%%%%%%%%%%%\n");
        pR->print();
        printf("%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%\n");
//        bigint_x.print_hex();
//        bigint_y.print_hex();
//        bigint_z.print_hex();
    }


//#define DBG_RESULT
#ifdef DBG_RESULT
    G tmpR = G::zero();
    typename std::vector<G*>::iterator pqPlus_it = plusPQs.begin();
    for(; pqPlus_it != plusPQs.end(); ++pqPlus_it){
        tmpR = tmpR + **pqPlus_it;
    }
    printf("------------------ GPU sum %d P+Q ---------------------\n", plusPQs.size());
    tmpR.print();
    printf("--------------------------------------------------\n");
#endif
#endif

    delete xyz;
//    delete dbg;

//    printf("------------------ GPU sum %d P+Q ---------------------\n", size);
    return;
#endif
}

template<typename G, typename Fr>
G calcMultiexpG1(typename std::vector<Fr>::const_iterator scalar_start,
           typename std::vector<G>::const_iterator g_start,
           size_t length) {
    if (length == 0) {
        return G::zero();
    }

    typename std::vector<G>::const_iterator vec_start = g_start;
    typename std::vector<G>::const_iterator vec_end = g_start + length;
    typename std::vector<Fr>::const_iterator scalar_end = scalar_start + length;
    typename std::vector<G>::const_iterator vec_it;
    typename std::vector<Fr>::const_iterator scalar_it;
    G acc = G::zero();
    int size = length;//vec_end - vec_start;
    int esize = 96;
    int total_size = size*esize;
    uint8_t *xyz = new uint8_t[total_size * 3];
    uint8_t *x_val = xyz;
    uint8_t *y_val = xyz + total_size;
    uint8_t *z_val = xyz + total_size*2;
    uint8_t *scalar_val = new uint8_t[total_size];
    //memset(xyz, 0x0, total_size*3);
    //memset(scalar_val, 0x0, total_size);

    uint8_t x3[esize];
    uint8_t y3[esize];
    uint8_t z3[esize];
    uint8_t *val = new uint8_t[esize];;

    int i = 0;
    for (vec_it = vec_start, scalar_it = scalar_start; vec_it != vec_end; ++vec_it, ++scalar_it)
    {
        ((*vec_it).X()).mont_repr.as_bytes(val);
        memcpy(x_val + i*esize, val, esize);

        ((*vec_it).Y()).mont_repr.as_bytes(val);
        memcpy(y_val + i*esize, val, esize);

        ((*vec_it).Z()).mont_repr.as_bytes(val);
        memcpy(z_val + i*esize, val, esize);

        (*scalar_it).as_bigint().as_bytes(val);
        memcpy(scalar_val + i*esize, val, esize);
        i ++;
    }
    //do_calc_np_sigma(size, scalar_val, x_val, y_val, z_val, x3, y3, z3);
    mnt4_g1_do_calc_np_sigma(size, scalar_val, x_val, y_val, z_val, x3, y3, z3);
    delete xyz;
    delete scalar_val;
    delete val;
    bigint<mnt4753_q_limbs> bigint_x, bigint_y, bigint_z;
    setBigDataG1(&bigint_x, x3, esize);
    setBigDataG1(&bigint_y, y3, esize);
    setBigDataG1(&bigint_z, z3, esize);
//    bigint_x.print_hex();
//    bigint_y.print_hex();
//    bigint_z.print_hex();
    G result = G(bigint_x, bigint_y, bigint_z);
//    result.X().mont_repr.print_hex();
//    result.Y().mont_repr.print_hex();
//    result.Z().mont_repr.print_hex();
    return result;
}

template<mp_size_t n>
class ordered_exponent {
// to use std::push_heap and friends later
public:
    size_t idx;
    bigint<n> r;

    ordered_exponent(const size_t idx, const bigint<n> &r) : idx(idx), r(r) {};

    bool operator<(const ordered_exponent<n> &other) const
    {
#if defined(__x86_64__) && defined(USE_ASM)
        if (n == 3)
        {
            long res;
            __asm__
                ("// check for overflow           \n\t"
                 "mov $0, %[res]                  \n\t"
                 ADD_CMP(16)
                 ADD_CMP(8)
                 ADD_CMP(0)
                 "jmp done%=                      \n\t"
                 "subtract%=:                     \n\t"
                 "mov $1, %[res]                  \n\t"
                 "done%=:                         \n\t"
                 : [res] "=&r" (res)
                 : [A] "r" (other.r.data), [mod] "r" (this->r.data)
                 : "cc", "%rax");
            return res;
        }
        else if (n == 4)
        {
            long res;
            __asm__
                ("// check for overflow           \n\t"
                 "mov $0, %[res]                  \n\t"
                 ADD_CMP(24)
                 ADD_CMP(16)
                 ADD_CMP(8)
                 ADD_CMP(0)
                 "jmp done%=                      \n\t"
                 "subtract%=:                     \n\t"
                 "mov $1, %[res]                  \n\t"
                 "done%=:                         \n\t"
                 : [res] "=&r" (res)
                 : [A] "r" (other.r.data), [mod] "r" (this->r.data)
                 : "cc", "%rax");
            return res;
        }
        else if (n == 5)
        {
            long res;
            __asm__
                ("// check for overflow           \n\t"
                 "mov $0, %[res]                  \n\t"
                 ADD_CMP(32)
                 ADD_CMP(24)
                 ADD_CMP(16)
                 ADD_CMP(8)
                 ADD_CMP(0)
                 "jmp done%=                      \n\t"
                 "subtract%=:                     \n\t"
                 "mov $1, %[res]                  \n\t"
                 "done%=:                         \n\t"
                 : [res] "=&r" (res)
                 : [A] "r" (other.r.data), [mod] "r" (this->r.data)
                 : "cc", "%rax");
            return res;
        }
        else
#endif
        {
            return (mpn_cmp(this->r.data, other.r.data, n) < 0);
        }
    }
};

/**
 * multi_exp_inner<T, FieldT, Method>() implementes the specified
 * multiexponentiation method.
 * this implementation relies on some rather arcane template magic:
 * function templates cannot be partially specialized, so we cannot just write
 *     template<typename T, typename FieldT>
 *     T multi_exp_inner<T, FieldT, multi_exp_method_naive>
 * thus we resort to using std::enable_if. the basic idea is that *overloading*
 * is what's actually happening here, it's just that, for any given value of
 * Method, only one of the templates will be valid, and thus the correct
 * implementation will be used.
 */

template<typename T, typename FieldT, multi_exp_method Method,
    typename std::enable_if<(Method == multi_exp_method_naive), int>::type = 0>
T multi_exp_inner(
    typename std::vector<T>::const_iterator vec_start,
    typename std::vector<T>::const_iterator vec_end,
    typename std::vector<FieldT>::const_iterator scalar_start,
    typename std::vector<FieldT>::const_iterator scalar_end)
{
    T result(T::zero());

    typename std::vector<T>::const_iterator vec_it;
    typename std::vector<FieldT>::const_iterator scalar_it;

    for (vec_it = vec_start, scalar_it = scalar_start; vec_it != vec_end; ++vec_it, ++scalar_it)
    {
        bigint<FieldT::num_limbs> scalar_bigint = scalar_it->as_bigint();
        result = result + opt_window_wnaf_exp(*vec_it, scalar_bigint, scalar_bigint.num_bits());
    }
    assert(scalar_it == scalar_end);

    return result;
}

template<typename T, typename FieldT, multi_exp_method Method,
    typename std::enable_if<(Method == multi_exp_method_naive_plain), int>::type = 0>
T multi_exp_inner(
    typename std::vector<T>::const_iterator vec_start,
    typename std::vector<T>::const_iterator vec_end,
    typename std::vector<FieldT>::const_iterator scalar_start,
    typename std::vector<FieldT>::const_iterator scalar_end)
{
    T result(T::zero());

    typename std::vector<T>::const_iterator vec_it;
    typename std::vector<FieldT>::const_iterator scalar_it;

    for (vec_it = vec_start, scalar_it = scalar_start; vec_it != vec_end; ++vec_it, ++scalar_it)
    {
        result = result + (*scalar_it) * (*vec_it);
    }
    assert(scalar_it == scalar_end);

    return result;
}

template<typename T, typename FieldT, multi_exp_method Method,
    typename std::enable_if<(Method == multi_exp_method_BDLO12), int>::type = 0>
T multi_exp_inner(
    typename std::vector<T>::const_iterator bases,
    typename std::vector<T>::const_iterator bases_end,
    typename std::vector<FieldT>::const_iterator exponents,
    typename std::vector<FieldT>::const_iterator exponents_end)
{
    UNUSED(exponents_end);
    size_t length = bases_end - bases;

    // empirically, this seems to be a decent estimate of the optimal value of c
    size_t log2_length = log2(length);
    size_t c = log2_length - (log2_length / 3 - 2);

    const mp_size_t exp_num_limbs =
        std::remove_reference<decltype(*exponents)>::type::num_limbs;
    std::vector<bigint<exp_num_limbs> > bn_exponents(length);
    size_t num_bits = 0;

    for (size_t i = 0; i < length; i++)
    {
        bn_exponents[i] = exponents[i].as_bigint();
        num_bits = std::max(num_bits, bn_exponents[i].num_bits());
    }

    size_t num_groups = (num_bits + c - 1) / c;

    T result;
    bool result_nonzero = false;

    for (size_t k = num_groups - 1; k <= num_groups; k--)
    {
        if (result_nonzero)
        {
            for (size_t i = 0; i < c; i++)
            {
                result = result.dbl();
            }
        }

        std::vector<T> buckets(1 << c);
        std::vector<bool> bucket_nonzero(1 << c);

        for (size_t i = 0; i < length; i++)
        {
            size_t id = 0;
            for (size_t j = 0; j < c; j++)
            {
                if (bn_exponents[i].test_bit(k*c + j))
                {
                    id |= 1 << j;
                }
            }

            if (id == 0)
            {
                continue;
            }

            if (bucket_nonzero[id])
            {
#ifdef USE_MIXED_ADDITION
                buckets[id] = buckets[id].mixed_add(bases[i]);
#else
                buckets[id] = buckets[id] + bases[i];
#endif
            }
            else
            {
                buckets[id] = bases[i];
                bucket_nonzero[id] = true;
            }
        }

#ifdef USE_MIXED_ADDITION
        batch_to_special(buckets);
#endif

        T running_sum;
        bool running_sum_nonzero = false;

        for (size_t i = (1u << c) - 1; i > 0; i--)
        {
            if (bucket_nonzero[i])
            {
                if (running_sum_nonzero)
                {
#ifdef USE_MIXED_ADDITION
                    running_sum = running_sum.mixed_add(buckets[i]);
#else
                    running_sum = running_sum + buckets[i];
#endif
                }
                else
                {
                    running_sum = buckets[i];
                    running_sum_nonzero = true;
                }
            }

            if (running_sum_nonzero)
            {
                if (result_nonzero)
                {
                    result = result + running_sum;
                }
                else
                {
                    result = running_sum;
                    result_nonzero = true;
                }
            }
        }
    }

    return result;
}

template<typename T, typename FieldT, multi_exp_method Method,
    typename std::enable_if<(Method == multi_exp_method_bos_coster), int>::type = 0>
T multi_exp_inner(
    typename std::vector<T>::const_iterator vec_start,
    typename std::vector<T>::const_iterator vec_end,
    typename std::vector<FieldT>::const_iterator scalar_start,
    typename std::vector<FieldT>::const_iterator scalar_end)
{
    const mp_size_t n = std::remove_reference<decltype(*scalar_start)>::type::num_limbs;

    if (vec_start == vec_end)
    {
        return T::zero();
    }

    if (vec_start + 1 == vec_end)
    {
        return (*scalar_start)*(*vec_start);
    }

    std::vector<ordered_exponent<n> > opt_q;
    const size_t vec_len = scalar_end - scalar_start;
    const size_t odd_vec_len = (vec_len % 2 == 1 ? vec_len : vec_len + 1);
    opt_q.reserve(odd_vec_len);
    std::vector<T> g;
    g.reserve(odd_vec_len);

    typename std::vector<T>::const_iterator vec_it;
    typename std::vector<FieldT>::const_iterator scalar_it;
    size_t i;
    for (i=0, vec_it = vec_start, scalar_it = scalar_start; vec_it != vec_end; ++vec_it, ++scalar_it, ++i)
    {
        g.emplace_back(*vec_it);

        opt_q.emplace_back(ordered_exponent<n>(i, scalar_it->as_bigint()));
    }
    std::make_heap(opt_q.begin(),opt_q.end());
    assert(scalar_it == scalar_end);

    if (vec_len != odd_vec_len)
    {
        g.emplace_back(T::zero());
        opt_q.emplace_back(ordered_exponent<n>(odd_vec_len - 1, bigint<n>(0ul)));
    }
    assert(g.size() % 2 == 1);
    assert(opt_q.size() == g.size());

    T opt_result = T::zero();

    while (true)
    {
        ordered_exponent<n> &a = opt_q[0];
        ordered_exponent<n> &b = (opt_q[1] < opt_q[2] ? opt_q[2] : opt_q[1]);

        const size_t abits = a.r.num_bits();

        if (b.r.is_zero())
        {
            // opt_result = opt_result + (a.r * g[a.idx]);
            opt_result = opt_result + opt_window_wnaf_exp(g[a.idx], a.r, abits);
            break;
        }

        const size_t bbits = b.r.num_bits();
        const size_t limit = (abits-bbits >= 20 ? 20 : abits-bbits);

        if (bbits < 1ul<<limit)
        {
            /*
              In this case, exponentiating to the power of a is cheaper than
              subtracting b from a multiple times, so let's do it directly
            */
            // opt_result = opt_result + (a.r * g[a.idx]);
            opt_result = opt_result + opt_window_wnaf_exp(g[a.idx], a.r, abits);
#ifdef DEBUG
            printf("Skipping the following pair (%zu bit number vs %zu bit):\n", abits, bbits);
            a.r.print();
            b.r.print();
#endif
            a.r.clear();
        }
        else
        {
            // x A + y B => (x-y) A + y (B+A)
            mpn_sub_n(a.r.data, a.r.data, b.r.data, n);
            g[b.idx] = g[b.idx] + g[a.idx];
        }

        // regardless of whether a was cleared or subtracted from we push it down, then take back up

        /* heapify A down */
        size_t a_pos = 0;
        while (2*a_pos + 2< odd_vec_len)
        {
            // this is a max-heap so to maintain a heap property we swap with the largest of the two
            if (opt_q[2*a_pos+1] < opt_q[2*a_pos+2])
            {
                std::swap(opt_q[a_pos], opt_q[2*a_pos+2]);
                a_pos = 2*a_pos+2;
            }
            else
            {
                std::swap(opt_q[a_pos], opt_q[2*a_pos+1]);
                a_pos = 2*a_pos+1;
            }
        }

        /* now heapify A up appropriate amount of times */
        while (a_pos > 0 && opt_q[(a_pos-1)/2] < opt_q[a_pos])
        {
            std::swap(opt_q[a_pos], opt_q[(a_pos-1)/2]);
            a_pos = (a_pos-1) / 2;
        }
    }

    return opt_result;
}

template<typename T, typename FieldT, multi_exp_method Method,
    typename std::enable_if<(Method == multi_exp_method_bos_coster_parallel), int>::type = 0>
T multi_exp_inner_parallel(
    typename std::vector<T>::const_iterator vec_start,
    typename std::vector<T>::const_iterator vec_end,
    typename std::vector<FieldT>::const_iterator scalar_start,
    typename std::vector<FieldT>::const_iterator scalar_end)
{
    const mp_size_t n = std::remove_reference<decltype(*scalar_start)>::type::num_limbs;

    if (vec_start == vec_end)
    {
        return T::zero();
    }

    if (vec_start + 1 == vec_end)
    {
        return (*scalar_start)*(*vec_start);
    }

    std::vector<ordered_exponent<n> > opt_q;
    const size_t vec_len = scalar_end - scalar_start;
    const size_t even_vec_len = (vec_len % 2 == 1 ? vec_len + 1 : vec_len);
    opt_q.reserve(even_vec_len);
    std::vector<T> g;
    g.reserve(even_vec_len);

    typename std::vector<T>::const_iterator vec_it;
    typename std::vector<FieldT>::const_iterator scalar_it;
    size_t i;
    for (i=0, vec_it = vec_start, scalar_it = scalar_start; vec_it != vec_end; ++vec_it, ++scalar_it, ++i)
    {
        g.emplace_back(*vec_it);
        opt_q.emplace_back(ordered_exponent<n>(i, scalar_it->as_bigint()));
    }
    assert(scalar_it == scalar_end);

    if (vec_len != even_vec_len)
    {
        g.emplace_back(T::zero());
        opt_q.emplace_back(ordered_exponent<n>(even_vec_len - 1, bigint<n>(0ul)));
    }

    std::sort(opt_q.begin(),opt_q.end());

    assert(g.size() % 2 == 0);
    assert(opt_q.size() == g.size());

    T opt_result = T::zero();
    printf("-------------------- run parallel algorithm --------------------\n");

#if 0
    {
        typename std::vector<ordered_exponent<n> >::iterator iter = opt_q.begin();
        int segment = 2;
        printf("-------------------- seg begin = %d --------------------\n", segment);
        while (iter != opt_q.end()) {
            typename std::vector<T> aPPoints;
            typename std::vector<FieldT> aPScalars;

            for (int i = 0; i < segment && iter != opt_q.end(); i++, iter++){
                aPScalars.emplace_back(iter->r);
                aPPoints.push_back(g[iter->idx]);
            }
            opt_result = opt_result + calcMultiexpG1<T, FieldT>(aPScalars.begin(), aPPoints.begin(), aPScalars.size());
            segment <<=2;
        }
        return opt_result;
    }
#endif

    int loopCnt = 0;
    bigint<n> intlimit((unsigned long)0);
    ordered_exponent<n> boundlimit(0, intlimit);
    typename std::vector<ordered_exponent<n> >::iterator begin = opt_q.begin();
    typename std::vector<ordered_exponent<n> >::iterator start = opt_q.begin();
    typename std::vector<ordered_exponent<n> >::iterator end = opt_q.end();
    typename std::vector<ordered_exponent<n> >::iterator start0 = std::upper_bound(start, opt_q.end(), boundlimit);
    size_t len = opt_q.size();

    while (len > 1)
    {
        // aP array calculates sum(a_i * P_i), which is made by those
        // bbits < limits items
        typename std::vector<T> aPPoints;
        typename std::vector<FieldT> aPScalars;

        // P & Q array calculates sum(P_i + Q_i).Q
        std::vector<int> idxPs;
        std::vector<int> idxQs;
        // scan the inputs and generate aP, scalar array & P, Q array
        for (typename std::vector<ordered_exponent<n> >::iterator it = start0; it != opt_q.end() && it+1 != opt_q.end(); it+=2) {
            typename std::vector<ordered_exponent<n> >::iterator itNext = it+1;
            //*it = *it-*itNext;
            ordered_exponent<n>& a = *itNext;
            ordered_exponent<n>& b = *it;
            // update opt_q
            // mpn_sub_n(itNext->r.data, itNext->r.data, it->r.data, n);

            const size_t abits = a.r.num_bits();
            const size_t bbits = b.r.num_bits();
            const size_t limit = (abits-bbits >= 20 ? 20 : abits-bbits);

            if (bbits < 1ul<<limit)
            {
                // update a to rA and change r to 0, so next sort will
                // move this a to zero region and will be added to
                // opt_result eventually.
                // opt_result = opt_result + opt_window_wnaf_exp(g[a.idx], a.r, abits);
                // TODO: a.r should be scalar type to avoid constructor call
                aPScalars.emplace_back(a.r);
                aPPoints.push_back(g[a.idx]);
            }
            else
            {
                //g[b.idx] = g[b.idx] + g[a.idx];
                //g[b.idx] = G1AddPQ<T, FieldT>(g[b.idx], g[a.idx]);
                // bP + aQ = b(P+Q) + (a-b)P. (a>b
                idxPs.push_back(b.idx);
                idxQs.push_back(a.idx);
            }
        }

#define ENABLE_GPU 1
#if ENABLE_GPU
        // give tasks to GPU and do a few CPU works.
        if (aPScalars.size() >= 1) {
            aPScalars.emplace_back(bigint<n>(1ul));
            aPPoints.push_back(opt_result);
            opt_result = calcMultiexpG1<T, FieldT>(aPScalars.begin(), aPPoints.begin(), aPScalars.size());
        }
        calcMultiPQPlusExt<T, FieldT>(g.begin(), idxPs.begin(), idxQs.begin(), idxPs.size());
#endif

        for (typename std::vector<ordered_exponent<n> >::iterator it = start0; it != opt_q.end() && it+1 != opt_q.end(); it+=2) {
            typename std::vector<ordered_exponent<n> >::iterator itNext = it+1;
            //*it = *it-*itNext;
            ordered_exponent<n>& a = *itNext;
            ordered_exponent<n>& b = *it;
            // update opt_q
            // mpn_sub_n(itNext->r.data, itNext->r.data, it->r.data, n);

            const size_t abits = a.r.num_bits();
            const size_t bbits = b.r.num_bits();
            const size_t limit = (abits-bbits >= 20 ? 20 : abits-bbits);

            if (bbits < 1ul<<limit)
            {
                // update a to rA and change r to 0, so next sort will
                // move this a to zero region and will be added to
                // opt_result eventually.
#if !ENABLE_GPU
                opt_result = opt_result + opt_window_wnaf_exp(g[a.idx], a.r, abits);
#endif
                a.r.clear();
            }
            else
            {
                // x A + y B => (x-y) A + y (B+A)
                mpn_sub_n(a.r.data, a.r.data, b.r.data, n);
#if !ENABLE_GPU
                g[b.idx] = g[a.idx]+g[b.idx];
#endif
            }
        }

        std::sort(start, opt_q.end());
        start = start0;
        start0 = std::upper_bound(start, opt_q.end(), boundlimit);
        len = std::distance(start, opt_q.end());
    }

    ordered_exponent<n> &a = opt_q[opt_q.size()-1];
    const size_t abits = a.r.num_bits();
    // opt_result = opt_result + (a.r * g[a.idx]);
    opt_result = opt_result + opt_window_wnaf_exp(g[a.idx], a.r, abits);

    return opt_result;
}

template<typename T, typename FieldT, multi_exp_method Method>
T multi_exp(typename std::vector<T>::const_iterator vec_start,
            typename std::vector<T>::const_iterator vec_end,
            typename std::vector<FieldT>::const_iterator scalar_start,
            typename std::vector<FieldT>::const_iterator scalar_end,
            const size_t chunks)
{
    const size_t total = vec_end - vec_start;
    if ((total < chunks) || (chunks == 1))
    {
        // no need to split into "chunks", can call implementation directly
        return multi_exp_inner<T, FieldT, Method>(
            vec_start, vec_end, scalar_start, scalar_end);
    }

    const size_t one = total/chunks;

    std::vector<T> partial(chunks, T::zero());

#ifdef MULTICORE
#pragma omp parallel for
#endif
    for (size_t i = 0; i < chunks; ++i)
    {
        partial[i] = multi_exp_inner<T, FieldT, Method>(
             vec_start + i*one,
             (i == chunks-1 ? vec_end : vec_start + (i+1)*one),
             scalar_start + i*one,
             (i == chunks-1 ? scalar_end : scalar_start + (i+1)*one));
    }

    T final = T::zero();

    for (size_t i = 0; i < chunks; ++i)
    {
        final = final + partial[i];
    }

    return final;
}

template<typename T, typename FieldT, multi_exp_method Method>
T multi_exp_gpu_parallel(typename std::vector<T>::const_iterator vec_start,
            typename std::vector<T>::const_iterator vec_end,
            typename std::vector<FieldT>::const_iterator scalar_start,
            typename std::vector<FieldT>::const_iterator scalar_end,
            const size_t chunks)
{
    const size_t total = vec_end - vec_start;
    if ((total < chunks) || (chunks == 1))
    {
        // no need to split into "chunks", can call implementation directly
        return multi_exp_inner_parallel<T, FieldT, Method>(
            vec_start, vec_end, scalar_start, scalar_end);
    }

    const size_t one = total/chunks;

    std::vector<T> partial(chunks, T::zero());

#ifdef MULTICORE
#pragma omp parallel for
#endif
    for (size_t i = 0; i < chunks; ++i)
    {
        partial[i] = multi_exp_inner_parallel<T, FieldT, Method>(
             vec_start + i*one,
             (i == chunks-1 ? vec_end : vec_start + (i+1)*one),
             scalar_start + i*one,
             (i == chunks-1 ? scalar_end : scalar_start + (i+1)*one));
    }

    T final = T::zero();

    for (size_t i = 0; i < chunks; ++i)
    {
        final = final + partial[i];
    }

    return final;
}


template<typename T, typename FieldT, multi_exp_method Method>
T multi_exp_with_mixed_addition(typename std::vector<T>::const_iterator vec_start,
                                typename std::vector<T>::const_iterator vec_end,
                                typename std::vector<FieldT>::const_iterator scalar_start,
                                typename std::vector<FieldT>::const_iterator scalar_end,
                                const size_t chunks)
{
    assert(std::distance(vec_start, vec_end) == std::distance(scalar_start, scalar_end));
    enter_block("Process scalar vector");
    auto value_it = vec_start;
    auto scalar_it = scalar_start;

    const FieldT zero = FieldT::zero();
    const FieldT one = FieldT::one();
    std::vector<FieldT> p;
    std::vector<T> g;

    T acc = T::zero();

    size_t num_skip = 0;
    size_t num_add = 0;
    size_t num_other = 0;

    for (; scalar_it != scalar_end; ++scalar_it, ++value_it)
    {
        if (*scalar_it == zero)
        {
            // do nothing
            ++num_skip;
        }
        else if (*scalar_it == one)
        {
#ifdef USE_MIXED_ADDITION
            acc = acc.mixed_add(*value_it);
#else
            acc = acc + (*value_it);
#endif
            ++num_add;
        }
        else
        {
            p.emplace_back(*scalar_it);
            g.emplace_back(*value_it);
            ++num_other;
        }
    }
    print_indent(); printf("* Elements of w skipped: %zu (%0.2f%%)\n", num_skip, 100.*num_skip/(num_skip+num_add+num_other));
    print_indent(); printf("* Elements of w processed with special addition: %zu (%0.2f%%)\n", num_add, 100.*num_add/(num_skip+num_add+num_other));
    print_indent(); printf("* Elements of w remaining: %zu (%0.2f%%)\n", num_other, 100.*num_other/(num_skip+num_add+num_other));

    leave_block("Process scalar vector");

    return acc + multi_exp<T, FieldT, Method>(g.begin(), g.end(), p.begin(), p.end(), chunks);
}

template<typename T, typename FieldT, multi_exp_method Method>
T multi_exp_with_mixed_addition_gpu(typename std::vector<T>::const_iterator vec_start,
                                typename std::vector<T>::const_iterator vec_end,
                                typename std::vector<FieldT>::const_iterator scalar_start,
                                typename std::vector<FieldT>::const_iterator scalar_end,
                                const size_t chunks)
{
    assert(std::distance(vec_start, vec_end) == std::distance(scalar_start, scalar_end));
    enter_block("Process scalar vector");
    auto value_it = vec_start;
    auto scalar_it = scalar_start;

    const FieldT zero = FieldT::zero();
    const FieldT one = FieldT::one();
    std::vector<FieldT> p;
    std::vector<T> g;

    T acc = T::zero();

    size_t num_skip = 0;
    size_t num_add = 0;
    size_t num_other = 0;

    for (; scalar_it != scalar_end; ++scalar_it, ++value_it)
    {
        if (*scalar_it == zero)
        {
            // do nothing
            ++num_skip;
        }
        else if (*scalar_it == one)
        {
#ifdef USE_MIXED_ADDITION
            acc = acc.mixed_add(*value_it);
#else
            acc = acc + (*value_it);
#endif
            ++num_add;
        }
        else
        {
            p.emplace_back(*scalar_it);
            g.emplace_back(*value_it);
            ++num_other;
        }
    }
    print_indent(); printf("* Elements of w skipped: %zu (%0.2f%%)\n", num_skip, 100.*num_skip/(num_skip+num_add+num_other));
    print_indent(); printf("* Elements of w processed with special addition: %zu (%0.2f%%)\n", num_add, 100.*num_add/(num_skip+num_add+num_other));
    print_indent(); printf("* Elements of w remaining: %zu (%0.2f%%)\n", num_other, 100.*num_other/(num_skip+num_add+num_other));

    leave_block("Process scalar vector");

    return acc + multi_exp_gpu_parallel<T, FieldT, multi_exp_method_bos_coster_parallel>(g.begin(), g.end(), p.begin(), p.end(), chunks);
}

template <typename T>
T inner_product(typename std::vector<T>::const_iterator a_start,
                typename std::vector<T>::const_iterator a_end,
                typename std::vector<T>::const_iterator b_start,
                typename std::vector<T>::const_iterator b_end)
{
    return multi_exp<T, T, multi_exp_method_naive_plain>(
        a_start, a_end,
        b_start, b_end, 1);
}

template<typename T>
size_t get_exp_window_size(const size_t num_scalars)
{
    if (T::fixed_base_exp_window_table.empty())
    {
#ifdef LOWMEM
        return 14;
#else
        return 17;
#endif
    }
    size_t window = 1;
    for (long i = T::fixed_base_exp_window_table.size()-1; i >= 0; --i)
    {
#ifdef DEBUG
        if (!inhibit_profiling_info)
        {
            printf("%ld %zu %zu\n", i, num_scalars, T::fixed_base_exp_window_table[i]);
        }
#endif
        if (T::fixed_base_exp_window_table[i] != 0 && num_scalars >= T::fixed_base_exp_window_table[i])
        {
            window = i+1;
            break;
        }
    }

    if (!inhibit_profiling_info)
    {
        print_indent(); printf("Choosing window size %zu for %zu elements\n", window, num_scalars);
    }

#ifdef LOWMEM
    window = std::min((size_t)14, window);
#endif
    return window;
}

template<typename T>
window_table<T> get_window_table(const size_t scalar_size,
                                 const size_t window,
                                 const T &g)
{
    const size_t in_window = 1ul<<window;
    const size_t outerc = (scalar_size+window-1)/window;
    const size_t last_in_window = 1ul<<(scalar_size - (outerc-1)*window);
#ifdef DEBUG
    if (!inhibit_profiling_info)
    {
        print_indent(); printf("* scalar_size=%zu; window=%zu; in_window=%zu; outerc=%zu\n", scalar_size, window, in_window, outerc);
    }
#endif

    window_table<T> powers_of_g(outerc, std::vector<T>(in_window, T::zero()));

    T gouter = g;

    for (size_t outer = 0; outer < outerc; ++outer)
    {
        T ginner = T::zero();
        size_t cur_in_window = outer == outerc-1 ? last_in_window : in_window;
        for (size_t inner = 0; inner < cur_in_window; ++inner)
        {
            powers_of_g[outer][inner] = ginner;
            ginner = ginner + gouter;
        }

        for (size_t i = 0; i < window; ++i)
        {
            gouter = gouter + gouter;
        }
    }

    return powers_of_g;
}

template<typename T, typename FieldT>
T windowed_exp(const size_t scalar_size,
               const size_t window,
               const window_table<T> &powers_of_g,
               const FieldT &pow)
{
    const size_t outerc = (scalar_size+window-1)/window;
    const bigint<FieldT::num_limbs> pow_val = pow.as_bigint();

    /* exp */
    T res = powers_of_g[0][0];

    for (size_t outer = 0; outer < outerc; ++outer)
    {
        size_t inner = 0;
        for (size_t i = 0; i < window; ++i)
        {
            if (pow_val.test_bit(outer*window + i))
            {
                inner |= 1u << i;
            }
        }

        res = res + powers_of_g[outer][inner];
    }

    return res;
}

template<typename T, typename FieldT>
std::vector<T> batch_exp(const size_t scalar_size,
                         const size_t window,
                         const window_table<T> &table,
                         const std::vector<FieldT> &v)
{
    if (!inhibit_profiling_info)
    {
        print_indent();
    }
    std::vector<T> res(v.size(), table[0][0]);

#ifdef MULTICORE
#pragma omp parallel for
#endif
    for (size_t i = 0; i < v.size(); ++i)
    {
        res[i] = windowed_exp(scalar_size, window, table, v[i]);

        if (!inhibit_profiling_info && (i % 10000 == 0))
        {
            printf(".");
            fflush(stdout);
        }
    }

    if (!inhibit_profiling_info)
    {
        printf(" DONE!\n");
    }

    return res;
}

template<typename T, typename FieldT>
std::vector<T> batch_exp_with_coeff(const size_t scalar_size,
                                    const size_t window,
                                    const window_table<T> &table,
                                    const FieldT &coeff,
                                    const std::vector<FieldT> &v)
{
    if (!inhibit_profiling_info)
    {
        print_indent();
    }
    std::vector<T> res(v.size(), table[0][0]);

#ifdef MULTICORE
#pragma omp parallel for
#endif
    for (size_t i = 0; i < v.size(); ++i)
    {
        res[i] = windowed_exp(scalar_size, window, table, coeff * v[i]);

        if (!inhibit_profiling_info && (i % 10000 == 0))
        {
            printf(".");
            fflush(stdout);
        }
    }

    if (!inhibit_profiling_info)
    {
        printf(" DONE!\n");
    }

    return res;
}

template<typename T>
void batch_to_special(std::vector<T> &vec)
{
    enter_block("Batch-convert elements to special form");

    std::vector<T> non_zero_vec;
    for (size_t i = 0; i < vec.size(); ++i)
    {
        if (!vec[i].is_zero())
        {
            non_zero_vec.emplace_back(vec[i]);
        }
    }

    T::batch_to_special_all_non_zeros(non_zero_vec);
    auto it = non_zero_vec.begin();
    T zero_special = T::zero();
    zero_special.to_special();

    for (size_t i = 0; i < vec.size(); ++i)
    {
        if (!vec[i].is_zero())
        {
            vec[i] = *it;
            ++it;
        }
        else
        {
            vec[i] = zero_special;
        }
    }
    leave_block("Batch-convert elements to special form");
}

} // libff

#endif // MULTIEXP_TCC_
