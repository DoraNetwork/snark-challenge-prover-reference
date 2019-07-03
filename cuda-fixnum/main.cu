#include <cstdio>
#include <cstring>
#include <cassert>

#include "fixnum/warp_fixnum.cu"
#include "array/fixnum_array.h"
#include "functions/modexp.cu"
#include "functions/multi_modexp.cu"
#include "modnum/modnum_monty_redc.cu"
#include "modnum/modnum_monty_cios.cu"
#include "mnt4_g1.cu"
#include "mnt4_g2.cu"
#include "mnt6_g1.cu"
#include "mnt6_g2.cu"

using namespace std;
using namespace cuFIXNUM;
using namespace MNT_G;
int BLOCK_NUM = 4096;
#define MNT_SIZE (96)
#define PARALLEL_SIGMA
#define MNT4 1
#define MNT6 2
#define WARP_SIZE  32
#define WARP_DATA_WIDTH 24

#if 1
#define printf(fmt, ...) (0)
#endif

// mnt4_q
const uint8_t mnt4_modulus[MNT_SIZE] = {1,128,94,36,222,99,144,94,159,17,221,44,82,84,157,227,240,37,196,154,113,16,136,99,164,84,114,118,233,204,90,104,56,126,83,203,165,13,15,184,157,5,24,242,118,231,23,177,157,247,90,161,217,36,209,153,141,237,160,232,37,185,253,7,115,216,151,108,249,232,183,94,237,175,143,91,80,151,249,183,173,205,226,238,34,144,34,16,17,196,146,45,198,196,1,0};

__device__ uint8_t mnt4_modulus_d[MNT_SIZE] = {1,128,94,36,222,99,144,94,159,17,221,44,82,84,157,227,240,37,196,154,113,16,136,99,164,84,114,118,233,204,90,104,56,126,83,203,165,13,15,184,157,5,24,242,118,231,23,177,157,247,90,161,217,36,209,153,141,237,160,232,37,185,253,7,115,216,151,108,249,232,183,94,237,175,143,91,80,151,249,183,173,205,226,238,34,144,34,16,17,196,146,45,198,196,1,0};

// mnt6_q
uint8_t mnt6_modulus[MNT_SIZE] = {1,0,0,64,226,118,7,217,79,58,161,15,23,153,160,78,151,87,0,63,188,129,195,214,164,58,153,52,118,249,223,185,54,38,33,41,148,202,235,62,155,169,89,200,40,92,108,178,157,247,90,161,217,36,209,153,141,237,160,232,37,185,253,7,115,216,151,108,249,232,183,94,237,175,143,91,80,151,249,183,173,205,226,238,34,144,34,16,17,196,146,45,198,196,1,0};

__device__ uint8_t mnt6_modulus_d[MNT_SIZE] = {1,0,0,64,226,118,7,217,79,58,161,15,23,153,160,78,151,87,0,63,188,129,195,214,164,58,153,52,118,249,223,185,54,38,33,41,148,202,235,62,155,169,89,200,40,92,108,178,157,247,90,161,217,36,209,153,141,237,160,232,37,185,253,7,115,216,151,108,249,232,183,94,237,175,143,91,80,151,249,183,173,205,226,238,34,144,34,16,17,196,146,45,198,196,1,0};

template< typename fixnum >
__device__ void dump(fixnum n, int size) {
#if 0
	for (int i = 0; i < size; i++) {
		printf("DUMP [%d] %x\n", i, fixnum::get(n, i));
	}
#endif
	printf("dump [%d]=\%x\n", threadIdx.x, fixnum::get(n, threadIdx.x));
}

template< typename fixnum >
struct mnt4g1_pq_plus {
    __device__ void operator()(fixnum x1, fixnum y1, fixnum z1, fixnum x2, fixnum y2, fixnum z2, fixnum &x3, fixnum &y3, fixnum &z3) {
        typedef mnt4_g1<fixnum> mnt4g1;
        typedef modnum_monty_cios<fixnum> modnum;
        __shared__ uint32_t modulus_data[WARP_DATA_WIDTH];
        if (threadIdx.x < WARP_DATA_WIDTH) {
            modulus_data[threadIdx.x] = *((uint32_t *)mnt4_modulus_d + threadIdx.x);
        }

        __syncthreads();

        modnum m(*((fixnum *)modulus_data + threadIdx.x % 32));
        mnt4g1::pq_plus(m, x1, y1, z1, x2, y2, z2, x3, y3, z3);
  }
};

template< typename fixnum >
struct mnt4g2_pq_plus {
    __device__ void operator()(fixnum x10, fixnum x11, fixnum y10, fixnum y11, fixnum z10, fixnum z11, fixnum x20, fixnum x21, fixnum y20, fixnum y21, fixnum z20, fixnum z21, fixnum &x30, fixnum &x31, fixnum &y30, fixnum &y31, fixnum &z30, fixnum &z31) {
        typedef mnt4_g2<fixnum> mnt4g2;
        typedef modnum_monty_cios<fixnum> modnum;
        __shared__ uint32_t modulus_data[WARP_DATA_WIDTH];
        if (threadIdx.x < WARP_DATA_WIDTH) {
            modulus_data[threadIdx.x] = *((uint32_t *)mnt4_modulus_d + threadIdx.x);
        }

        __syncthreads();

        modnum m(*((fixnum *)modulus_data + threadIdx.x % 32));
        mnt4g2::pq_plus(m, x10, x11, y10, y11, z10, z11, x20, x21, y20, y21, z20, z21, x30, x31, y30, y31, z30, z31);
  }
};

template< typename fixnum >
struct mnt6g1_pq_plus {
    __device__ void operator()(fixnum x1, fixnum y1, fixnum z1, fixnum x2, fixnum y2, fixnum z2, fixnum &x3, fixnum &y3, fixnum &z3) {
        typedef mnt6_g1<fixnum> mnt6g1;
        typedef modnum_monty_cios<fixnum> modnum;
        __shared__ uint32_t modulus_data[WARP_DATA_WIDTH];
        if (threadIdx.x < WARP_DATA_WIDTH) {
            modulus_data[threadIdx.x] = *((uint32_t *)mnt6_modulus_d + threadIdx.x);
        }

        __syncthreads();

        modnum m(*((fixnum *)modulus_data + threadIdx.x % 32));
        mnt6g1::pq_plus(m, x1, y1, z1, x2, y2, z2, x3, y3, z3);
  }
};

template< typename fixnum >
struct mnt6g2_pq_plus {
    __device__ void operator()(fixnum x10, fixnum x11, fixnum x12, fixnum y10, fixnum y11, fixnum y12, fixnum z10, fixnum z11, fixnum z12, fixnum x20, fixnum x21, fixnum x22, fixnum y20, fixnum y21, fixnum y22, fixnum z20, fixnum z21, fixnum z22, fixnum &x30, fixnum &x31, fixnum &x32, fixnum &y30, fixnum &y31, fixnum &y32, fixnum &z30, fixnum &z31, fixnum &z32) {
        typedef mnt6_g2<fixnum> mnt6g2;
        typedef modnum_monty_cios<fixnum> modnum;
        __shared__ uint32_t modulus_data[WARP_DATA_WIDTH];
        if (threadIdx.x < WARP_DATA_WIDTH) {
            modulus_data[threadIdx.x] = *((uint32_t *)mnt6_modulus_d + threadIdx.x);
        }

        __syncthreads();

        modnum m(*((fixnum *)modulus_data + threadIdx.x % 32));
        mnt6g2::pq_plus(m, x10, x11, x12, y10, y11, y12, z10, z11, z12, x20, x21, x22, y20, y21, y22, z20, z21, z22, x30, x31, x32, y30, y31, y32, z30, z31, z32);
  }
};

template< typename fixnum >
struct mnt4g1_calc_np {
    __device__ void operator()(fixnum w, fixnum x1, fixnum y1, fixnum z1, fixnum &x3, fixnum &y3, fixnum &z3) {
    typedef modnum_monty_cios<fixnum> modnum;
    typedef mnt4_g1<fixnum> mnt4g1;
    fixnum rx, ry, rz;
    int i = 767;    //24*32 - 1;
    int j = 23;     // total 24 bytes, each 32bit
    bool found_one = false;

    __shared__ uint32_t modulus_data[WARP_DATA_WIDTH];
    if (threadIdx.x < WARP_DATA_WIDTH) {
        modulus_data[threadIdx.x] = *((uint32_t *)mnt4_modulus_d + threadIdx.x);
    }

    __syncthreads();

    modnum m(*((fixnum *)modulus_data + threadIdx.x % 32));
#if 0
    if (threadIdx.x > 23) {
        dump(w, 24);
        dump(x1, 24);
        dump(y1, 24);
        dump(z1, 24);
    }
#endif
    //while(fixnum::cmp(tempw, fixnum::zero()) && i >= 0) {
    size_t value;
    while(i >= 0) {
        if ((i+1) == (j+1)*32) {
            value = fixnum::get(w, j);
            j --;
        }
#if 0
        if (threadIdx.x > 23) {
            printf("i %d value[%d] %x\n", i, i/32, value);
        }
#endif
        if (found_one) {
            mnt4g1::p_double(m, rx, ry, rz, rx, ry, rz);
#if 0
            if (threadIdx.x > 23) {
            printf("double result\n");
            dump(rx, 24);
            }
#endif
        }
        if ((value)&(1<<i%32)) {
            if (found_one == false) {
                rx = x1;
                ry = y1;
                rz = z1;
            } else {
                mnt4g1::pq_plus(m, rx, ry, rz, x1, y1, z1, rx, ry, rz);
            }
#if 0
            if (threadIdx.x > 23) {
            printf("add result\n");
            dump(rx, 24);
            }
#endif
            found_one = true;
        }
        i --;
#if 0
        if (threadIdx.x > 23) {
        //if (count >20) break;
        }
#endif
    }
    x3 = rx;
    y3 = ry;
    z3 = rz;
#if 0
    printf("final result\n");
    dump(x3, 24);
    dump(y3, 24);
    dump(z3, 24);
#endif
  }
};

template< typename fixnum >
struct mnt4g2_calc_np {
    __device__ void operator()(fixnum w, fixnum x10, fixnum x11, fixnum y10, fixnum y11, fixnum z10, fixnum z11, fixnum &x30, fixnum &x31, fixnum &y30, fixnum &y31, fixnum &z30, fixnum &z31) {
    typedef modnum_monty_cios<fixnum> modnum;
    typedef mnt4_g2<fixnum> mnt4g2;

    __shared__ uint32_t modulus_data[WARP_DATA_WIDTH];
    if (threadIdx.x < WARP_DATA_WIDTH) {
        modulus_data[threadIdx.x] = *((uint32_t *)mnt4_modulus_d + threadIdx.x);
    }

    __syncthreads();

    modnum m(*((fixnum *)modulus_data + threadIdx.x % 32));

    int i = 767;    //24*32 - 1;
    int j = 23;     // total 24 bytes, each 32bit
    bool found_one = false;
    size_t value;
    while(i >= 0) {
        if ((i+1) == (j+1)*32) {
            value = fixnum::get(w, j);
            j --;
        }
        //printf("value[%d] is %x\n", i, value);
        if (found_one) {
            mnt4g2::p_double(m, x30, x31, y30, y31, z30, z31, x30, x31, y30, y31, z30, z31);
        }
        if ((value)&(1<<i%32)) {
            if (found_one == false) {
                x30 = x10;
                x31 = x11;
                y30 = y10;
                y31 = y11;
                z30 = z10;
                z31 = z11;
            } else {
                mnt4g2::pq_plus(m, x30, x31, y30, y31, z30, z31, x10, x11, y10, y11, z10, z11, x30, x31, y30, y31, z30, z31);
            }
            found_one = true;
        }
        i --;
        //if (count >= 50) break;
    }
  }
};

template< typename fixnum >
struct mnt6g1_calc_np {
    __device__ void operator()(fixnum w, fixnum x1, fixnum y1, fixnum z1, fixnum &x3, fixnum &y3, fixnum &z3) {
    typedef modnum_monty_cios<fixnum> modnum;
    typedef mnt6_g1<fixnum> mnt6g1;
    fixnum rx, ry, rz;
    int i = 24*32 - 1;
    bool found_one = false;

    __shared__ uint32_t modulus_data[WARP_DATA_WIDTH];
    if (threadIdx.x < WARP_DATA_WIDTH) {
        modulus_data[threadIdx.x] = *((uint32_t *)mnt6_modulus_d + threadIdx.x);
    }

    __syncthreads();

    modnum m(*((fixnum *)modulus_data + threadIdx.x % 32));
#if 0
    dump(w, 24);
    dump(x1, 24);
    dump(y1, 24);
    dump(z1, 24);
#endif
    while(i >= 0) {
        size_t value = fixnum::get(w, i/32);
        //printf("value[%d] is %x\n", i, value);
        if (found_one) {
            mnt6g1::p_double(m, rx, ry, rz, rx, ry, rz);
        }
        if ((value)&(1<<i%32)) {
            if (found_one == false) {
                rx = x1;
                ry = y1;
                rz = z1;
            } else {
                mnt6g1::pq_plus(m, rx, ry, rz, x1, y1, z1, rx, ry, rz);
            }
            found_one = true;
        }
        i --;
        //if (count >= 25) break;
    }
    x3 = rx;
    y3 = ry;
    z3 = rz;
  }
};

template< typename fixnum >
struct mnt6g2_calc_np {
    __device__ void operator()(fixnum w, fixnum x10, fixnum x11, fixnum x12, fixnum y10, fixnum y11, fixnum y12, fixnum z10, fixnum z11, fixnum z12, fixnum &x30, fixnum &x31, fixnum &x32, fixnum &y30, fixnum &y31, fixnum &y32, fixnum &z30, fixnum &z31, fixnum &z32) {
    typedef modnum_monty_cios<fixnum> modnum;
    typedef mnt6_g2<fixnum> mnt6g2;
    int i = 24*32 - 1;
    bool found_one = false;

    __shared__ uint32_t modulus_data[WARP_DATA_WIDTH];
    if (threadIdx.x < WARP_DATA_WIDTH) {
        modulus_data[threadIdx.x] = *((uint32_t *)mnt6_modulus_d + threadIdx.x);
    }

    __syncthreads();

    modnum m(*((fixnum *)modulus_data + threadIdx.x % 32));
#if 0
    dump(w, 24);
    dump(x10, 24);
    dump(y10, 24);
    dump(z10, 24);
#endif
    while(i >= 0) {
        size_t value = fixnum::get(w, i/32);
        //printf("value[%d] is %x\n", i, value);
        if (found_one) {
            mnt6g2::p_double(m, x30, x31, x32, y30, y31, y32, z30, z31, z32, x30, x31, x32, y30, y31, y32, z30, z31, z32);
        }
        if ((value)&(1<<i%32)) {
            if (found_one == false) {
                x30 = x10;
                x31 = x11;
                x32 = x12;
                y30 = y10;
                y31 = y11;
                y32 = y12;
                z30 = z10;
                z31 = z11;
                z32 = z12;
            } else {
                mnt6g2::pq_plus(m, x30, x31, x32, y30, y31, y32, z30, z31, z32, x10, x11, x12, y10, y11, y12, z10, z11, z12, x30, x31, x32, y30, y31, y32, z30, z31, z32);
            }
            found_one = true;
        }
        i --;
        //if (count >= 25) break;
    }
  }
};


int mnt4_g1_pq_plus(int n, uint8_t* x1, uint8_t* y1, uint8_t* z1, uint8_t* x2, uint8_t* y2, uint8_t* z2, uint8_t *x3, uint8_t *y3, uint8_t *z3) {
    typedef warp_fixnum<96, u32_fixnum> fixnum;
    typedef fixnum_array<fixnum> fixnum_array;
    fixnum_array *x1in, *y1in, *z1in, *x2in, *y2in, *z2in;
    fixnum_array *rx3, *ry3, *rz3;
    int fn_bytes = 96;
    int step_bytes = n * fn_bytes;
    
    x1in = fixnum_array::create(x1, step_bytes, fn_bytes);
    y1in = fixnum_array::create(y1, step_bytes, fn_bytes);
    z1in = fixnum_array::create(z1, step_bytes, fn_bytes);
    x2in = fixnum_array::create(x2, step_bytes, fn_bytes);
    y2in = fixnum_array::create(y2, step_bytes, fn_bytes);
    z2in = fixnum_array::create(z2, step_bytes, fn_bytes);

    rx3 = fixnum_array::create(n);
    ry3 = fixnum_array::create(n);
    rz3 = fixnum_array::create(n);
    fixnum_array::template map<mnt4g1_pq_plus>(x1in, y1in, z1in, x2in, y2in, z2in, rx3, ry3, rz3);

    int size = n; 
    rx3->retrieve_all(x3, step_bytes, &size);
    ry3->retrieve_all(y3, step_bytes, &size);
    rz3->retrieve_all(z3, step_bytes, &size);
   
    delete x1in; 
    delete y1in; 
    delete z1in; 
    delete x2in; 
    delete y2in; 
    delete z2in; 
    delete rx3;
    delete ry3;
    delete rz3;
    return 0;
}

template <template <typename> class PQ_plus>
inline void do_sigma(int nelts, uint8_t *x, uint8_t *y, uint8_t *z, uint8_t *rx, uint8_t *ry, uint8_t *rz) {
    typedef warp_fixnum<96, u32_fixnum> fixnum;
    typedef fixnum_array<fixnum> fixnum_array;
    fixnum_array *x1in, *y1in, *z1in, *x2in, *y2in, *z2in;
    fixnum_array *rx3, *ry3, *rz3;

    int half_bytes = MNT_SIZE*nelts/2;
    x1in = fixnum_array::create(x, half_bytes, MNT_SIZE);
    y1in = fixnum_array::create(y, half_bytes, MNT_SIZE);
    z1in = fixnum_array::create(z, half_bytes, MNT_SIZE);
    x2in = fixnum_array::create(x + half_bytes, half_bytes, MNT_SIZE);
    y2in = fixnum_array::create(y + half_bytes, half_bytes, MNT_SIZE);
    z2in = fixnum_array::create(z + half_bytes, half_bytes, MNT_SIZE);

    rx3 = fixnum_array::create(nelts/2);
    ry3 = fixnum_array::create(nelts/2);
    rz3 = fixnum_array::create(nelts/2);
    fixnum_array::template map<PQ_plus>(x1in, y1in, z1in, x2in, y2in, z2in, rx3, ry3, rz3);
    
    int size = nelts/2;
    rx3->retrieve_all(rx, half_bytes, &size);
    ry3->retrieve_all(ry, half_bytes, &size);
    rz3->retrieve_all(rz, half_bytes, &size);
    delete x1in;
    delete y1in;
    delete z1in;
    delete x2in;
    delete y2in;
    delete z2in;
    delete rx3;
    delete ry3;
    delete rz3;
}

inline void do_sigma(int nelts, uint8_t *x0, uint8_t *x1, uint8_t *x2, uint8_t *y0, uint8_t *y1, uint8_t *y2, uint8_t *z0, uint8_t *z1, uint8_t *z2, uint8_t *rx0, uint8_t *rx1, uint8_t *rx2, uint8_t *ry0, uint8_t *ry1, uint8_t *ry2, uint8_t *rz0, uint8_t *rz1, uint8_t *rz2) {
    typedef warp_fixnum<96, u32_fixnum> fixnum;
    typedef fixnum_array<fixnum> fixnum_array;
    fixnum_array *x10in, *x11in, *x12in, *y10in, *y11in, *y12in, *z10in, *z11in, *z12in, *x20in, *x21in, *x22in, *y20in, *y21in, *y22in, *z20in, *z21in, *z22in;
    fixnum_array *rx30, *rx31, *rx32, *ry30, *ry31, *ry32, *rz30, *rz31, *rz32;

    int half_bytes = MNT_SIZE*nelts/2;
    x10in = fixnum_array::create(x0, half_bytes, MNT_SIZE);
    x11in = fixnum_array::create(x1, half_bytes, MNT_SIZE);
    x12in = fixnum_array::create(x2, half_bytes, MNT_SIZE);
    y10in = fixnum_array::create(y0, half_bytes, MNT_SIZE);
    y11in = fixnum_array::create(y1, half_bytes, MNT_SIZE);
    y12in = fixnum_array::create(y2, half_bytes, MNT_SIZE);
    z10in = fixnum_array::create(z0, half_bytes, MNT_SIZE);
    z11in = fixnum_array::create(z1, half_bytes, MNT_SIZE);
    z12in = fixnum_array::create(z2, half_bytes, MNT_SIZE);
    x20in = fixnum_array::create(x0 + half_bytes, half_bytes, MNT_SIZE);
    x21in = fixnum_array::create(x1 + half_bytes, half_bytes, MNT_SIZE);
    x22in = fixnum_array::create(x2 + half_bytes, half_bytes, MNT_SIZE);
    y20in = fixnum_array::create(y0 + half_bytes, half_bytes, MNT_SIZE);
    y21in = fixnum_array::create(y1 + half_bytes, half_bytes, MNT_SIZE);
    y22in = fixnum_array::create(y2 + half_bytes, half_bytes, MNT_SIZE);
    z20in = fixnum_array::create(z0 + half_bytes, half_bytes, MNT_SIZE);
    z21in = fixnum_array::create(z1 + half_bytes, half_bytes, MNT_SIZE);
    z22in = fixnum_array::create(z2 + half_bytes, half_bytes, MNT_SIZE);

    rx30 = fixnum_array::create(nelts/2);
    rx31 = fixnum_array::create(nelts/2);
    rx32 = fixnum_array::create(nelts/2);
    ry30 = fixnum_array::create(nelts/2);
    ry31 = fixnum_array::create(nelts/2);
    ry32 = fixnum_array::create(nelts/2);
    rz30 = fixnum_array::create(nelts/2);
    rz31 = fixnum_array::create(nelts/2);
    rz32 = fixnum_array::create(nelts/2);
    fixnum_array::template map<mnt6g2_pq_plus>(x10in, x11in, x12in, y10in, y11in, y12in, z10in, z11in, z12in, x20in, x21in, x22in, y20in, y21in, y22in, z20in, z21in, z22in, rx30, rx31, rx32, ry30, ry31, ry32, rz30, rz31, rz32);
    
    int size = nelts/2;
    rx30->retrieve_all(rx0, half_bytes, &size);
    rx31->retrieve_all(rx1, half_bytes, &size);
    rx32->retrieve_all(rx2, half_bytes, &size);
    ry30->retrieve_all(ry0, half_bytes, &size);
    ry31->retrieve_all(ry1, half_bytes, &size);
    ry32->retrieve_all(ry2, half_bytes, &size);
    rz30->retrieve_all(rz0, half_bytes, &size);
    rz31->retrieve_all(rz1, half_bytes, &size);
    rz32->retrieve_all(rz2, half_bytes, &size);
    delete x10in;
    delete x11in;
    delete x12in;
    delete y10in;
    delete y11in;
    delete y12in;
    delete z10in;
    delete z11in;
    delete z12in;
    delete x20in;
    delete x21in;
    delete x22in;
    delete y20in;
    delete y21in;
    delete y22in;
    delete z20in;
    delete z21in;
    delete z22in;
    delete rx30;
    delete rx31;
    delete rx32;
    delete ry30;
    delete ry31;
    delete ry32;
    delete rz30;
    delete rz31;
    delete rz32;
}

template <template <typename> class Calc_np, template <typename> class PQ_plus>
int do_calc_np_sigma(int nelts, uint8_t* scalar, uint8_t* x1, uint8_t* y1, uint8_t* z1, uint8_t *x3, uint8_t *y3, uint8_t *z3) {
    clock_t start = clock();
    typedef warp_fixnum<96, u32_fixnum> fixnum;
    typedef fixnum_array<fixnum> fixnum_array;
    printf("calc do_calc_np_sigma\n");
    printf("nelts %d\n", nelts);
    int step = nelts;
    int size = nelts;
    int DATA_SIZE = 96;
    int fn_bytes = DATA_SIZE;
    int step_bytes = fn_bytes * step;
    uint8_t *x1bytes = x1;
    uint8_t *y1bytes = y1;
    uint8_t *z1bytes = z1;
    uint8_t *x3bytes = new uint8_t[step_bytes];
    uint8_t *y3bytes = new uint8_t[step_bytes];
    uint8_t *z3bytes = new uint8_t[step_bytes];
    fixnum_array *dx3, *dy3, *dz3, *x1in, *y1in, *z1in;
    fixnum_array *x2in, *y2in, *z2in;
#if 0
    printf("x:");
    for (int i = 0; i < step_bytes; i ++) {
        printf("%x", x1[i]);
        if ((i+1)%fn_bytes == 0) printf("\t");
    }
    printf("\ny:");
    for (int i = 0; i < step_bytes; i ++) {
        printf("%x", y1[i]);
        if ((i+1)%fn_bytes == 0) printf("\t");
    }
    printf("\nz:");
    for (int i = 0; i < step_bytes; i ++) {
        printf("%x", z1[i]);
        if ((i+1)%fn_bytes == 0) printf("\t");
    }
    printf("\nscalar:");
    for (int i = 0; i < step_bytes; i ++) {
        printf("%x", scalar[i]);
        if ((i+1)%fn_bytes == 0) printf("\t");
    }
    printf("\n");
#endif

    // scaler
    auto modulusw = fixnum_array::create(scalar, step_bytes, fn_bytes);
    
    // sigma result
    fixnum_array *rx3, *ry3, *rz3;
    int got_result = false;

    for (int i = 0; i < nelts; i+=step) {
        dx3 = fixnum_array::create(step);
        dy3 = fixnum_array::create(step);
        dz3 = fixnum_array::create(step);
        x1in = fixnum_array::create(x1bytes, step_bytes, fn_bytes);
        y1in = fixnum_array::create(y1bytes, step_bytes, fn_bytes);
        z1in = fixnum_array::create(z1bytes, step_bytes, fn_bytes);
        fixnum_array::template map<Calc_np>(modulusw, x1in, y1in, z1in, dx3, dy3, dz3);

        dx3->retrieve_all(x3bytes, step_bytes, &size);
        dy3->retrieve_all(y3bytes, step_bytes, &size);
        dz3->retrieve_all(z3bytes, step_bytes, &size);
#if 0
        printf("calc np result\n");
        printf("x3:");
        for (int i = 0; i < step_bytes; i ++) {
            printf("%x", x3bytes[i]);
            if ((i+1)%fn_bytes == 0) printf("\t");
        }
        printf("\ny3:");
        for (int i = 0; i < step_bytes; i ++) {
            printf("%x", y3bytes[i]);
            if ((i+1)%fn_bytes == 0) printf("\t");
        }
        printf("\nz3:");
        for (int i = 0; i < step_bytes; i ++) {
            printf("%x", z3bytes[i]);
            if ((i+1)%fn_bytes == 0) printf("\t");
        }
        printf("\n");
#endif
        if (nelts == 1) {
            memcpy(x3, x3bytes, step_bytes);
            memcpy(y3, y3bytes, step_bytes);
            memcpy(z3, z3bytes, step_bytes);
            got_result = true;
            delete x1in;
            delete y1in;
            delete z1in;
            delete dx3;
            delete dy3;
            delete dz3;
            break;
        }
#ifdef PARALLEL_SIGMA
        int start = nelts%2;
        int rnelts = nelts - start;
        uint8_t *rx, *ry, *rz;
        rx = new uint8_t[MNT_SIZE*rnelts/2];
        ry = new uint8_t[MNT_SIZE*rnelts/2];
        rz = new uint8_t[MNT_SIZE*rnelts/2];
        while(rnelts > 1) {
            do_sigma<PQ_plus>(rnelts, x3bytes + start*MNT_SIZE, y3bytes + start*MNT_SIZE, z3bytes + start*MNT_SIZE, rx, ry, rz);
            rnelts = rnelts >> 1;
            memcpy(x3bytes + start*MNT_SIZE, rx, rnelts*MNT_SIZE);
            memcpy(y3bytes + start*MNT_SIZE, ry, rnelts*MNT_SIZE);
            memcpy(z3bytes + start*MNT_SIZE, rz, rnelts*MNT_SIZE);
            if (rnelts > 1 && rnelts%2) {
                if (start == 0) {
                    start = 1;
                    rnelts -= 1;
                } else {
                    start = 0;
                    rnelts += 1;
                }
            }
        }
        delete rx;
        delete ry;
        delete rz;
        if (start == 1) {
            // add the first element
            x2in = fixnum_array::create(x3bytes, fn_bytes, fn_bytes);
            y2in = fixnum_array::create(y3bytes, fn_bytes, fn_bytes);
            z2in = fixnum_array::create(z3bytes, fn_bytes, fn_bytes);
            rx3 = fixnum_array::create(x3bytes + fn_bytes, fn_bytes, fn_bytes);
            ry3 = fixnum_array::create(y3bytes + fn_bytes, fn_bytes, fn_bytes);
            rz3 = fixnum_array::create(z3bytes + fn_bytes, fn_bytes, fn_bytes);
            fixnum_array::template map<PQ_plus>(x2in, y2in, z2in, rx3, ry3, rz3, rx3, ry3, rz3);
            delete x2in;
            delete y2in;
            delete z2in;
        } else {
            memcpy(x3, x3bytes, fn_bytes);
            memcpy(y3, y3bytes, fn_bytes);
            memcpy(z3, z3bytes, fn_bytes);
            got_result = true;
        }
#else
        bool result_set = false;
        // start add from second element
        int start = 1;
        if (i == 0) {
            rx3 = fixnum_array::create(x3bytes + start * fn_bytes, fn_bytes, fn_bytes);
            ry3 = fixnum_array::create(y3bytes + start * fn_bytes, fn_bytes, fn_bytes);
            rz3 = fixnum_array::create(z3bytes + start * fn_bytes, fn_bytes, fn_bytes);
            result_set = true;
        }
        int k = 0;
        if (result_set && i == 0) {
            k = start + 1;
        }
        for (; k < step; k ++)
        {
            x2in = fixnum_array::create(x3bytes + k * fn_bytes, fn_bytes, fn_bytes);
            y2in = fixnum_array::create(y3bytes + k * fn_bytes, fn_bytes, fn_bytes);
            z2in = fixnum_array::create(z3bytes + k * fn_bytes, fn_bytes, fn_bytes);
            fixnum_array::template map<PQ_plus>(rx3, ry3, rz3, x2in, y2in, z2in, rx3, ry3, rz3);
            delete x2in;
            delete y2in;
            delete z2in;
        }
        // add the first element
        x2in = fixnum_array::create(x3bytes, fn_bytes, fn_bytes);
        y2in = fixnum_array::create(y3bytes, fn_bytes, fn_bytes);
        z2in = fixnum_array::create(z3bytes, fn_bytes, fn_bytes);
        fixnum_array::template map<PQ_plus>(x2in, y2in, z2in, rx3, ry3, rz3, rx3, ry3, rz3);
        delete x2in;
        delete y2in;
        delete z2in;
#endif
        delete x1in;
        delete y1in;
        delete z1in;
        delete dx3;
        delete dy3;
        delete dz3;
    }
    if (!got_result) {
        size = 1;
        rx3->retrieve_all(x3, fn_bytes, &size);
        ry3->retrieve_all(y3, fn_bytes, &size);
        rz3->retrieve_all(z3, fn_bytes, &size);
        delete rx3;
        delete ry3;
        delete rz3;
    }
    delete x3bytes;
    delete y3bytes;
    delete z3bytes;

    printf("final result");
    printf("\nx3:");
    for (int k = fn_bytes-1; k >= 0; k--) {
        printf("%02x", x3[k]);
    }
    printf("\ny3:");
    for (int k = fn_bytes-1; k >= 0; k--) {
        printf("%02x", y3[k]);
    }
    printf("\nz3:");
    for (int k = fn_bytes-1; k >= 0; k--) {
       printf("%02x", z3[k]);
    }
    printf("\n");
    clock_t diff = clock() - start;
    printf("cost time %ld\n", diff);
    return 0;
}

int mnt4_g1_do_calc_np_sigma(int n, uint8_t* scalar, uint8_t* x1, uint8_t* y1, uint8_t* z1, uint8_t *x3, uint8_t *y3, uint8_t *z3) {
    do_calc_np_sigma<mnt4g1_calc_np, mnt4g1_pq_plus>(n, scalar, x1, y1, z1, x3, y3, z3);
    return 0;
}

int mnt6_g1_do_calc_np_sigma(int n, uint8_t* scalar, uint8_t* x1, uint8_t* y1, uint8_t* z1, uint8_t *x3, uint8_t *y3, uint8_t *z3) {
#if 0
    // test
    n = 1;
    int sn = 96;
    x1 += sn;
    y1 += sn;
    z1 += sn;
    scalar += sn;
#endif
    do_calc_np_sigma<mnt6g1_calc_np, mnt6g1_pq_plus>(n, scalar, x1, y1, z1, x3, y3, z3);
    return 0;
}

inline void do_sigma(int nelts, uint8_t *x0, uint8_t *x1, uint8_t *y0, uint8_t *y1, uint8_t *z0, uint8_t *z1, uint8_t *rx0, uint8_t *rx1, uint8_t *ry0, uint8_t *ry1, uint8_t *rz0, uint8_t *rz1) {
    typedef warp_fixnum<96, u32_fixnum> fixnum;
    typedef fixnum_array<fixnum> fixnum_array;
    fixnum_array *x10in, *x11in, *y10in, *y11in, *z10in, *z11in, *x20in, *x21in, *y20in, *y21in, *z20in, *z21in;
    fixnum_array *rx30, *rx31, *ry30, *ry31, *rz30, *rz31;

    int half_bytes = MNT_SIZE*nelts/2;
    x10in = fixnum_array::create(x0, half_bytes, MNT_SIZE);
    x11in = fixnum_array::create(x1, half_bytes, MNT_SIZE);
    y10in = fixnum_array::create(y0, half_bytes, MNT_SIZE);
    y11in = fixnum_array::create(y1, half_bytes, MNT_SIZE);
    z10in = fixnum_array::create(z0, half_bytes, MNT_SIZE);
    z11in = fixnum_array::create(z1, half_bytes, MNT_SIZE);
    x20in = fixnum_array::create(x0 + half_bytes, half_bytes, MNT_SIZE);
    x21in = fixnum_array::create(x1 + half_bytes, half_bytes, MNT_SIZE);
    y20in = fixnum_array::create(y0 + half_bytes, half_bytes, MNT_SIZE);
    y21in = fixnum_array::create(y1 + half_bytes, half_bytes, MNT_SIZE);
    z20in = fixnum_array::create(z0 + half_bytes, half_bytes, MNT_SIZE);
    z21in = fixnum_array::create(z1 + half_bytes, half_bytes, MNT_SIZE);

    rx30 = fixnum_array::create(nelts/2);
    rx31 = fixnum_array::create(nelts/2);
    ry30 = fixnum_array::create(nelts/2);
    ry31 = fixnum_array::create(nelts/2);
    rz30 = fixnum_array::create(nelts/2);
    rz31 = fixnum_array::create(nelts/2);
    fixnum_array::template map<mnt4g2_pq_plus>(x10in, x11in, y10in, y11in, z10in, z11in, x20in, x21in, y20in, y21in, z20in, z21in, rx30, rx31, ry30, ry31, rz30, rz31);
    
    int size = nelts/2;
    rx30->retrieve_all(rx0, half_bytes, &size);
    rx31->retrieve_all(rx1, half_bytes, &size);
    ry30->retrieve_all(ry0, half_bytes, &size);
    ry31->retrieve_all(ry1, half_bytes, &size);
    rz30->retrieve_all(rz0, half_bytes, &size);
    rz31->retrieve_all(rz1, half_bytes, &size);
    delete x10in;
    delete x11in;
    delete y10in;
    delete y11in;
    delete z10in;
    delete z11in;
    delete x20in;
    delete x21in;
    delete y20in;
    delete y21in;
    delete z20in;
    delete z21in;
    delete rx30;
    delete rx31;
    delete ry30;
    delete ry31;
    delete rz30;
    delete rz31;
}

void print_g2(uint8_t *x30, uint8_t *x31, uint8_t *y30, uint8_t *y31, uint8_t *z30, uint8_t *z31) {
    int fn_bytes = 96;
    printf("\nx3:");
    for (int k = fn_bytes-1; k >= 0; k--) {
        printf("%02x", x30[k]);
    }
    printf(",");
    for (int k = fn_bytes-1; k >= 0; k--) {
        printf("%02x", x31[k]);
    }
    printf("\ny3:");
    for (int k = fn_bytes-1; k >= 0; k--) {
        printf("%02x", y30[k]);
    }
    printf(",");
    for (int k = fn_bytes-1; k >= 0; k--) {
        printf("%02x", y31[k]);
    }
    printf("\nz3:");
    for (int k = fn_bytes-1; k >= 0; k--) {
       printf("%02x", z30[k]);
    }
    printf(",");
    for (int k = fn_bytes-1; k >= 0; k--) {
       printf("%02x", z31[k]);
    }
    printf("\n");

}

int mnt4_g2_do_calc_np_sigma(int nelts, uint8_t * scalar, uint8_t* x10, uint8_t* x11, uint8_t* y10, uint8_t* y11, uint8_t* z10, uint8_t* z11, uint8_t *x30, uint8_t *x31, uint8_t *y30, uint8_t *y31, uint8_t *z30, uint8_t *z31) {
    clock_t start = clock();
    typedef warp_fixnum<96, u32_fixnum> fixnum;
    typedef fixnum_array<fixnum> fixnum_array;
    printf("mnt4_g2_do_calc_np_sigma start nelts %d\n", nelts);
    // test
#if 0
    nelts = 1;
    int sn = 96;
    x10 += sn;
    x11 += sn;
    y10 += sn;
    y11 += sn;
    z10 += sn;
    z11 += sn;
    scalar += sn;
#endif
    int step = nelts;
    int size = nelts;
    int DATA_SIZE = 96;
    int fn_bytes = DATA_SIZE;
    int step_bytes = fn_bytes * step;
    uint8_t *x10bytes = x10;
    uint8_t *x11bytes = x11;
    uint8_t *y10bytes = y10;
    uint8_t *y11bytes = y11;
    uint8_t *z10bytes = z10;
    uint8_t *z11bytes = z11;
    uint8_t *x30bytes = new uint8_t[step_bytes];
    uint8_t *x31bytes = new uint8_t[step_bytes];
    uint8_t *y30bytes = new uint8_t[step_bytes];
    uint8_t *y31bytes = new uint8_t[step_bytes];
    uint8_t *z30bytes = new uint8_t[step_bytes];
    uint8_t *z31bytes = new uint8_t[step_bytes];
    fixnum_array *dx30, *dx31, *dy30, *dy31, *dz30, *dz31, *x10in, *x11in, *y10in, *y11in, *z10in, *z11in;
    fixnum_array *x20in, *x21in, *y20in, *y21in, *z20in, *z21in;
    // scalar
    auto modulusw = fixnum_array::create(scalar, step_bytes, fn_bytes);
    
    // sigma result
    fixnum_array *rx30, *rx31, *ry30, *ry31, *rz30, *rz31;
    int got_result = false;

#if 0
    printf("x10:\n");
    for (int i = 0; i < step_bytes; i ++) {
        printf("%02x", x10bytes[i]); 
        if ((i+1) % 96 == 0) {
            printf("\t");
        }
    }
    printf("\nx11:");
    for (int i = 0; i < step_bytes; i ++) {
        printf("%02x", x11bytes[i]); 
        if ((i+1) % 96 == 0) {
            printf("\t");
        }
    }
    printf("\ny10:");
    for (int i = 0; i < step_bytes; i ++) {
        printf("%02x", y10bytes[i]); 
        if ((i+1) % 96 == 0) {
            printf("\t");
        }
    }
    printf("\ny11:");
    for (int i = 0; i < step_bytes; i ++) {
        printf("%02x", y11bytes[i]); 
        if ((i+1) % 96 == 0) {
            printf("\t");
        }
    }
    printf("\nz10:");
    for (int i = 0; i < step_bytes; i ++) {
        printf("%02x", z10bytes[i]); 
        if ((i+1) % 96 == 0) {
            printf("\t");
        }
    }
    printf("\nz11:");
    for (int i = 0; i < step_bytes; i ++) {
        printf("%02x", z11bytes[i]); 
        if ((i+1) % 96 == 0) {
            printf("\t");
        }
    }
    printf("\nscalar:");
    for (int i = 0; i < step_bytes; i ++) {
        printf("%02x", scalar[i]); 
        if ((i+1) % 96 == 0) {
            printf("\t");
        }
    }
    printf("\n");
#endif
    for (int i = 0; i < nelts; i+=step) {
        dx30 = fixnum_array::create(step);
        dx31 = fixnum_array::create(step);
        dy30 = fixnum_array::create(step);
        dy31 = fixnum_array::create(step);
        dz30 = fixnum_array::create(step);
        dz31 = fixnum_array::create(step);
        x10in = fixnum_array::create(x10bytes, step_bytes, fn_bytes);
        x11in = fixnum_array::create(x11bytes, step_bytes, fn_bytes);
        y10in = fixnum_array::create(y10bytes, step_bytes, fn_bytes);
        y11in = fixnum_array::create(y11bytes, step_bytes, fn_bytes);
        z10in = fixnum_array::create(z10bytes, step_bytes, fn_bytes);
        z11in = fixnum_array::create(z11bytes, step_bytes, fn_bytes);
        fixnum_array::template map<mnt4g2_calc_np>(modulusw, x10in, x11in, y10in, y11in, z10in, z11in, dx30, dx31, dy30, dy31, dz30, dz31);

        dx30->retrieve_all(x30bytes, step_bytes, &size);
        dx31->retrieve_all(x31bytes, step_bytes, &size);
        dy30->retrieve_all(y30bytes, step_bytes, &size);
        dy31->retrieve_all(y31bytes, step_bytes, &size);
        dz30->retrieve_all(z30bytes, step_bytes, &size);
        dz31->retrieve_all(z31bytes, step_bytes, &size);
        delete x10in;
        delete x11in;
        delete y10in;
        delete y11in;
        delete z10in;
        delete z11in;
        delete dx30;
        delete dx31;
        delete dy30;
        delete dy31;
        delete dz30;
        delete dz31;
#ifdef PARALLEL_SIGMA
        int start = nelts%2;
        int rnelts = nelts - start;
        uint8_t *rx0, *rx1, *ry0, *ry1, *rz0, *rz1;
        rx0 = new uint8_t[MNT_SIZE*rnelts/2];
        rx1 = new uint8_t[MNT_SIZE*rnelts/2];
        ry0 = new uint8_t[MNT_SIZE*rnelts/2];
        ry1 = new uint8_t[MNT_SIZE*rnelts/2];
        rz0 = new uint8_t[MNT_SIZE*rnelts/2];
        rz1 = new uint8_t[MNT_SIZE*rnelts/2];
        while(rnelts > 1) {
            do_sigma(rnelts, x30bytes + start*MNT_SIZE, x31bytes + start*MNT_SIZE, y30bytes + start*MNT_SIZE, y31bytes + start*MNT_SIZE, z30bytes + start*MNT_SIZE, z31bytes + start*MNT_SIZE, rx0, rx1, ry0, ry1, rz0, rz1);
            rnelts = rnelts >> 1;
            memcpy(x30bytes + start*MNT_SIZE, rx0, rnelts*MNT_SIZE);
            memcpy(x31bytes + start*MNT_SIZE, rx1, rnelts*MNT_SIZE);
            memcpy(y30bytes + start*MNT_SIZE, ry0, rnelts*MNT_SIZE);
            memcpy(y31bytes + start*MNT_SIZE, ry1, rnelts*MNT_SIZE);
            memcpy(z30bytes + start*MNT_SIZE, rz0, rnelts*MNT_SIZE);
            memcpy(z31bytes + start*MNT_SIZE, rz1, rnelts*MNT_SIZE);
            if (rnelts > 1 && rnelts%2) {
                if (start == 0) {
                    start = 1;
                    rnelts -= 1;
                } else {
                    start = 0;
                    rnelts += 1;
                }
            }
        }
        delete rx0;
        delete rx1;
        delete ry0;
        delete ry1;
        delete rz0;
        delete rz1;
        if (start == 1) {
            // add the first element
            x20in = fixnum_array::create(x30bytes, fn_bytes, fn_bytes);
            x21in = fixnum_array::create(x31bytes, fn_bytes, fn_bytes);
            y20in = fixnum_array::create(y30bytes, fn_bytes, fn_bytes);
            y21in = fixnum_array::create(y31bytes, fn_bytes, fn_bytes);
            z20in = fixnum_array::create(z30bytes, fn_bytes, fn_bytes);
            z21in = fixnum_array::create(z31bytes, fn_bytes, fn_bytes);
            rx30 = fixnum_array::create(x30bytes + fn_bytes, fn_bytes, fn_bytes);
            rx31 = fixnum_array::create(x31bytes + fn_bytes, fn_bytes, fn_bytes);
            ry30 = fixnum_array::create(y30bytes + fn_bytes, fn_bytes, fn_bytes);
            ry31 = fixnum_array::create(y31bytes + fn_bytes, fn_bytes, fn_bytes);
            rz30 = fixnum_array::create(z30bytes + fn_bytes, fn_bytes, fn_bytes);
            rz31 = fixnum_array::create(z31bytes + fn_bytes, fn_bytes, fn_bytes);
            fixnum_array::template map<mnt4g2_pq_plus>(rx30, rx31, ry30, ry31, rz30, rz31, x20in, x21in, y20in, y21in, z20in, z21in, rx30, rx31, ry30, ry31, rz30, rz31);
            delete x20in;
            delete x21in;
            delete y20in;
            delete y21in;
            delete z20in;
            delete z21in;
        } else {
            memcpy(x30, x30bytes, fn_bytes);
            memcpy(x31, x31bytes, fn_bytes);
            memcpy(y30, y30bytes, fn_bytes);
            memcpy(y31, y31bytes, fn_bytes);
            memcpy(z30, z30bytes, fn_bytes);
            memcpy(z31, z31bytes, fn_bytes);
            got_result = true;
        }
#else
        bool result_set = false;
        // start add from second element
        int start = 1;
        if (i == 0) {
            //print_g2(x30bytes + start * fn_bytes, x31bytes + start * fn_bytes, y30bytes + start * fn_bytes, y31bytes + start * fn_bytes, z30bytes + start * fn_bytes, z31bytes + start * fn_bytes);
            rx30 = fixnum_array::create(x30bytes + start * fn_bytes, fn_bytes, fn_bytes);
            rx31 = fixnum_array::create(x31bytes + start * fn_bytes, fn_bytes, fn_bytes);
            ry30 = fixnum_array::create(y30bytes + start * fn_bytes, fn_bytes, fn_bytes);
            ry31 = fixnum_array::create(y31bytes + start * fn_bytes, fn_bytes, fn_bytes);
            rz30 = fixnum_array::create(z30bytes + start * fn_bytes, fn_bytes, fn_bytes);
            rz31 = fixnum_array::create(z31bytes + start * fn_bytes, fn_bytes, fn_bytes);
            result_set = true;
        }
        int k = 0;
        if (result_set && i == 0) {
            k = start + 1;
        }
        for (; k < step; k ++)
        {
            //print_g2(x30bytes + k * fn_bytes, x31bytes + k * fn_bytes, y30bytes + k * fn_bytes, y31bytes + k * fn_bytes, z30bytes + k * fn_bytes, z31bytes + k * fn_bytes);
            x20in = fixnum_array::create(x30bytes + k * fn_bytes, fn_bytes, fn_bytes);
            x21in = fixnum_array::create(x31bytes + k * fn_bytes, fn_bytes, fn_bytes);
            y20in = fixnum_array::create(y30bytes + k * fn_bytes, fn_bytes, fn_bytes);
            y21in = fixnum_array::create(y31bytes + k * fn_bytes, fn_bytes, fn_bytes);
            z20in = fixnum_array::create(z30bytes + k * fn_bytes, fn_bytes, fn_bytes);
            z21in = fixnum_array::create(z31bytes + k * fn_bytes, fn_bytes, fn_bytes);
            fixnum_array::template map<mnt4g2_pq_plus>(rx30, rx31, ry30, ry31, rz30, rz31, x20in, x21in, y20in, y21in, z20in, z21in, rx30, rx31, ry30, ry31, rz30, rz31);
            delete x20in;
            delete x21in;
            delete y20in;
            delete y21in;
            delete z20in;
            delete z21in;
#if 0
        rx30->retrieve_all(x30, fn_bytes, &size);
        rx31->retrieve_all(x31, fn_bytes, &size);
        ry30->retrieve_all(y30, fn_bytes, &size);
        ry31->retrieve_all(y31, fn_bytes, &size);
        rz30->retrieve_all(z30, fn_bytes, &size);
        rz31->retrieve_all(z31, fn_bytes, &size);
            print_g2(x30, x31, y30, y31, z30, z31);
#endif
        }
        // add the first element
        x20in = fixnum_array::create(x30bytes, fn_bytes, fn_bytes);
        x21in = fixnum_array::create(x31bytes, fn_bytes, fn_bytes);
        y20in = fixnum_array::create(y30bytes, fn_bytes, fn_bytes);
        y21in = fixnum_array::create(y31bytes, fn_bytes, fn_bytes);
        z20in = fixnum_array::create(z30bytes, fn_bytes, fn_bytes);
        z21in = fixnum_array::create(z31bytes, fn_bytes, fn_bytes);
        fixnum_array::template map<mnt4g2_pq_plus>(rx30, rx31, ry30, ry31, rz30, rz31, x20in, x21in, y20in, y21in, z20in, z21in, rx30, rx31, ry30, ry31, rz30, rz31);
        delete x20in;
        delete x21in;
        delete y20in;
        delete y21in;
        delete z20in;
        delete z21in;
#endif
    }
    if (!got_result) {
        size = 1;
        rx30->retrieve_all(x30, fn_bytes, &size);
        rx31->retrieve_all(x31, fn_bytes, &size);
        ry30->retrieve_all(y30, fn_bytes, &size);
        ry31->retrieve_all(y31, fn_bytes, &size);
        rz30->retrieve_all(z30, fn_bytes, &size);
        rz31->retrieve_all(z31, fn_bytes, &size);
        delete rx30;
        delete rx31;
        delete ry30;
        delete ry31;
        delete rz30;
        delete rz31;
    }
    delete x30bytes;
    delete x31bytes;
    delete y30bytes;
    delete y31bytes;
    delete z30bytes;
    delete z31bytes;
    delete modulusw;

    printf("mnt4_g2 final result");
    print_g2(x30, x31, y30, y31, z30, z31);
    clock_t diff = clock() - start;
    printf("cost time %ld\n", diff);
    return 0;
}


int mnt6_g2_do_calc_np_sigma(int nelts, uint8_t * scalar, uint8_t* x, uint8_t* y, uint8_t* z, uint8_t *x3, uint8_t *y3, uint8_t *z3) {
    clock_t start = clock();
    typedef warp_fixnum<96, u32_fixnum> fixnum;
    typedef fixnum_array<fixnum> fixnum_array;
    printf("mnt6_g2_do_calc_np_sigma start nelts %d\n", nelts);
    // test
    int sn = 0;
    int skip = 1;
#if 0
    skip = nelts; 
    nelts = 1;
    sn = 96;
    scalar += sn;
#endif
    int step = nelts;
    int size = nelts;
    int DATA_SIZE = 96;
    int fn_bytes = DATA_SIZE;
    int step_bytes = fn_bytes * step;
    uint8_t *x10bytes = x + sn;
    uint8_t *x11bytes = x + skip*step_bytes + sn;
    uint8_t *x12bytes = x + 2 * skip*step_bytes + sn;
    uint8_t *y10bytes = y + sn;
    uint8_t *y11bytes = y + skip*step_bytes + sn;
    uint8_t *y12bytes = y + 2 * skip*step_bytes + sn;
    uint8_t *z10bytes = z + sn;
    uint8_t *z11bytes = z + skip*step_bytes + sn;
    uint8_t *z12bytes = z + 2 * skip*step_bytes + sn;
    uint8_t *x30bytes = new uint8_t[step_bytes];
    uint8_t *x31bytes = new uint8_t[step_bytes];
    uint8_t *x32bytes = new uint8_t[step_bytes];
    uint8_t *y30bytes = new uint8_t[step_bytes];
    uint8_t *y31bytes = new uint8_t[step_bytes];
    uint8_t *y32bytes = new uint8_t[step_bytes];
    uint8_t *z30bytes = new uint8_t[step_bytes];
    uint8_t *z31bytes = new uint8_t[step_bytes];
    uint8_t *z32bytes = new uint8_t[step_bytes];
    fixnum_array *dx30, *dx31, *dx32, *dy30, *dy31, *dy32, *dz30, *dz31, *dz32, *x10in, *x11in, *x12in, *y10in, *y11in, *y12in, *z10in, *z11in, *z12in;
    fixnum_array *x20in, *x21in, *x22in, *y20in, *y21in, *y22in, *z20in, *z21in, *z22in;

    // scalar
    auto modulusw = fixnum_array::create(scalar, step_bytes, fn_bytes);
    
    // sigma result
    fixnum_array *rx30, *rx31, *rx32, *ry30, *ry31, *ry32, *rz30, *rz31, *rz32;
    int got_result = false;

#if 0
    printf("input:\n");
    printf("x10:");
    for (int i = 0; i < step_bytes; i ++) {
        printf("%02x", x10bytes[i]); 
        if ((i+1) % 96 == 0) {
            printf("\t");
        }
    }
    printf("\nx11:");
    for (int i = 0; i < step_bytes; i ++) {
        printf("%02x", x11bytes[i]); 
        if ((i+1) % 96 == 0) {
            printf("\t");
        }
    }
    printf("\nx12:");
    for (int i = 0; i < step_bytes; i ++) {
        printf("%02x", x12bytes[i]); 
        if ((i+1) % 96 == 0) {
            printf("\t");
        }
    }
    printf("\ny10:");
    for (int i = 0; i < step_bytes; i ++) {
        printf("%02x", y10bytes[i]); 
        if ((i+1) % 96 == 0) {
            printf("\t");
        }
    }
    printf("\ny11:");
    for (int i = 0; i < step_bytes; i ++) {
        printf("%02x", y11bytes[i]); 
        if ((i+1) % 96 == 0) {
            printf("\t");
        }
    }
    printf("\ny12:");
    for (int i = 0; i < step_bytes; i ++) {
        printf("%02x", y12bytes[i]); 
        if ((i+1) % 96 == 0) {
            printf("\t");
        }
    }
    printf("\nz10:");
    for (int i = 0; i < step_bytes; i ++) {
        printf("%02x", z10bytes[i]); 
        if ((i+1) % 96 == 0) {
            printf("\t");
        }
    }
    printf("\nz11:");
    for (int i = 0; i < step_bytes; i ++) {
        printf("%02x", z11bytes[i]); 
        if ((i+1) % 96 == 0) {
            printf("\t");
        }
    }
    printf("\nz12:");
    for (int i = 0; i < step_bytes; i ++) {
        printf("%02x", z12bytes[i]); 
        if ((i+1) % 96 == 0) {
            printf("\t");
        }
    }
    printf("\nscalar:");
    for (int i = 0; i < step_bytes; i ++) {
        printf("%02x", scalar[i]); 
        if ((i+1) % 96 == 0) {
            printf("\t");
        }
    }
    printf("\n");
#endif
    for (int i = 0; i < nelts; i+=step) {
        dx30 = fixnum_array::create(step);
        dx31 = fixnum_array::create(step);
        dx32 = fixnum_array::create(step);
        dy30 = fixnum_array::create(step);
        dy31 = fixnum_array::create(step);
        dy32 = fixnum_array::create(step);
        dz30 = fixnum_array::create(step);
        dz31 = fixnum_array::create(step);
        dz32 = fixnum_array::create(step);
        x10in = fixnum_array::create(x10bytes, step_bytes, fn_bytes);
        x11in = fixnum_array::create(x11bytes, step_bytes, fn_bytes);
        x12in = fixnum_array::create(x12bytes, step_bytes, fn_bytes);
        y10in = fixnum_array::create(y10bytes, step_bytes, fn_bytes);
        y11in = fixnum_array::create(y11bytes, step_bytes, fn_bytes);
        y12in = fixnum_array::create(y12bytes, step_bytes, fn_bytes);
        z10in = fixnum_array::create(z10bytes, step_bytes, fn_bytes);
        z11in = fixnum_array::create(z11bytes, step_bytes, fn_bytes);
        z12in = fixnum_array::create(z12bytes, step_bytes, fn_bytes);
        fixnum_array::template map<mnt6g2_calc_np>(modulusw, x10in, x11in, x12in, y10in, y11in, y12in, z10in, z11in, z12in, dx30, dx31, dx32, dy30, dy31, dy32, dz30, dz31, dz32);

        dx30->retrieve_all(x30bytes, step_bytes, &size);
        dx31->retrieve_all(x31bytes, step_bytes, &size);
        dx32->retrieve_all(x32bytes, step_bytes, &size);
        dy30->retrieve_all(y30bytes, step_bytes, &size);
        dy31->retrieve_all(y31bytes, step_bytes, &size);
        dy32->retrieve_all(y32bytes, step_bytes, &size);
        dz30->retrieve_all(z30bytes, step_bytes, &size);
        dz31->retrieve_all(z31bytes, step_bytes, &size);
        dz32->retrieve_all(z32bytes, step_bytes, &size);
        delete x10in;
        delete x11in;
        delete x12in;
        delete y10in;
        delete y11in;
        delete y12in;
        delete z10in;
        delete z11in;
        delete z12in;
        delete dx30;
        delete dx31;
        delete dx32;
        delete dy30;
        delete dy31;
        delete dy32;
        delete dz30;
        delete dz31;
        delete dz32;
#if 0
        printf("calc np result\n");
        printf("x30:");
        for (int i = 0; i < step_bytes; i ++) {
            printf("%x", x30bytes[i]);
            if ((i+1)%fn_bytes == 0) printf("\t");
        }
        printf("\nx31:");
        for (int i = 0; i < step_bytes; i ++) {
            printf("%x", x31bytes[i]);
            if ((i+1)%fn_bytes == 0) printf("\t");
        }
        printf("\nx32:");
        for (int i = 0; i < step_bytes; i ++) {
            printf("%x", x32bytes[i]);
            if ((i+1)%fn_bytes == 0) printf("\t");
        }
        printf("\ny30:");
        for (int i = 0; i < step_bytes; i ++) {
            printf("%x", y30bytes[i]);
            if ((i+1)%fn_bytes == 0) printf("\t");
        }
        printf("\ny31:");
        for (int i = 0; i < step_bytes; i ++) {
            printf("%x", y31bytes[i]);
            if ((i+1)%fn_bytes == 0) printf("\t");
        }
        printf("\ny32:");
        for (int i = 0; i < step_bytes; i ++) {
            printf("%x", y32bytes[i]);
            if ((i+1)%fn_bytes == 0) printf("\t");
        }
        printf("\nz30:");
        for (int i = 0; i < step_bytes; i ++) {
            printf("%x", z30bytes[i]);
            if ((i+1)%fn_bytes == 0) printf("\t");
        }
        printf("\nz31:");
        for (int i = 0; i < step_bytes; i ++) {
            printf("%x", z31bytes[i]);
            if ((i+1)%fn_bytes == 0) printf("\t");
        }
        printf("\nz32:");
        for (int i = 0; i < step_bytes; i ++) {
            printf("%x", z32bytes[i]);
            if ((i+1)%fn_bytes == 0) printf("\t");
        }
        printf("\n");
#endif
#ifdef PARALLEL_SIGMA
        int start = nelts%2;
        int rnelts = nelts - start;
        uint8_t *rx0, *rx1, *rx2, *ry0, *ry1, *ry2, *rz0, *rz1, *rz2;
        rx0 = new uint8_t[MNT_SIZE*rnelts/2];
        rx1 = new uint8_t[MNT_SIZE*rnelts/2];
        rx2 = new uint8_t[MNT_SIZE*rnelts/2];
        ry0 = new uint8_t[MNT_SIZE*rnelts/2];
        ry1 = new uint8_t[MNT_SIZE*rnelts/2];
        ry2 = new uint8_t[MNT_SIZE*rnelts/2];
        rz0 = new uint8_t[MNT_SIZE*rnelts/2];
        rz1 = new uint8_t[MNT_SIZE*rnelts/2];
        rz2 = new uint8_t[MNT_SIZE*rnelts/2];
        while(rnelts > 1) {
            do_sigma(rnelts, x30bytes + start*MNT_SIZE, x31bytes + start*MNT_SIZE, x32bytes + start*MNT_SIZE, y30bytes + start*MNT_SIZE, y31bytes + start*MNT_SIZE, y32bytes + start*MNT_SIZE, z30bytes + start*MNT_SIZE, z31bytes + start*MNT_SIZE, z32bytes + start*MNT_SIZE, rx0, rx1, rx2, ry0, ry1, ry2, rz0, rz1, rz2);
            rnelts = rnelts >> 1;
            memcpy(x30bytes + start*MNT_SIZE, rx0, rnelts*MNT_SIZE);
            memcpy(x31bytes + start*MNT_SIZE, rx1, rnelts*MNT_SIZE);
            memcpy(x32bytes + start*MNT_SIZE, rx2, rnelts*MNT_SIZE);
            memcpy(y30bytes + start*MNT_SIZE, ry0, rnelts*MNT_SIZE);
            memcpy(y31bytes + start*MNT_SIZE, ry1, rnelts*MNT_SIZE);
            memcpy(y32bytes + start*MNT_SIZE, ry2, rnelts*MNT_SIZE);
            memcpy(z30bytes + start*MNT_SIZE, rz0, rnelts*MNT_SIZE);
            memcpy(z31bytes + start*MNT_SIZE, rz1, rnelts*MNT_SIZE);
            memcpy(z32bytes + start*MNT_SIZE, rz2, rnelts*MNT_SIZE);
            if (rnelts > 1 && rnelts%2) {
                if (start == 0) {
                    start = 1;
                    rnelts -= 1;
                } else {
                    start = 0;
                    rnelts += 1;
                }
            }
        }
        delete rx0;
        delete rx1;
        delete rx2;
        delete ry0;
        delete ry1;
        delete ry2;
        delete rz0;
        delete rz1;
        delete rz2;
        if (start == 1) {
            // add the first element
            x20in = fixnum_array::create(x30bytes, fn_bytes, fn_bytes);
            x21in = fixnum_array::create(x31bytes, fn_bytes, fn_bytes);
            x22in = fixnum_array::create(x32bytes, fn_bytes, fn_bytes);
            y20in = fixnum_array::create(y30bytes, fn_bytes, fn_bytes);
            y21in = fixnum_array::create(y31bytes, fn_bytes, fn_bytes);
            y22in = fixnum_array::create(y32bytes, fn_bytes, fn_bytes);
            z20in = fixnum_array::create(z30bytes, fn_bytes, fn_bytes);
            z21in = fixnum_array::create(z31bytes, fn_bytes, fn_bytes);
            z22in = fixnum_array::create(z32bytes, fn_bytes, fn_bytes);
            rx30 = fixnum_array::create(x30bytes + fn_bytes, fn_bytes, fn_bytes);
            rx31 = fixnum_array::create(x31bytes + fn_bytes, fn_bytes, fn_bytes);
            rx32 = fixnum_array::create(x32bytes + fn_bytes, fn_bytes, fn_bytes);
            ry30 = fixnum_array::create(y30bytes + fn_bytes, fn_bytes, fn_bytes);
            ry31 = fixnum_array::create(y31bytes + fn_bytes, fn_bytes, fn_bytes);
            ry32 = fixnum_array::create(y32bytes + fn_bytes, fn_bytes, fn_bytes);
            rz30 = fixnum_array::create(z30bytes + fn_bytes, fn_bytes, fn_bytes);
            rz31 = fixnum_array::create(z31bytes + fn_bytes, fn_bytes, fn_bytes);
            rz32 = fixnum_array::create(z32bytes + fn_bytes, fn_bytes, fn_bytes);
            fixnum_array::template map<mnt6g2_pq_plus>(rx30, rx31, rx32, ry30, ry31, ry32, rz30, rz31, rz32, x20in, x21in, x22in, y20in, y21in, y22in, z20in, z21in, z22in, rx30, rx31, rx32, ry30, ry31, ry32, rz30, rz31, rz32);
            delete x20in;
            delete x21in;
            delete x22in;
            delete y20in;
            delete y21in;
            delete y22in;
            delete z20in;
            delete z21in;
            delete z22in;
        } else {
            memcpy(x3, x30bytes, fn_bytes);
            memcpy(x3 + fn_bytes, x31bytes, fn_bytes);
            memcpy(x3 + 2*fn_bytes, x32bytes, fn_bytes);
            memcpy(y3, y30bytes, fn_bytes);
            memcpy(y3 + fn_bytes, y31bytes, fn_bytes);
            memcpy(y3 + 2*fn_bytes, y32bytes, fn_bytes);
            memcpy(z3, z30bytes, fn_bytes);
            memcpy(z3 + fn_bytes, z31bytes, fn_bytes);
            memcpy(z3 + 2*fn_bytes, z32bytes, fn_bytes);
            got_result = true;
        }
#else
        bool result_set = false;
        // start add from second element
        int start = 1;
        if (i == 0) {
            //print_g2(x30bytes + start * fn_bytes, x31bytes + start * fn_bytes, y30bytes + start * fn_bytes, y31bytes + start * fn_bytes, z30bytes + start * fn_bytes, z31bytes + start * fn_bytes);
            rx30 = fixnum_array::create(x30bytes + start * fn_bytes, fn_bytes, fn_bytes);
            rx31 = fixnum_array::create(x31bytes + start * fn_bytes, fn_bytes, fn_bytes);
            rx32 = fixnum_array::create(x32bytes + start * fn_bytes, fn_bytes, fn_bytes);
            ry30 = fixnum_array::create(y30bytes + start * fn_bytes, fn_bytes, fn_bytes);
            ry31 = fixnum_array::create(y31bytes + start * fn_bytes, fn_bytes, fn_bytes);
            ry32 = fixnum_array::create(y32bytes + start * fn_bytes, fn_bytes, fn_bytes);
            rz30 = fixnum_array::create(z30bytes + start * fn_bytes, fn_bytes, fn_bytes);
            rz31 = fixnum_array::create(z31bytes + start * fn_bytes, fn_bytes, fn_bytes);
            rz32 = fixnum_array::create(z32bytes + start * fn_bytes, fn_bytes, fn_bytes);
            result_set = true;
        }
        int k = 0;
        if (result_set && i == 0) {
            k = start + 1;
        }
        for (; k < step; k ++)
        {
            //print_g2(x30bytes + k * fn_bytes, x31bytes + k * fn_bytes, y30bytes + k * fn_bytes, y31bytes + k * fn_bytes, z30bytes + k * fn_bytes, z31bytes + k * fn_bytes);
            x20in = fixnum_array::create(x30bytes + k * fn_bytes, fn_bytes, fn_bytes);
            x21in = fixnum_array::create(x31bytes + k * fn_bytes, fn_bytes, fn_bytes);
            x22in = fixnum_array::create(x32bytes + k * fn_bytes, fn_bytes, fn_bytes);
            y20in = fixnum_array::create(y30bytes + k * fn_bytes, fn_bytes, fn_bytes);
            y21in = fixnum_array::create(y31bytes + k * fn_bytes, fn_bytes, fn_bytes);
            y22in = fixnum_array::create(y32bytes + k * fn_bytes, fn_bytes, fn_bytes);
            z20in = fixnum_array::create(z30bytes + k * fn_bytes, fn_bytes, fn_bytes);
            z21in = fixnum_array::create(z31bytes + k * fn_bytes, fn_bytes, fn_bytes);
            z22in = fixnum_array::create(z32bytes + k * fn_bytes, fn_bytes, fn_bytes);
            fixnum_array::template map<mnt6g2_pq_plus>(rx30, rx31, rx32, ry30, ry31, ry32, rz30, rz31, rz32, x20in, x21in, x22in, y20in, y21in, y22in, z20in, z21in, z22in, rx30, rx31, rx32, ry30, ry31, ry32, rz30, rz31, rz32);
            delete x20in;
            delete x21in;
            delete x22in;
            delete y20in;
            delete y21in;
            delete y22in;
            delete z20in;
            delete z21in;
            delete z22in;
#if 0
        rx30->retrieve_all(x30, fn_bytes, &size);
        rx31->retrieve_all(x31, fn_bytes, &size);
        ry30->retrieve_all(y30, fn_bytes, &size);
        ry31->retrieve_all(y31, fn_bytes, &size);
        rz30->retrieve_all(z30, fn_bytes, &size);
        rz31->retrieve_all(z31, fn_bytes, &size);
            print_g2(x30, x31, y30, y31, z30, z31);
#endif
        }
        // add the first element
        x20in = fixnum_array::create(x30bytes, fn_bytes, fn_bytes);
        x21in = fixnum_array::create(x31bytes, fn_bytes, fn_bytes);
        x22in = fixnum_array::create(x32bytes, fn_bytes, fn_bytes);
        y20in = fixnum_array::create(y30bytes, fn_bytes, fn_bytes);
        y21in = fixnum_array::create(y31bytes, fn_bytes, fn_bytes);
        y22in = fixnum_array::create(y32bytes, fn_bytes, fn_bytes);
        z20in = fixnum_array::create(z30bytes, fn_bytes, fn_bytes);
        z21in = fixnum_array::create(z31bytes, fn_bytes, fn_bytes);
        z22in = fixnum_array::create(z32bytes, fn_bytes, fn_bytes);
        fixnum_array::template map<mnt6g2_pq_plus>(rx30, rx31, rx32, ry30, ry31, ry32, rz30, rz31, rz32, x20in, x21in, x22in, y20in, y21in, y22in, z20in, z21in, z22in, rx30, rx31, rx32, ry30, ry31, ry32, rz30, rz31, rz32);
        delete x20in;
        delete x21in;
        delete x22in;
        delete y20in;
        delete y21in;
        delete y22in;
        delete z20in;
        delete z21in;
        delete z22in;
#endif
    }
    if (!got_result) {
        size = 1;
        rx30->retrieve_all(x3, fn_bytes, &size);
        rx31->retrieve_all(x3 + fn_bytes, fn_bytes, &size);
        rx32->retrieve_all(x3 + 2*fn_bytes, fn_bytes, &size);
        ry30->retrieve_all(y3, fn_bytes, &size);
        ry31->retrieve_all(y3 + fn_bytes, fn_bytes, &size);
        ry32->retrieve_all(y3 + 2*fn_bytes, fn_bytes, &size);
        rz30->retrieve_all(z3, fn_bytes, &size);
        rz31->retrieve_all(z3 + fn_bytes, fn_bytes, &size);
        rz32->retrieve_all(z3 + 2*fn_bytes, fn_bytes, &size);
        delete rx30;
        delete rx31;
        delete rx32;
        delete ry30;
        delete ry31;
        delete ry32;
        delete rz30;
        delete rz31;
        delete rz32;
    }
    delete x30bytes;
    delete x31bytes;
    delete x32bytes;
    delete y30bytes;
    delete y31bytes;
    delete y32bytes;
    delete z30bytes;
    delete z31bytes;
    delete z32bytes;
    delete modulusw;

    printf("mnt6_g2 final result:");
    //print_g2(x30, x31, y30, y31, z30, z31);
    clock_t diff = clock() - start;
    printf("cost time %ld\n", diff);
    return 0;
}
