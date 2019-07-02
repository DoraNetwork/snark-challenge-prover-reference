#include <cstdio>
#include <cstring>
#include <cassert>

#include "fixnum/warp_fixnum.cu"
#include "array/fixnum_array.h"
#include "functions/modexp.cu"
#include "functions/multi_modexp.cu"
#include "modnum/modnum_monty_redc.cu"
#include "modnum/modnum_monty_cios.cu"
int do_calc_np_sigma(int n, std::vector<uint8_t *> scaler, std::vector<uint8_t *> x1, std::vector<uint8_t *> y1, std::vector<uint8_t *> z1, uint8_t *x3, uint8_t *y3, uint8_t *z3);

using namespace std;
using namespace cuFIXNUM;
int BLOCK_NUM = 4096;
const int THREAD_NUM = 1024;
#define MNT_SIZE (96)
// mnt4_q
const uint8_t mnt4_modulus[MNT_SIZE] = {1,128,94,36,222,99,144,94,159,17,221,44,82,84,157,227,240,37,196,154,113,16,136,99,164,84,114,118,233,204,90,104,56,126,83,203,165,13,15,184,157,5,24,242,118,231,23,177,157,247,90,161,217,36,209,153,141,237,160,232,37,185,253,7,115,216,151,108,249,232,183,94,237,175,143,91,80,151,249,183,173,205,226,238,34,144,34,16,17,196,146,45,198,196,1,0};

// mnt6_q
uint8_t mnt6_modulus[MNT_SIZE] = {1,0,0,64,226,118,7,217,79,58,161,15,23,153,160,78,151,87,0,63,188,129,195,214,164,58,153,52,118,249,223,185,54,38,33,41,148,202,235,62,155,169,89,200,40,92,108,178,157,247,90,161,217,36,209,153,141,237,160,232,37,185,253,7,115,216,151,108,249,232,183,94,237,175,143,91,80,151,249,183,173,205,226,238,34,144,34,16,17,196,146,45,198,196,1,0};

// mnt4 a, for calc 2p, not use now
uint8_t mnt4_a[MNT_SIZE] = {0x84,0xde,0xb8,0xb3,0x57,0xd9,0x51,0x31,0xd,0x8d,0x6,0xb4,0x8c,0x63,0x9a,0x23,0x5d,0xae,0x28,0x9a,0x41,0xc9,0x87,0x2f,0x3,0x6c,0x11,0x8f,0x33,0x30,0xb1,0xf2,0xde,0x2e,0x11,0x42,0x28,0x39,0x4d,0xda,0xd1,0x3a,0x06,0x9e,0x15,0x9b,0x1e,0x3c,0xb2,0xa,0x67,0x26,0x6e,0x77,0x18,0x64,0xc4,0x14,0xe0,0xa5,0x05,0x86,0x16,0xb3,0x42,0x4c,0x19,0xfb,0x97,0x93,0xe9,0x80,0x18,0xd1,0xcb,0x70,0xb6,0xfd,0x48,0x1f,0x2a,0x43,0xf3,0x3f,0x66,0xbf,0x8a,0x2a,0x85,0xc4,0x91,0x3d,0x8f,0xf6};

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
struct mul_wide {
    __device__ void operator()(fixnum &r, fixnum &s, fixnum a, fixnum b) {
        fixnum rr, ss;
        fixnum::mul_wide(ss, rr, a, b);
#if 0
	dump<fixnum>(rr, 24);
#endif
        r = rr;
	s = ss;
    }
};

template< typename fixnum >
struct mod_redc {
    __device__ void operator()(fixnum a, fixnum b, fixnum mod, fixnum &r) {
    	typedef modnum_monty_redc<fixnum> modnum;
        fixnum rr,hh,ll;
 	modnum m(mod);
#if 1
	//dump<fixnum>(mod, 32);
	fixnum::mul_wide(hh, ll, a, b);
	m.redc(rr, hh, ll);
#else
 	m.mul(rr, a, b);
#endif
#if 0
    dump<fixnum>(mod, 32);
	dump<fixnum>(rr, 32);
#endif
        r = rr;
    }
};

template< typename fixnum >
__device__ void pq_plus_inner(fixnum mod, fixnum x1, fixnum y1, fixnum z1, fixnum x2, fixnum y2, fixnum z2, fixnum &x3, fixnum &y3, fixnum &z3) {
    typedef modnum_monty_cios<fixnum> modnum;
    modnum m(mod);
    fixnum y1z2, x1z2, z1z2, y2z1, x2z1;

    // y1z2
    m.mul(y1z2, y1, z2);

    // x1z2
    m.mul(x1z2, x1, z2);

    // z1z2
    m.mul(z1z2, z1, z2);

    // y2z1
    m.mul(y2z1, y2, z1);

    // x2z1
    m.mul(x2z1, x2, z1);

    fixnum u;
    // u = Y2Z1-Y1Z2
    m.sub(u, y2z1, y1z2);
    //fixnum::sub(u, y2z1, y1z2);

    fixnum uu;
    // uu = u*u
    m.mul(uu, u, u);

    fixnum v;
    // v = X2Z1-X1Z2
    m.sub(v, x2z1, x1z2);
    //fixnum::sub(v, x2z1, x1z2);

    fixnum vv;
    // vv = v*v
    m.mul(vv, v, v);

    fixnum vvv;
    // vvv = v*vv
    m.mul(vvv, v, vv);

    fixnum R;
    // R = vv * X1Z2
    m.mul(R, vv, x1z2);

    fixnum A;
    // A = uu * Z1Z2 - vvv - 2R
    m.mul(A, uu, z1z2);
    m.sub(A, A, vvv);
    m.sub(A, A, R);
    m.sub(A, A, R);

    // x3 = v*A
    m.mul(x3, v, A);

    // y3 = u*(R-A)-vvv*Y1Z2
    fixnum temp;
    m.sub(temp, R, A);
    m.mul(temp, u, temp);
    fixnum temp2;
    m.mul(temp2, vvv, y1z2);
    m.sub(y3, temp, temp2);

    // z3 = vvv*Z1Z2
    m.mul(z3, vvv, z1z2);
};

template< typename fixnum >
struct pq_plus {
    __device__ void operator()(fixnum mod, fixnum x1, fixnum y1, fixnum z1, fixnum x2, fixnum y2, fixnum z2, fixnum &x3, fixnum &y3, fixnum &z3) {
        if (fixnum::is_zero(x1) && fixnum::is_zero(z1)) {
            x3 = x2;
            y3 = y2;
            z3 = z2;
            return;
        }
        if (fixnum::is_zero(x2) && fixnum::is_zero(z2)) {
            x3 = x1;
            y3 = y1;
            z3 = z1;
            return;
        }
        pq_plus_inner(mod, x1, y1, z1, x2, y2, z2, x3, y3, z3);
  }
};

template< typename fixnum >
__device__ void p_double_inner(fixnum mod, fixnum a, fixnum x1, fixnum y1, fixnum z1, fixnum &x3, fixnum &y3, fixnum &z3) {
    typedef modnum_monty_cios<fixnum> modnum;
    modnum m(mod);

    fixnum temp, temp2;

    fixnum XX;
    // XX = X1*X1
    m.mul(XX, x1, x1);

    fixnum ZZ;
    // ZZ = Z1*Z1
    m.mul(ZZ, z1, z1);

    fixnum w;
    // w = a*ZZ+3*XX    TODO: correct a, cur = 2
    m.add(temp, ZZ, ZZ);
    m.add(temp2, XX, XX);
    m.add(temp2, temp2, XX);
    m.add(w, temp, temp2);

    fixnum s;
    // s = 2*Y1*Z1
    m.mul(temp, y1, z1);
    m.add(s, temp, temp);

    fixnum ss;
    // ss = s*s
    m.mul(ss, s, s);

    fixnum sss;
    // sss = s * ss
    m.mul(sss, s, ss);

    fixnum R;
    // R = y1 * s
    m.mul(R, y1, s);

    fixnum RR;
    // RR = R * R
    m.mul(RR, R, R);

    fixnum B;
    // B = (X1+R)2-XX-RR
    m.add(temp, x1, R);
    m.mul(temp2, temp, temp);
    m.sub(temp, temp2, XX);
    m.sub(B, temp, RR);

    fixnum h;
    // h = w*2-2*B
    m.mul(temp, w, w);
    m.add(temp2, B, B);
    m.sub(h, temp, temp2);

    // X3 = h * s
    m.mul(x3, h, s);

    // Y3 = w*(B-h)-2*RR
    m.sub(temp, B, h);
    m.add(temp2, RR, RR);
    m.mul(temp, w, temp);
    m.sub(y3, temp, temp2);

    //  Z3 = sss
    z3 = sss;
};

template< typename fixnum >
struct p_double {
    __device__ void operator()(fixnum mod, fixnum a, fixnum x1, fixnum y1, fixnum z1, fixnum &x3, fixnum &y3, fixnum &z3) {
        p_double_inner(mod, a, x1, y1, z1, x3, y3, z3);
  }
};

template< typename fixnum >
struct calc_np {
    __device__ void operator()(fixnum mod, fixnum w, fixnum x1, fixnum y1, fixnum z1, fixnum &x3, fixnum &y3, fixnum &z3) {
    typedef modnum_monty_cios<fixnum> modnum;
    modnum m(mod);
    fixnum rx, ry, rz;
    fixnum tempw = w;
    int i = 24*32 - 1;
    bool found_one = false;
    int count = 0;
#if 0
    if (threadIdx.x > 23) 
#endif
    {
        dump(w, 24);
        dump(x1, 24);
        dump(y1, 24);
        dump(z1, 24);
    }
    //while(fixnum::cmp(tempw, fixnum::zero()) && i >= 0) {
    for (;i >= 0;i--) {
        size_t value = fixnum::get(tempw, i/32);
#if 0
        if (threadIdx.x > 23) {
            printf("i %d value[%d] %x\n", i, i/32, value);
        }
#endif
        if (found_one) {
            p_double_inner(mod, mod, rx, ry, rz, rx, ry, rz);
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
                pq_plus_inner(mod, rx, ry, rz, x1, y1, z1, rx, ry, rz);
            }
#if 0
            if (threadIdx.x > 23) {
            printf("add result\n");
            dump(rx, 24);
            }
#endif
            found_one = true;
        }
        count ++;
#if 0
        if (threadIdx.x > 23) {
        //if (count >20) break;
        }
#endif
    }
    x3 = rx;
    y3 = ry;
    z3 = rz;
#if 1
    printf("final result\n");
    dump(x3, 24);
    dump(y3, 24);
    dump(z3, 24);
#endif
  }
};

template< int fn_bytes, typename word_fixnum, template <typename> class Func>
void bench(int nelts, FILE *in_file, FILE *out_file) {
    typedef warp_fixnum<fn_bytes, word_fixnum> fixnum;
    typedef fixnum_array<fixnum> fixnum_array;
    typedef modnum_monty_redc<fixnum> modnum;

    if (nelts == 0) {
        puts(" -*-  nelts == 0; skipping...  -*-");
        return;
    }

    clock_t c = clock();
    clock_t c_read = 0;
    clock_t c_malloc = 0;
    clock_t c_cal = 0;
    clock_t c_write = 0;
    clock_t c_pinput = 0;
    int size = 0;

    std::vector<uint8_t *> x1;
    std::vector<uint8_t *> y1;
    std::vector<uint8_t *> z1;
    std::vector<uint8_t *> x2;
    std::vector<uint8_t *> y2;
    std::vector<uint8_t *> z2;
    std::vector<uint8_t *> scalar;

    uint8_t *input;
    const int DATA_SIZE = fn_bytes;
    // calc warp num
    int step = BLOCK_NUM*THREAD_NUM/32;
    while (nelts%step != 0) {
       step = step >> 1; 
    }
    step = 1;
    uint8_t *c_val = new uint8_t[DATA_SIZE*step];
    int step_bytes = fn_bytes * step;
    uint8_t *x1bytes = new uint8_t[step_bytes];
    uint8_t *y1bytes = new uint8_t[step_bytes];
    uint8_t *z1bytes = new uint8_t[step_bytes];
#ifdef PQPLUS
    uint8_t *x2bytes = new uint8_t[step_bytes];
    uint8_t *y2bytes = new uint8_t[step_bytes];
    uint8_t *z2bytes = new uint8_t[step_bytes];
#endif
    uint8_t *wbytes  = new uint8_t[step_bytes];

    uint8_t *modulus_bytes = new uint8_t[step_bytes];
    // mnt4 q
    memset(modulus_bytes, step_bytes, 0);
    for(int i = 0; i < step; i++) {
        memcpy(modulus_bytes + i*fn_bytes, mnt4_modulus, fn_bytes);
    }
    auto modulus4 = fixnum_array::create(modulus_bytes, step_bytes, fn_bytes);

    // mnt6 q
    memset(modulus_bytes, step_bytes, 0);
    for(int i = 0; i < step; i++) {
        memcpy(modulus_bytes + i*fn_bytes, mnt6_modulus, fn_bytes);
    }
    auto modulus6 = fixnum_array::create(modulus_bytes, step_bytes, fn_bytes);

    // mnt4 a
    memset(modulus_bytes, step_bytes, 0);
    for(int i = 0; i < step; i++) {
        memcpy(modulus_bytes + i*fn_bytes, mnt4_a, fn_bytes);
    }
    auto mnt4a = fixnum_array::create(modulus_bytes, step_bytes, fn_bytes);

    // read w
    fread((void *)wbytes, DATA_SIZE, 1, in_file);
    scalar.emplace_back(wbytes);
    memset(modulus_bytes, step_bytes, 0);
    for(int i = 0; i < step; i++) {
        memcpy(modulus_bytes + i*fn_bytes, wbytes, fn_bytes);
    }
    auto modulusw = fixnum_array::create(modulus_bytes, step_bytes, fn_bytes);

    for (int i = 0; i < nelts; i++) {
        input = new uint8_t[fn_bytes];
        fread((void *)input, DATA_SIZE, 1, in_file);
        x1.emplace_back(input);
        input = new uint8_t[fn_bytes];
        fread((void *)input, DATA_SIZE, 1, in_file);
        y1.emplace_back(input);
        input = new uint8_t[fn_bytes];
        fread((void *)input, DATA_SIZE, 1, in_file);
        z1.emplace_back(input);
#ifdef PQPLUS
        input = new uint8_t[fn_bytes];
        fread((void *)input, DATA_SIZE, 1, in_file);
        x2.emplace_back(input);
        input = new uint8_t[fn_bytes];
        fread((void *)input, DATA_SIZE, 1, in_file);
        y2.emplace_back(input);
        input = new uint8_t[fn_bytes];
        fread((void *)input, DATA_SIZE, 1, in_file);
        z2.emplace_back(input);
#endif
    }
#if 1
    uint8_t rx3[fn_bytes];
    uint8_t ry3[fn_bytes];
    uint8_t rz3[fn_bytes];
    do_calc_np_sigma(1, scalar, x1, y1, z1, rx3, ry3, rz3);
    return;
#endif
    c_read = clock() - c;
    clock_t temp, diff;
    fixnum_array *x3, *y3, *z3, *x1in, *y1in, *z1in;
#ifdef PQPLUS
    fixnum_array *x2in, *y2in, *z2in;
#endif
    for (int k = 0; k < 1; k++) {
        for (int i = 0; i < nelts; i+=step) {
            temp = clock();
            for (int j = 0; j < step; j++) {
                memcpy(x1bytes + j*fn_bytes, x1[i+j], fn_bytes);
                memcpy(y1bytes + j*fn_bytes, y1[i+j], fn_bytes);
                memcpy(z1bytes + j*fn_bytes, z1[i+j], fn_bytes);
#ifdef PQPLUS
                memcpy(x2bytes + j*fn_bytes, x2[i+j], fn_bytes);
                memcpy(y2bytes + j*fn_bytes, y2[i+j], fn_bytes);
                memcpy(z2bytes + j*fn_bytes, z2[i+j], fn_bytes);
#endif
            }
            diff = clock() - temp;
            c_pinput += diff;
            temp = clock();
            x3 = fixnum_array::create(step);
            y3 = fixnum_array::create(step);
            z3 = fixnum_array::create(step);
            x1in = fixnum_array::create(x1bytes, step_bytes, fn_bytes);
            y1in = fixnum_array::create(y1bytes, step_bytes, fn_bytes);
            z1in = fixnum_array::create(z1bytes, step_bytes, fn_bytes);
#ifdef PQPLUS
            x2in = fixnum_array::create(x2bytes, step_bytes, fn_bytes);
            y2in = fixnum_array::create(y2bytes, step_bytes, fn_bytes);
            z2in = fixnum_array::create(z2bytes, step_bytes, fn_bytes);
#endif
            diff = clock() - temp;
            c_malloc += diff;
            temp = clock();
#ifdef PQPLUS
            fixnum_array::template map<Func>(modulus4, x1in, y1in, z1in, x2in, y2in, z2in, x3, y3, z3);
#endif
            //fixnum_array::template map<Func>(modulus4, mnt4a, x1in, y1in, z1in, x3, y3, z3);
            fixnum_array::template map<Func>(modulus4, modulusw, x1in, y1in, z1in, x3, y3, z3);
            diff = clock() - temp;
            c_cal += diff;
            // write output
            temp = clock();
            //memset(c_val, 0x0, DATA_SIZE*step);
            x3->retrieve_all(c_val, DATA_SIZE*step, &size);
            fwrite(c_val, 1, DATA_SIZE*step, out_file);
            y3->retrieve_all(c_val, DATA_SIZE*step, &size);
            fwrite(c_val, 1, DATA_SIZE*step, out_file);
            z3->retrieve_all(c_val, DATA_SIZE*step, &size);
            fwrite(c_val, 1, DATA_SIZE*step, out_file);
            delete x1in;
            delete y1in;
            delete z1in;
#ifdef PQPLUS
            delete x2in;
            delete y2in;
            delete z2in;
#endif
            //delete res;
            x1in = x3;
            diff = clock() - temp;
            c_write += diff;
	    }
    }
    c = clock() - c;

    double secinv = (double)CLOCKS_PER_SEC / c;
    double total_MiB = fixnum::BYTES * (double)nelts / (1 << 20);
    printf(" %4d   %3d    %6.1f   %7.3f  %12.1f\n",
           fixnum::BITS, fixnum::digit::BITS, total_MiB,
           1/secinv, nelts * 1e-3 * secinv);
    diff = c - (c_pinput + c_read + c_malloc + c_cal + c_write);
    printf("read file %lld prepare input %lld cuda malloc %lld cuda calc %lld get and write file %lld total %lld diff %lld\n", 
        /*1/((double)CLOCKS_PER_SEC /*/ (c_read), 
        c_pinput,
        (c_malloc), 
        (c_cal),
        (c_write),
        c,
        diff);
    delete modulus_bytes;
    delete modulus4;
    delete mnt4a;
    delete modulus6;
    delete c_val;
    delete x1bytes;
    delete y1bytes;
    delete z1bytes;
#ifdef PQPLUS
    delete x2bytes;
    delete y2bytes;
    delete z2bytes;
#endif
}

template< template <typename> class Func>
void bench_func(const char *fn_name, int nelts, FILE *in_file, FILE *out_file) {
    printf("Function: %s, #elts: %de3\n", fn_name, (int)(nelts * 1e-3));
    printf("fixnum digit  total data   time       Kops/s\n");
    printf(" bits  bits     (MiB)    (seconds)\n");
    bench<96, u32_fixnum, Func>(nelts, in_file, out_file);
}

int main(int argc, char *argv[]) {
    // argv should be
    // { "main", "compute", inputs, outputs }

    size_t n;

    auto inputs = fopen(argv[2], "r");
    auto outputs = fopen(argv[3], "w");
    if (argc > 4) {
        printf("blocknum is %d\n", atoi(argv[4]));
        BLOCK_NUM = atoi(argv[4]);
    }

    while (true) {
        size_t elts_read = fread((void *) &n, sizeof(size_t), 1, inputs);
        if (elts_read == 0) { return 0; }
        printf("Totally %d elements...\n", n);

        //bench_func<mul_wide>("mul_wide", n, inputs, outputs);
        //bench_func<pq_plus>("pqplus", n, inputs, outputs);
        //bench_func<p_double>("pdouble", n, inputs, outputs);
        bench_func<calc_np>("pdouble", n, inputs, outputs);
    }

    fclose(inputs);
    fclose(outputs);

    return 0;
}

int do_calc_np(size_t nelts, std::vector<uint8_t *> scaler, std::vector<uint8_t *> x1, std::vector<uint8_t *> y1, std::vector<uint8_t *> z1, uint8_t *x3, uint8_t *y3, uint8_t *z3) {
    typedef warp_fixnum<96, u32_fixnum> fixnum;
    typedef fixnum_array<fixnum> fixnum_array;
    // calc warp num
    int step = BLOCK_NUM*THREAD_NUM/32;
    while (nelts%step != 0) {
       step = step >> 1; 
    }
    int size = 0;
    step = 1; // test
    int DATA_SIZE = 96;
    int fn_bytes = DATA_SIZE;
    uint8_t *c_val = new uint8_t[DATA_SIZE*step];
    int step_bytes = fn_bytes * step;
    uint8_t *x1bytes = new uint8_t[step_bytes];
    uint8_t *y1bytes = new uint8_t[step_bytes];
    uint8_t *z1bytes = new uint8_t[step_bytes];
    fixnum_array *dx3, *dy3, *dz3, *x1in, *y1in, *z1in;
    uint8_t *modulus_bytes = new uint8_t[step_bytes];
    // mnt4 q
    memset(modulus_bytes, step_bytes, 0);
    for(int i = 0; i < step; i++) {
        memcpy(modulus_bytes + i*fn_bytes, mnt4_modulus, fn_bytes);
    }
    auto modulus4 = fixnum_array::create(modulus_bytes, step_bytes, fn_bytes);

    // mnt6 q
    memset(modulus_bytes, step_bytes, 0);
    for(int i = 0; i < step; i++) {
        memcpy(modulus_bytes + i*fn_bytes, mnt6_modulus, fn_bytes);
    }
    auto modulus6 = fixnum_array::create(modulus_bytes, step_bytes, fn_bytes);

    // mnt4 a
    memset(modulus_bytes, step_bytes, 0);
    for(int i = 0; i < step; i++) {
        memcpy(modulus_bytes + i*fn_bytes, mnt4_a, fn_bytes);
    }
    auto mnt4a = fixnum_array::create(modulus_bytes, step_bytes, fn_bytes);

    // scaler
    memset(modulus_bytes, step_bytes, 0);
    for(int i = 0; i < step; i++) {
        memcpy(modulus_bytes + i*fn_bytes, scaler[i], fn_bytes);
    }
    auto modulusw = fixnum_array::create(modulus_bytes, step_bytes, fn_bytes);

    for (int i = 0; i < nelts; i+=step) {
        for (int j = 0; j < step; j++) {
            memcpy(x1bytes + j*fn_bytes, x1[i+j], fn_bytes);
            memcpy(y1bytes + j*fn_bytes, y1[i+j], fn_bytes);
            memcpy(z1bytes + j*fn_bytes, z1[i+j], fn_bytes);
        }
        dx3 = fixnum_array::create(step);
        dy3 = fixnum_array::create(step);
        dz3 = fixnum_array::create(step);
        x1in = fixnum_array::create(x1bytes, step_bytes, fn_bytes);
        y1in = fixnum_array::create(y1bytes, step_bytes, fn_bytes);
        z1in = fixnum_array::create(z1bytes, step_bytes, fn_bytes);
        fixnum_array::template map<calc_np>(modulus4, modulusw, x1in, y1in, z1in, dx3, dy3, dz3);

        //memset(c_val, 0x0, DATA_SIZE*step);
        dx3->retrieve_all(x3 + i*DATA_SIZE*step, DATA_SIZE*step, &size);
        dy3->retrieve_all(y3 + i*DATA_SIZE*step, DATA_SIZE*step, &size);
        dz3->retrieve_all(z3 + i*DATA_SIZE*step, DATA_SIZE*step, &size);
        delete x1in;
        delete y1in;
        delete z1in;
        delete dx3;
        delete dy3;
        delete dz3;
    }
    return 0;
}

int do_calc_np_sigma(int nelts, std::vector<uint8_t *> scaler, std::vector<uint8_t *> x1, std::vector<uint8_t *> y1, std::vector<uint8_t *> z1, uint8_t *x3, uint8_t *y3, uint8_t *z3) {
    typedef warp_fixnum<96, u32_fixnum> fixnum;
    typedef fixnum_array<fixnum> fixnum_array;
    printf("calc do_calc_np_sigma\n");
    printf("nelts %d\n", nelts);
    //do_calc_np(nelts, scaler, x1, y1, z1, x3, y3, z3);
    // calc warp num
#if 0
    int step = BLOCK_NUM*THREAD_NUM/32;
    while (nelts%step != 0) {
       step = step >> 1; 
    }
#endif
    if (nelts > 2)
        nelts = 2; //test
    int step = nelts;
    int size = nelts;
    int DATA_SIZE = 96;
    int fn_bytes = DATA_SIZE;
    int step_bytes = fn_bytes * step;
    uint8_t *c_val = new uint8_t[step_bytes];
    uint8_t *x1bytes = new uint8_t[step_bytes];
    uint8_t *y1bytes = new uint8_t[step_bytes];
    uint8_t *z1bytes = new uint8_t[step_bytes];
    uint8_t *x3bytes = new uint8_t[step_bytes];
    uint8_t *y3bytes = new uint8_t[step_bytes];
    uint8_t *z3bytes = new uint8_t[step_bytes];
    fixnum_array *dx3, *dy3, *dz3, *x1in, *y1in, *z1in;
    fixnum_array *x2in, *y2in, *z2in;
    uint8_t *modulus_bytes = new uint8_t[step_bytes];
    // mnt4 q
    memset(modulus_bytes, step_bytes, 0);
    for(int i = 0; i < step; i++) {
        memcpy(modulus_bytes + i*fn_bytes, mnt4_modulus, fn_bytes);
    }
    auto modulus4 = fixnum_array::create(modulus_bytes, step_bytes, fn_bytes);

    // mnt6 q
    memset(modulus_bytes, step_bytes, 0);
    for(int i = 0; i < step; i++) {
        memcpy(modulus_bytes + i*fn_bytes, mnt6_modulus, fn_bytes);
    }
    auto modulus6 = fixnum_array::create(modulus_bytes, step_bytes, fn_bytes);

    // mnt4 a
    memset(modulus_bytes, step_bytes, 0);
    for(int i = 0; i < step; i++) {
        memcpy(modulus_bytes + i*fn_bytes, mnt4_a, fn_bytes);
    }
    auto mnt4a = fixnum_array::create(modulus_bytes, step_bytes, fn_bytes);

    // scaler
    memset(modulus_bytes, step_bytes, 0);
    for(int i = 0; i < step; i++) {
        memcpy(modulus_bytes + i*fn_bytes, scaler[i], fn_bytes);
    }
    auto modulusw = fixnum_array::create(modulus_bytes, step_bytes, fn_bytes);
    
    // sigma result
    fixnum_array *rx3, *ry3, *rz3;
    bool result_set = false;

    for (int i = 0; i < nelts; i+=step) {
        for (int j = 0; j < step; j++) {
            memcpy(x1bytes + j*fn_bytes, x1[i+j], fn_bytes);
            memcpy(y1bytes + j*fn_bytes, y1[i+j], fn_bytes);
            memcpy(z1bytes + j*fn_bytes, z1[i+j], fn_bytes);
        }
        dx3 = fixnum_array::create(step);
        dy3 = fixnum_array::create(step);
        dz3 = fixnum_array::create(step);
        x1in = fixnum_array::create(x1bytes, step_bytes, fn_bytes);
        y1in = fixnum_array::create(y1bytes, step_bytes, fn_bytes);
        z1in = fixnum_array::create(z1bytes, step_bytes, fn_bytes);
        fixnum_array::template map<calc_np>(modulus4, modulusw, x1in, y1in, z1in, dx3, dy3, dz3);

        dx3->retrieve_all(x3bytes, step_bytes, &size);
        dy3->retrieve_all(y3bytes, step_bytes, &size);
        dz3->retrieve_all(z3bytes, step_bytes, &size);
         
#if 0
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
        for (; k < step; k ++) {
            x2in = fixnum_array::create(x3bytes + k * fn_bytes, fn_bytes, fn_bytes);
            y2in = fixnum_array::create(y3bytes + k * fn_bytes, fn_bytes, fn_bytes);
            z2in = fixnum_array::create(z3bytes + k * fn_bytes, fn_bytes, fn_bytes);
            fixnum_array::template map<pq_plus>(modulus4, rx3, ry3, rz3, x2in, y2in, z2in, rx3, ry3, rz3);
            delete x2in;
            delete y2in;
            delete z2in;
        }
        // add first element
        x2in = fixnum_array::create(x3bytes, fn_bytes, fn_bytes);
        y2in = fixnum_array::create(y3bytes, fn_bytes, fn_bytes);
        z2in = fixnum_array::create(z3bytes, fn_bytes, fn_bytes);
        fixnum_array::template map<pq_plus>(modulus4, x2in, y2in, z2in, rx3, ry3, rz3, rx3, ry3, rz3);
        delete x2in;
        delete y2in;
        delete z2in;
        
        delete x1in;
        delete y1in;
        delete z1in;
        delete dx3;
        delete dy3;
        delete dz3;
#endif
    }
#if 0
    rx3->retrieve_all(x3, fn_bytes, &size);
    ry3->retrieve_all(y3, fn_bytes, &size);
    rz3->retrieve_all(z3, fn_bytes, &size);
#endif

    printf("final result");
    printf("\nx3:");
    for (int k = fn_bytes-1; k >= 0; k--) {
        printf("%x", x3[k]);
    }
    printf("\ny3:");
    for (int k = fn_bytes-1; k >= 0; k--) {
        printf("%x", y3[k]);
    }
    printf("\nz3:");
    for (int k = fn_bytes-1; k >= 0; k--) {
       printf("%x", z3[k]);
    }
    printf("\n");
    return 0;
}
