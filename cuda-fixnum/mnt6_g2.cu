#include <cstdio>
#include <cstring>
#include <cassert>

#include "fixnum/warp_fixnum.cu"
#include "array/fixnum_array.h"
#include "functions/modexp.cu"
#include "functions/multi_modexp.cu"
#include "modnum/modnum_monty_redc.cu"
#include "modnum/modnum_monty_cios.cu"

using namespace std;
using namespace cuFIXNUM;

namespace MNT_G{
template< typename fixnum >
class mnt6_g2 {
public:
static __device__ void dump(fixnum n, int size) {
#if 0
	for (int i = 0; i < size; i++) {
		printf("DUMP [%d] %x\n", i, fixnum::get(n, i));
	}
#endif
	printf("dump [%d]=\%04x\n", threadIdx.x, fixnum::get(n, threadIdx.x));
}

//non_residue = 11

typedef modnum_monty_cios<fixnum> modnum;
static __device__ void fp3_multi(modnum m, fixnum a, fixnum b, fixnum c, fixnum A, fixnum B, fixnum C, fixnum &r10, fixnum &r11, fixnum &r12) {
    fixnum aA, bB, cC;

    m.mul(aA, a, A);
    m.mul(bB, b, B);
    m.mul(cC, c, C);

    fixnum apb, bpc, BpC, ApB, apc, ApC;
    m.add(apb, a, b);
    m.add(bpc, b, c);
    m.add(BpC, B, C);
    m.add(ApB, A, B);
    m.add(apc, a, c);
    m.add(ApC, A, C);

    // aA + non_residue*((b+c)*(B+C)-bB-cC)
    m.mul(r10, bpc, BpC);
    m.sub(r10, r10, bB);
    m.sub(r10, r10, cC);

    fixnum temp = r10;
    fixnum temp1;
    for (int i = 0; i < 10; i++) {
        m.add(temp1, temp, r10);
        temp = temp1;
    } 
    m.add(r10, aA, temp1);

    // (a+b)*(A+B)-aA-bB+non_residue*cC
    m.mul(r11, apb, ApB);
    m.sub(r11, r11, aA);
    m.sub(r11, r11, bB);
    temp = cC;
    for (int i = 0; i < 10; i++) {
        m.add(temp1, temp, cC);
        temp = temp1;
    } 
    m.add(r11, r11, temp1);

    // (a+c)*(A+C)-aA+bB-cC
    m.mul(r12, apc, ApC);
    m.sub(r12, r12, aA);
    m.add(r12, r12, bB);
    m.sub(r12, r12, cC);
}

static __device__ void fp3_sub(modnum m, fixnum x10, fixnum x11, fixnum x12, fixnum y10, fixnum y11, fixnum y12, fixnum &r10, fixnum &r11, fixnum &r12) {
    m.sub(r10, x10, y10);

    m.sub(r11, x11, y11); 

    m.sub(r12, x12, y12);
}

static __device__ void fp3_add(modnum m, fixnum x10, fixnum x11, fixnum x12, fixnum y10, fixnum y11, fixnum y12, fixnum &r10, fixnum &r11, fixnum &r12) {
    m.add(r10, x10, y10);

    m.add(r11, x11, y11); 

    m.add(r12, x12, y12); 
}

static __device__ void fp3_square(modnum m, fixnum a, fixnum b, fixnum c, fixnum &r10, fixnum &r11, fixnum &r12) {
    fixnum s0;
    m.mul(s0, a, a);

    fixnum ab;
    m.mul(ab, a, b);

    fixnum s1;
    m.add(s1, ab, ab);

    fixnum s2;
    m.sub(s2, a, b);
    m.add(s2, s2, c);
    m.mul(s2, s2, s2);

    fixnum bc;
    m.mul(bc, b, c);

    fixnum s3;
    m.add(s3, bc, bc);

    fixnum s4;
    m.mul(s4, c, c);

    // s0 + non_residue * s3
    fixnum temp = s3;
    for (int i = 0; i < 10; i ++) {
        m.add(r10, temp, s3);
        temp = r10;
    }
    m.add(r10, s0, r10);

    // s1 + non_residue * s4
    temp = s4;
    for (int i = 0; i < 10; i ++) {
        m.add(r11, temp, s4);
        temp = r11;
    }
    m.add(r11, s1, r11);

    // s1 + s2 + s3 - s0 - s4
    m.add(r12, s1, s2);
    m.add(r12, r12, s3);
    m.sub(r12, r12, s0);
    m.sub(r12, r12, s4);
}

static __device__ int fp3_equal(fixnum x10, fixnum x11, fixnum x12, fixnum y10, fixnum y11, fixnum y12) {
    if (fixnum::cmp(x10, y10) == 0 && fixnum::cmp(x11, y11) == 0 && fixnum::cmp(x12, y12) == 0) {
        return 1;
    }
    return 0;
}

static __device__ void pq_plus(modnum m, fixnum x10, fixnum x11, fixnum x12, fixnum y10, fixnum y11, fixnum y12, fixnum z10, fixnum z11, fixnum z12, fixnum x20, fixnum x21, fixnum x22, fixnum y20, fixnum y21, fixnum y22, fixnum z20, fixnum z21, fixnum z22, fixnum &x30, fixnum &x31, fixnum &x32, fixnum &y30, fixnum &y31, fixnum &y32, fixnum &z30, fixnum &z31, fixnum &z32) {
#if 0
    printf("6g2 pq_plus\n");
    printf("x1, y1, z1\n");
    dump(x10, 24);
    dump(y10, 24);
    dump(z10, 24);
    printf("x2, y2, z2\n");
    dump(x20, 24);
    dump(y20, 24);
    dump(z20, 24);
#endif
    if (fixnum::is_zero(x10) && fixnum::is_zero(x11) && fixnum::is_zero(x12) && fixnum::is_zero(z10) && fixnum::is_zero(z11) && fixnum::is_zero(z12)) {
        //printf("this zero\n");
        x30 = x20;
        x31 = x21;
        x32 = x22;
        y30 = y20;
        y31 = y21;
        y32 = y22;
        z30 = z20;
        z31 = z21;
        z32 = z22;
        return;
    }
    if (fixnum::is_zero(x20) && fixnum::is_zero(x21) && fixnum::is_zero(x22) && fixnum::is_zero(z20) && fixnum::is_zero(z21) && fixnum::is_zero(z22)) {
        //printf("other zero\n");
        x30 = x10;
        x31 = x11;
        x32 = x12;
        y30 = y10;
        y31 = y11;
        y32 = y12;
        z30 = z10;
        z31 = z11;
        z32 = z12;
        return;
    }
    fixnum temp0, temp1, temp2;
    // X1Z2 = X1*Z2
    fixnum x1z20, x1z21, x1z22;
    fp3_multi(m, x10, x11, x12, z20, z21, z22, x1z20, x1z21, x1z22);
    
    // X2Z1 = X2*Z1
    fixnum x2z10, x2z11, x2z12;
    fp3_multi(m, x20, x21, x22, z10, z11, z12, x2z10, x2z11, x2z12);

    // Y1Z2 = Y1*Z2
    fixnum y1z20, y1z21, y1z22;
    fp3_multi(m, y10, y11, y12, z20, z21, z22, y1z20, y1z21, y1z22);

    // Y2Z1 = Y2*Z1
    fixnum y2z10, y2z11, y2z12;
    fp3_multi(m, y20, y21, y22, z10, z11, z12, y2z10, y2z11, y2z12);

    // if (X1Z2 == X2Z1 && Y1Z2 == Y2Z1)
    if (fp3_equal(x1z20, x1z21, x1z22, x2z10, x2z11, x2z12) && fp3_equal(y1z20, y1z21, y1z22, y2z10, y2z11, y2z12)) {
        p_double(m, x10, x11, x12, y10, y11, y12, z10, z11, z12, x30, x31, x32, y30, y31, y32, z30, z31, z32);
        return;
    }

    // Z1Z2 = Z1*Z2
    fixnum z1z20, z1z21, z1z22;
    fp3_multi(m, z10, z11, z12, z20, z21, z22, z1z20, z1z21, z1z22);

    // u    = Y2Z1-Y1Z2
    fixnum u0, u1, u2;
    fp3_sub(m, y2z10, y2z11, y2z12, y1z20, y1z21, y1z22, u0, u1, u2);

    // uu   = u^2
    fixnum uu0, uu1, uu2;
    fp3_square(m, u0, u1, u2, uu0, uu1, uu2);

    // v    = X2*Z1-X1Z2
    fixnum v0, v1, v2;
    fp3_sub(m, x2z10, x2z11, x2z12, x1z20, x1z21, x1z22, v0, v1, v2);

    // vv   = v^2
    fixnum vv0, vv1, vv2;
    fp3_square(m, v0, v1, v2, vv0, vv1, vv2);

    // vvv  = v*vv
    fixnum vvv0, vvv1, vvv2;
    fp3_multi(m, v0, v1, v2, vv0, vv1, vv2, vvv0, vvv1, vvv2);

    // R    = vv*X1Z2
    fixnum R0, R1, R2;
    fp3_multi(m, vv0, vv1, vv2, x1z20, x1z21, x1z22, R0, R1, R2);

    // A    = uu*Z1Z2 - vvv - 2*R
    fixnum A0, A1, A2;
    fp3_multi(m, uu0, uu1, uu2, z1z20, z1z21, z1z22, A0, A1, A2);
    fp3_add(m, vvv0, vvv1, vvv2, R0, R1, R2, temp0, temp1, temp2);
    fp3_add(m, temp0, temp1, temp2, R0, R1, R2, temp0, temp1, temp2);
    fp3_sub(m, A0, A1, A2, temp0, temp1, temp2, A0, A1, A2);

    // X3   = v*A
    fp3_multi(m, v0, v1, v2, A0, A1, A2, x30, x31, x32);

    // Y3   = u*(R-A) - vvv*Y1Z2
    fp3_sub(m, R0, R1, R2, A0, A1, A2, y30, y31, y32);
    fp3_multi(m, u0, u1, u2, y30, y31, y32, y30, y31, y32);
    fp3_multi(m, vvv0, vvv1, vvv2, y1z20, y1z21, y1z22, temp0, temp1, temp2);
    fp3_sub(m, y30, y31, y32, temp0, temp1, temp2, y30, y31, y32);

    // Z3   = vvv*Z1Z2
    fp3_multi(m, vvv0, vvv1, vvv2, z1z20, z1z21, z1z22, z30, z31, z32);

#if 0
    printf("pq x3, y3, z3:\n");
    dump(x30, 24);
    dump(x31, 24);
    dump(x32, 24);
    dump(y30, 24);
    dump(y31, 24);
    dump(y32, 24);
    dump(z30, 24);
    dump(z31, 24);
    dump(z32, 24);
#endif
}

static __device__ void multi_by_a(modnum m, fixnum in0, fixnum in1, fixnum in2, fixnum &r0, fixnum &r1, fixnum &r2) {
    // mnt6_twist_mul_by_a_c0 * elt.c1, mnt6_twist_mul_by_a_c1 * elt.c2,     mnt6_twist_mul_by_a_c2 * elt.c0
    // mnt6_twist_mul_by_a_c0 = mnt6_G1::coeff_a * mnt6_Fq3::non_residue;
    // mnt6_twist_mul_by_a_c1 = mnt6_G1::coeff_a * mnt6_Fq3::non_residue;
    // mnt6_twist_mul_by_a_c2 = mnt6_G1::coeff_a;

    // mnt6753_G1::coeff_a = mnt6753_Fq("11");
    // mnt6753_Fq3::non_residue = mnt6753_Fq("11");
    int c01n = 121;   //11*11
    int c2n = 11;
    fixnum temp;
    temp = in1;
    for (int i = 0; i < c01n - 1; i ++) {
        m.add(r0, temp, in1); 
        temp = r0;
    } 
    temp = in2;
    for (int i = 0; i < c01n - 1; i ++) {
        m.add(r1, temp, in2); 
        temp = r1;
    } 
    temp = in0;
    for (int i = 0; i < c2n - 1; i ++) {
        m.add(r2, temp, in0); 
        temp = r2;
    } 
}

static __device__ void p_double(modnum m, fixnum x10, fixnum x11, fixnum x12, fixnum y10, fixnum y11, fixnum y12, fixnum z10, fixnum z11, fixnum z12, fixnum &x30, fixnum &x31, fixnum &x32, fixnum &y30, fixnum &y31, fixnum &y32, fixnum &z30, fixnum &z31, fixnum &z32) {
#if 0
    printf("6g2 q double\n");
    printf("x1, y1, z1\n");
    dump(x10, 24);
    dump(y10, 24);
    dump(z10, 24);
#endif
    if (fixnum::is_zero(x10) && fixnum::is_zero(x11) && fixnum::is_zero(x12) && fixnum::is_zero(z10) && fixnum::is_zero(z11) && fixnum::is_zero(z12)) {
        //printf("this zero\n");
        x30 = x10;
        x31 = x11;
        x32 = x12;
        y30 = y10;
        y31 = y11;
        y32 = y12;
        z30 = z10;
        z31 = z11;
        z32 = z12;
        return;
    }
    fixnum temp0, temp1, temp2;

    // XX  = X1^2
    fixnum xx0, xx1, xx2;
    fp3_square(m, x10, x11, x12, xx0, xx1, xx2);

    // ZZ  = Z1^2
    fixnum zz0, zz1, zz2;
    fp3_square(m, z10, z11, z12, zz0, zz1, zz2);

    // w   = a*ZZ + 3*XX
    fixnum w0, w1, w2;
    multi_by_a(m, zz0, zz1, zz2, w0, w1, w2);
    fp3_add(m, w0, w1, w2, xx0, xx1, xx2, w0, w1, w2);
    fp3_add(m, w0, w1, w2, xx0, xx1, xx2, w0, w1, w2);
    fp3_add(m, w0, w1, w2, xx0, xx1, xx2, w0, w1, w2);

    // Y1Z1 = Y1*Z1
    fixnum y1z10, y1z11, y1z12;
    fp3_multi(m, y10, y11, y12, z10, z11, z12, y1z10, y1z11, y1z12);

    //s   = 2*Y1*Z1
    fixnum s0, s1, s2;
    fp3_add(m, y1z10, y1z11, y1z12, y1z10, y1z11, y1z12, s0, s1, s2);

    // ss  = s^2
    fixnum ss0, ss1, ss2;
    fp3_square(m, s0, s1, s2, ss0, ss1, ss2);

    // sss = s*ss
    fixnum sss0, sss1, sss2;
    fp3_multi(m, s0, s1, s2, ss0, ss1, ss2, sss0, sss1, sss2);

    // R   = Y1*s
    fixnum R0, R1, R2;
    fp3_multi(m, y10, y11, y12, s0, s1, s2, R0, R1, R2);

    // RR  = R^2
    fixnum RR0, RR1, RR2;
    fp3_square(m, R0, R1, R2, RR0, RR1, RR2);

    // B   = (X1+R)^2 - XX - RR
    fixnum B0, B1, B2;
    fp3_add(m, x10, x11, x12, R0, R1, R2, B0, B1, B2);
    fp3_square(m, B0, B1, B2, B0, B1, B2);
    fp3_sub(m, B0, B1, B2, xx0, xx1, xx2, B0, B1, B2);
    fp3_sub(m, B0, B1, B2, RR0, RR1, RR2, B0, B1, B2);

    // h   = w^2-2*B
    fixnum h0, h1, h2;
    fp3_square(m, w0, w1, w2, h0, h1, h2);
    fp3_sub(m, h0, h1, h2, B0, B1, B2, h0, h1, h2);
    fp3_sub(m, h0, h1, h2, B0, B1, B2, h0, h1, h2);

    // X3  = h*s
    fp3_multi(m, h0, h1, h2, s0, s1, s2, x30, x31, x32);

    // Y3  = w*(B-h) - 2*RR
    fp3_sub(m, B0, B1, B2, h0, h1, h2, y30, y31, y32);
    fp3_multi(m, w0, w1, w2, y30, y31, y32, y30, y31, y32);
    fp3_sub(m, y30, y31, y32, RR0, RR1, RR2, y30, y31, y32);
    fp3_sub(m, y30, y31, y32, RR0, RR1, RR2, y30, y31, y32);

    // Z3  = sss
    z30 = sss0;
    z31 = sss1;
    z32 = sss2;
#if 0
    printf("dbl x3, y3, z3:\n");
    dump(x30, 24);
    dump(y30, 24);
    dump(z30, 24);
#endif
}

};
}
