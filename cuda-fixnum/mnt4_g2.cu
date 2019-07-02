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
class mnt4_g2 {
public:
static __device__ void dump(fixnum n, int size) {
#if 0
	for (int i = 0; i < size; i++) {
		printf("DUMP [%d] %x\n", i, fixnum::get(n, i));
	}
#endif
	printf("dump [%d]=\%04x\n", threadIdx.x, fixnum::get(n, threadIdx.x));
}

typedef modnum_monty_cios<fixnum> modnum;
static __device__ void fp2_multi(modnum m, fixnum a, fixnum b, fixnum A, fixnum B, fixnum &r10, fixnum &r11) {
    fixnum aA, bB;
    // aA = a * A
    m.mul(aA, a, A);
    m.mul(bB, b, B);
    
    // r10 = aA + non_residue * bB
    // non_residue = 13
    fixnum temp, temp2;
    temp2= bB;
    for (int i = 0; i < 12; i++) {
        m.add(temp, temp2, bB);
        temp2 = temp;
    }
    m.add(r10, aA, temp);

    // r11 = (a + b)*(A+B) - aA - bB)    
    fixnum ab, AB;
    m.add(ab, a, b);
    m.add(AB, A, B);
    m.mul(temp, ab, AB);
    m.sub(temp, temp, aA);
    m.sub(r11, temp, bB);
}

static __device__ void fp2_sub(modnum m, fixnum x10, fixnum x11, fixnum y10, fixnum y11, fixnum &r10, fixnum &r11) {
    m.sub(r10, x10, y10);

    m.sub(r11, x11, y11); 
}

static __device__ void fp2_add(modnum m, fixnum x10, fixnum x11, fixnum y10, fixnum y11, fixnum &r10, fixnum &r11) {
    
    m.add(r10, x10, y10);

    m.add(r11, x11, y11); 
}

static __device__ void fp2_square(modnum m, fixnum x10, fixnum x11, fixnum &r10, fixnum &r11) {
    fixnum a = x10;
    fixnum b = x11;

    // ab = a*b
    fixnum ab;
    m.mul(ab, a, b);

    // r10 = (a + b) * (a + non_residue * b) - ab - non_residue * ab
    // non_residue * b
    fixnum temp, temp2;
    temp2 = b;
    for (int i = 0; i < 12; i++) {
        m.add(temp, temp2, b);
        temp2 = temp;
    }
    // a + non_residue * b
    m.add(temp, a, temp);
    // a + b
    fixnum apb;
    m.add(apb, a, b);
    // (a + b) * (a + non_residue * b)
    m.mul(temp, apb, temp);
    // - ab
    m.sub(temp, temp, ab);
    // - non_residue * ab
    fixnum temp3 = ab;
    for (int i = 0; i < 12; i++) {
        m.add(temp2, temp3, ab);
        temp3 = temp2;
    }
    m.sub(r10, temp, temp2);

    // r11 = ab + ab
    m.add(r11, ab, ab);
}

static __device__ int fp2_equal(fixnum x10, fixnum x11, fixnum y10, fixnum y11) {
    if (fixnum::cmp(x10, y10) == 0 && fixnum::cmp(x11, y11) == 0) {
        return 1;
    }
    return 0;
}

static __device__ void pq_plus(modnum m, fixnum x10, fixnum x11, fixnum y10, fixnum y11, fixnum z10, fixnum z11, fixnum x20, fixnum x21, fixnum y20, fixnum y21, fixnum z20, fixnum z21, fixnum &x30, fixnum &x31, fixnum &y30, fixnum &y31, fixnum &z30, fixnum &z31) {
#if 0
    printf("4g2 pq_plus\n");
    printf("x1, y1, z1\n");
    dump(x10, 24);
    dump(x11, 24);
    dump(y10, 24);
    dump(y11, 24);
    dump(z10, 24);
    dump(z11, 24);
    printf("x2, y2, z2\n");
    dump(x20, 24);
    dump(x21, 24);
    dump(y20, 24);
    dump(y21, 24);
    dump(z20, 24);
    dump(z21, 24);
#endif
    if (fixnum::is_zero(x10) && fixnum::is_zero(x11) && fixnum::is_zero(z10) && fixnum::is_zero(z11)) {
        //printf("this zero\n");
        x30 = x20;
        x31 = x21;
        y30 = y20;
        y31 = y21;
        z30 = z20;
        z31 = z21;
        return;
    }
    if (fixnum::is_zero(x20) && fixnum::is_zero(x21) && fixnum::is_zero(z20) && fixnum::is_zero(z21)) {
        //printf("other zero\n");
        x30 = x10;
        x31 = x11;
        y30 = y10;
        y31 = y11;
        z30 = z10;
        z31 = z11;
        return;
    }
    fixnum temp0, temp1;
    // X1Z2 = X1*Z2
    fixnum x1z20, x1z21;
    fp2_multi(m, x10, x11, z20, z21, x1z20, x1z21);

    // X2Z1 = X2*Z1
    fixnum x2z10, x2z11;
    fp2_multi(m, x20, x21, z10, z11, x2z10, x2z11);

    // Y1Z2 = Y1*Z2
    fixnum y1z20, y1z21;
    fp2_multi(m, y10, y11, z20, z21, y1z20, y1z21);

    // Y2Z1 = Y2*Z1
    fixnum y2z10, y2z11;
    fp2_multi(m, y20, y21, z10, z11, y2z10, y2z11);

    // if (X1Z2 == X2Z1 && Y1Z2 == Y2Z1)
    if (fp2_equal(x1z20, x1z21, x2z10, x2z11) && fp2_equal(y1z20, y1z21, y2z10, y2z11)) {
        p_double(m, x10, x11, y10, y11, z10, z11, x30, x31, y30, y31, z30, z31);
        return;
    }

    // Z1Z2 = Z1*Z2
    fixnum z1z20, z1z21;
    fp2_multi(m, z10, z11, z20, z21, z1z20, z1z21);

    // u    = Y2Z1-Y1Z2
    fixnum u0, u1;
    fp2_sub(m, y2z10, y2z11, y1z20, y1z21, u0, u1);

    // uu   = u^2
    fixnum uu0, uu1;
    fp2_square(m, u0, u1, uu0, uu1);
    
    // v    = X2Z1-X1Z2 
    fixnum v0, v1;
    fp2_sub(m, x2z10, x2z11, x1z20, x1z21, v0, v1);

    // vv   = v^2
    fixnum vv0, vv1;
    fp2_square(m, v0, v1, vv0, vv1);

    // vvv  = v*vv
    fixnum vvv0, vvv1;
    fp2_multi(m, v0, v1, vv0, vv1, vvv0, vvv1);

    // R    = vv*X1Z2
    fixnum R0, R1;
    fp2_multi(m, vv0, vv1, x1z20, x1z21, R0, R1);

    // A    = uu*Z1Z2 - vvv - 2*R
    fixnum A0, A1;
    fp2_multi(m, uu0, uu1, z1z20, z1z21, A0, A1);
    fp2_add(m, vvv0, vvv1, R0, R1, temp0, temp1);
    fp2_add(m, temp0, temp1, R0, R1, temp0, temp1);
    fp2_sub(m, A0, A1, temp0, temp1, A0, A1);

    // X3   = v*A
    fp2_multi(m, v0, v1, A0, A1, x30, x31);

    // Y3   = u*(R-A) - vvv*Y1Z2
    fp2_sub(m, R0, R1, A0, A1, temp0, temp1);
    fp2_multi(m, u0, u1, temp0, temp1, y30, y31);
    fp2_multi(m, vvv0, vvv1, y1z20, y1z21, temp0, temp1);
    fp2_sub(m, y30, y31, temp0, temp1, y30, y31); 

    // Z3   = vvv*Z1Z2
    fp2_multi(m, vvv0, vvv1, z1z20, z1z21, z30, z31);

#if 0
    printf("pq x3, y3, z3:\n");
    dump(x30, 24);
    dump(x31, 24);
    dump(y30, 24);
    dump(y31, 24);
    dump(z30, 24);
    dump(z31, 24);
#endif
}

static __device__ void multi_by_a(modnum m, fixnum in0, fixnum in1, fixnum &r0, fixnum &r1) {
    // mnt4_twist_mul_by_a_c0 * elt.c0, mnt4_twist_mul_by_a_c1 * elt.c1)
    // mnt4_twist_mul_by_a_c0 = mnt4_G1::coeff_a * mnt4_Fq2::non_residue;
    // mnt4_twist_mul_by_a_c1 = mnt4_G1::coeff_a * mnt4_Fq2::non_residue;
    // mnt4_G1::coeff_a = mnt4_Fq("2");
    // mnt4_Fq2::non_residue = mnt4_Fq("13");
    int an = 26;   //2*13
    fixnum temp;
    temp = in0;
    for (int i = 0; i < an - 1; i ++) {
        m.add(r0, temp, in0); 
        temp = r0;
    } 
    temp = in1;
    for (int i = 0; i < an - 1; i ++) {
        m.add(r1, temp, in1); 
        temp = r1;
    } 
}

static __device__ void p_double(modnum m, fixnum x10, fixnum x11, fixnum y10, fixnum y11, fixnum z10, fixnum z11, fixnum &x30, fixnum &x31, fixnum &y30, fixnum &y31, fixnum &z30, fixnum &z31) {
#if 0
    printf("4g2 q double\n");
    printf("x1, y1, z1\n");
    dump(x10, 24);
    dump(x11, 24);
    dump(y10, 24);
    dump(y11, 24);
    dump(z10, 24);
    dump(z11, 24);
#endif
    if (fixnum::is_zero(x10) && fixnum::is_zero(x11) && fixnum::is_zero(z10) && fixnum::is_zero(z11)) {
        //printf("this zero\n");
        x30 = x10;
        x31 = x11;
        y30 = y10;
        y31 = y11;
        z30 = z10;
        z31 = z11;
        return;
    }
    fixnum temp0, temp1;
    // XX  = X1^2 
    fixnum xx0, xx1;
    fp2_square(m, x10, x11, xx0, xx1);

    // ZZ  = Z1^2
    fixnum zz0, zz1;
    fp2_square(m, z10, z11, zz0, zz1);

    // w   = a*ZZ + 3*XX
    fixnum w0, w1;
    fp2_add(m, xx0, xx1, xx0, xx1, w0, w1);
    fp2_add(m, xx0, xx1, w0, w1, w0, w1);
    multi_by_a(m, zz0, zz1, temp0, temp1);
    fp2_add(m, w0, w1, temp0, temp1, w0, w1);

    // y1z1
    fixnum y1z10, y1z11;
    fp2_multi(m, y10, y11, z10, z11, y1z10, y1z11);

    // s   = 2*Y1*Z1
    fixnum s0, s1;
    fp2_add(m, y1z10, y1z11, y1z10, y1z11, s0, s1);

    // ss  = s^2
    fixnum ss0, ss1;
    fp2_square(m, s0, s1, ss0, ss1);

    // sss  = s*ss
    fixnum sss0, sss1;
    fp2_multi(m, s0, s1, ss0, ss1, sss0, sss1);

    // R   = Y1*s
    fixnum R0, R1;
    fp2_multi(m, y10, y11, s0, s1, R0, R1);

    // RR  = R^2
    fixnum RR0, RR1;
    fp2_square(m, R0, R1, RR0, RR1);

    // B   = (X1+R)^2 - XX - RR
    fixnum B0, B1;
    fp2_add(m, x10, x11, R0, R1, temp0, temp1);
    fp2_square(m, temp0, temp1, B0, B1);
    fp2_sub(m, B0, B1, xx0, xx1, B0, B1); 
    fp2_sub(m, B0, B1, RR0, RR1, B0, B1);

    // h   = w^2-2*B
    fixnum h0, h1;
    fp2_square(m, w0, w1, h0, h1);
    fp2_add(m, B0, B1, B0, B1, temp0, temp1);
    fp2_sub(m, h0, h1, temp0, temp1, h0, h1);

    // X3  = h*s
    fp2_multi(m, h0, h1, s0, s1, x30, x31);

    // Y3  = w*(B-h) - 2*RR
    fp2_sub(m, B0, B1, h0, h1, y30, y31);
    fp2_multi(m, w0, w1, y30, y31, y30, y31);
    fp2_add(m, RR0, RR1, RR0, RR1, temp0, temp1);
    fp2_sub(m, y30, y31, temp0, temp1, y30, y31);

    // Z3  = sss
    z30 = sss0;
    z31 = sss1;
#if 0
    printf("dbl x3, y3, z3:\n");
    dump(x30, 24);
    dump(x31, 24);
    dump(y30, 24);
    dump(y31, 24);
    dump(z30, 24);
    dump(z31, 24);
#endif
}

};
}
