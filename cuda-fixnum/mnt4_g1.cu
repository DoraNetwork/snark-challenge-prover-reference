#include <cstdio>
#include <cstring>
#include <cassert>

#include "fixnum/warp_fixnum.cu"
#include "array/fixnum_array.h"
#include "functions/modexp.cu"
#include "functions/multi_modexp.cu"
#include "modnum/modnum_monty_redc.cu"
#include "modnum/modnum_monty_cios.cu"

//#define USE_ADD_2007_BL
#define USE_ADD_1998_CMO_2

using namespace std;
using namespace cuFIXNUM;

namespace MNT_G{
template< typename fixnum >
class mnt4_g1 {
public:
__device__ void dump(fixnum n, int size) {
#if 0
	for (int i = 0; i < size; i++) {
		printf("DUMP [%d] %x\n", i, fixnum::get(n, i));
	}
#endif
	printf("dump [%d]=\%x\n", threadIdx.x, fixnum::get(n, threadIdx.x));
}

typedef modnum_monty_cios<fixnum> modnum;
static __device__ void pq_plus(modnum m, fixnum x1, fixnum y1, fixnum z1, fixnum x2, fixnum y2, fixnum z2, fixnum &x3, fixnum &y3, fixnum &z3) {
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

#ifdef USE_ADD_1998_CMO_2
    // http://www.hyperelliptic.org/EFD/g1p/auto-shortw-projective.html#addition-add-1998-cmo-2
    fixnum y1z2, x1z2, z1z2, y2z1, x2z1;

    // x1z2
    m.mul(x1z2, x1, z2);

    // x2z1
    m.mul(x2z1, x2, z1);

    // y1z2
    m.mul(y1z2, y1, z2);

    // y2z1
    m.mul(y2z1, y2, z1);

    if (fixnum::cmp(x1z2, x2z1) == 0 && fixnum::cmp(y1z2, y2z1) == 0) {
        p_double(m, x1, y1, z1, x3, y3, z3);
        return;
    }
    // z1z2
    m.mul(z1z2, z1, z2);

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
    m.sub(y3, R, A);
    m.mul(y3, u, y3);
    m.mul(temp, vvv, y1z2);
    m.sub(y3, y3, temp);

    // z3 = vvv*Z1Z2
    m.mul(z3, vvv, z1z2);
#endif

#ifdef USE_ADD_2007_BL
    // http://www.hyperelliptic.org/EFD/g1p/auto-shortw-projective.html#addition-add-2007-bl
    fixnum u1, u2, s1, s2, zz, t, tt, em, r, f, l, ll, g, w;

    // U1 = X1*Z2
    m.mul(u1, x1, z2);
    // U2 = X2*Z1
    m.mul(u2, x2, z1);
    // S1 = Y1*Z2
    m.mul(s1, y1, z2);
    // S2 = Y2*Z1
    m.mul(s2, y2, z1);
    // ZZ = Z1*Z2
    m.mul(zz, z1, z2);
    // T = U1+U2
    m.add(t, u1, u2);
    // TT = T^2
    m.mul(tt, t, t);
    // M = S1+S2
    m.add(em, s1, s2);
    // R = TT-U1*U2+a*ZZ^2, a=2
    fixnum u1u2, zz2, azz2, tt_u1u2;
    m.mul(u1u2, u1, u2);
    m.mul(zz2, zz, zz);
    m.add(azz2, zz2, zz2);
    m.sub(tt_u1u2, tt, u1u2);
    m.add(r, tt_u1u2, azz2);
    // F = ZZ*M
    m.mul(f, zz, em);
    // L = M*F
    m.mul(l, em, f);
    // LL = L^2
    m.mul(ll, l, l);
    // G = (T+L)^2-TT-LL
    fixnum tl, tl2, tl2_tt;
    m.add(tl, t, l);
    m.mul(tl2, tl, tl);
    m.sub(tl2_tt, tl2, tt);
    m.sub(g, tl2_tt, ll);
    // W = 2*R^2-G
    fixnum rr, rr2;
    m.mul(rr, r, r);
    m.add(rr2, rr, rr);
    m.sub(w, rr2, g);
    // X3 = 2*F*W
    fixnum fw;
    m.mul(fw, f, w);
    m.add(x3, fw, fw);
    // Y3 = R*(G-2*W)-2*LL
    fixnum w2, g_w2, rg_w2, ll2;
    m.add(w2, w, w);
    m.sub(g_w2, g, w2);
    m.mul(rg_w2, r, g_w2);
    m.add(ll2, ll, ll);
    m.sub(y3, rg_w2, ll2);
    // Z3 = 4*F*F^2
    fixnum f2, ff2, ff2_2;
    m.mul(f2, f, f);
    m.mul(ff2, f, f2);
    m.add(ff2_2, ff2, ff2);
    m.add(z3, ff2_2, ff2_2);
#endif
}

static __device__ void p_double(modnum m, fixnum x1, fixnum y1, fixnum z1, fixnum &x3, fixnum &y3, fixnum &z3) {
    if (fixnum::is_zero(x1) && fixnum::is_zero(z1)) {
        x3 = x1;
        y3 = y1;
        z3 = z1;
        return;
    }

    // http://www.hyperelliptic.org/EFD/g1p/auto-shortw-projective.html#doubling-dbl-2007-b
    fixnum XX;
    // XX = X1*X1
    m.mul(XX, x1, x1);

    fixnum ZZ;
    // ZZ = Z1*Z1
    m.mul(ZZ, z1, z1);

    fixnum w;
    // w = a*ZZ+3*XX    a = 2
    m.add(w, ZZ, ZZ);
    m.add(w, w, XX);
    m.add(w, w, XX);
    m.add(w, w, XX);

    fixnum s;
    // s = 2*Y1*Z1
    m.mul(s, y1, z1);
    m.add(s, s, s);

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
    m.add(B, x1, R);
    m.mul(B, B, B);
    m.sub(B, B, XX);
    m.sub(B, B, RR);

    fixnum h;
    // h = w*2-2*B
    m.mul(h, w, w);
    m.sub(h, h, B);
    m.sub(h, h, B);

    // X3 = h * s
    m.mul(x3, h, s);

    // Y3 = w*(B-h)-2*RR
    m.sub(y3, B, h);
    m.mul(y3, w, y3);
    m.sub(y3, y3, RR);
    m.sub(y3, y3, RR);

    //  Z3 = sss
    z3 = sss;
}

};
}
