#include <stdint.h>
#include <cuda_runtime.h>

#include "GPUMath.h"

#ifndef STEP_THREADS
#define STEP_THREADS 128
#endif

#define STEP_MAX_HALF 64
#define STEP_MAX_SYMBOLS 52
#define STEP_RECORD 10

struct StepParams {
    uint32_t threads;
    uint32_t iters;
    uint32_t bits;
    uint32_t count;
    uint32_t maxHits;
    uint32_t pad;
    uint32_t at[STEP_MAX_SYMBOLS];
    uint32_t want[STEP_MAX_SYMBOLS];
};

__device__ __forceinline__ void fcopy(uint64_t *d, const uint64_t *s) {
    d[0] = s[0]; d[1] = s[1]; d[2] = s[2]; d[3] = s[3];
}

__device__ __forceinline__ void fload(uint64_t *d, const uint64_t *table, uint32_t i, uint32_t which) {
    const uint64_t *src = table + (size_t)i * 8 + which * 4;
    d[0] = __ldg(src); d[1] = __ldg(src + 1); d[2] = __ldg(src + 2); d[3] = __ldg(src + 3);
}

__device__ __forceinline__ void freduce(uint64_t *a) {
    uint64_t t[5];
    t[0] = a[0]; t[1] = a[1]; t[2] = a[2]; t[3] = a[3]; t[4] = 0;
    SubP(t);
    if (_IsPositive(t)) { a[0] = t[0]; a[1] = t[1]; a[2] = t[2]; a[3] = t[3]; }
}

__device__ __forceinline__ void finv(uint64_t *a) {
    uint64_t t[5];
    t[0] = a[0]; t[1] = a[1]; t[2] = a[2]; t[3] = a[3]; t[4] = 0;
    _ModInv(t);
    a[0] = t[0]; a[1] = t[1]; a[2] = t[2]; a[3] = t[3];
}

__device__ __forceinline__ uint32_t symbolAt(const uint64_t *x, uint32_t i, uint32_t bits) {
    const uint32_t pos = i * bits;
    const uint32_t limb = 3 - (pos >> 6);
    const uint32_t off = pos & 63;
    uint64_t w = x[limb] << off;
    if (off > 0 && limb > 0) w |= x[limb - 1] >> (64 - off);
    return (uint32_t)(w >> (64 - bits));
}

__device__ __forceinline__ void check(const uint64_t *xr, uint32_t t, uint32_t code, const StepParams &p, uint32_t *out) {
    uint64_t x[4];
    fcopy(x, xr);
    freduce(x);
    for (uint32_t s = 0; s < p.count; s++) {
        if (symbolAt(x, p.at[s], p.bits) != p.want[s]) return;
    }
    const uint32_t k = atomicAdd(&out[0], 1u);
    if (k < p.maxHits) {
        uint32_t *rec = out + 1 + k * STEP_RECORD;
        rec[0] = t;
        rec[1] = code;
        for (int i = 0; i < 4; i++) {
            rec[2 + 2 * i] = (uint32_t)x[i];
            rec[3 + 2 * i] = (uint32_t)(x[i] >> 32);
        }
    }
}

template <int HALF>
__device__ __forceinline__ void stepKernel(const StepParams &p, const uint64_t *table, uint64_t *state, uint32_t *out) {
    const uint32_t t = blockIdx.x * STEP_THREADS + threadIdx.x;
    if (t >= p.threads) return;

    uint64_t px[4], py[4];
    for (int k = 0; k < 4; k++) {
        px[k] = state[(size_t)t * 8 + k];
        py[k] = state[(size_t)t * 8 + 4 + k];
    }

    for (uint32_t it = 0; it < p.iters; it++) {
        const uint32_t c = it * (2u * HALF + 1u) + HALF;
        check(px, t, c, p, out);

        uint64_t pref[HALF + 1][4];
        uint64_t g[4], dx[4];
        fload(g, table, 0, 0);
        _ModSub256(pref[0], g, px);
        #pragma unroll 1
        for (int i = 1; i <= HALF; i++) {
            fload(g, table, i, 0);
            _ModSub256(dx, g, px);
            _ModMult(pref[i], pref[i - 1], dx);
        }
        uint64_t inv[4];
        fcopy(inv, pref[HALF]);
        freduce(inv);
        finv(inv);

        uint64_t gx[4], gy[4], lam[4], tmp[4], nx[4], ny[4], jumpInv[4];
        fload(gx, table, HALF, 0);
        fload(gy, table, HALF, 1);
        _ModMult(jumpInv, inv, pref[HALF - 1]);
        _ModSub256(dx, gx, px);
        _ModMult(inv, dx);
        _ModSub256(tmp, gy, py);
        _ModMult(lam, tmp, jumpInv);
        _ModSqr(nx, lam);
        _ModSub256(nx, px);
        _ModSub256(nx, gx);
        _ModSub256(tmp, px, nx);
        _ModMult(ny, lam, tmp);
        _ModSub256(ny, py);

        #pragma unroll 1
        for (int i = HALF - 1; i >= 0; i--) {
            fload(gx, table, i, 0);
            fload(gy, table, i, 1);
            uint64_t invI[4];
            if (i > 0) {
                _ModMult(invI, inv, pref[i - 1]);
                _ModSub256(dx, gx, px);
                _ModMult(inv, dx);
            } else {
                fcopy(invI, inv);
            }
            uint64_t x3[4];
            _ModSub256(tmp, gy, py);
            _ModMult(lam, tmp, invI);
            _ModSqr(x3, lam);
            _ModSub256(x3, px);
            _ModSub256(x3, gx);
            check(x3, t, c + (uint32_t)i + 1u, p, out);

            _ModNeg256(tmp, gy);
            _ModSub256(tmp, py);
            _ModMult(lam, tmp, invI);
            _ModSqr(x3, lam);
            _ModSub256(x3, px);
            _ModSub256(x3, gx);
            check(x3, t, c - (uint32_t)i - 1u, p, out);
        }
        fcopy(px, nx);
        fcopy(py, ny);
    }
    for (int k = 0; k < 4; k++) {
        state[(size_t)t * 8 + k] = px[k];
        state[(size_t)t * 8 + 4 + k] = py[k];
    }
}

#define NGRIND_STEP(H) \
    extern "C" __global__ void __launch_bounds__(STEP_THREADS) \
    ngrind_step_##H(const StepParams p, const uint64_t *table, uint64_t *state, uint32_t *out) { \
        stepKernel<H>(p, table, state, out); \
    }

NGRIND_STEP(4)
NGRIND_STEP(8)
NGRIND_STEP(16)
NGRIND_STEP(32)
NGRIND_STEP(64)
