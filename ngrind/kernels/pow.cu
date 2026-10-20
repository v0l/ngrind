#include <stdint.h>
#include <cuda_runtime.h>

#ifndef POW_THREADS_PER_BLOCK
#define POW_THREADS_PER_BLOCK 256
#endif

#ifndef POW_LAUNCH_WAVES
#define POW_LAUNCH_WAVES 1
#endif

#define NONCES_PER_UNIT 1000
#define UNITS_PER_HIGH 10
#define LOW_DIGITS 4
#define MIN_NONCE_DIGITS 12
#define MAX_NONCE_DIGITS 20

__constant__ uint32_t c_K[64] = {
    0x428a2f98, 0x71374491, 0xb5c0fbcf, 0xe9b5dba5, 0x3956c25b, 0x59f111f1, 0x923f82a4, 0xab1c5ed5,
    0xd807aa98, 0x12835b01, 0x243185be, 0x550c7dc3, 0x72be5d74, 0x80deb1fe, 0x9bdc06a7, 0xc19bf174,
    0xe49b69c1, 0xefbe4786, 0x0fc19dc6, 0x240ca1cc, 0x2de92c6f, 0x4a7484aa, 0x5cb0a9dc, 0x76f988da,
    0x983e5152, 0xa831c66d, 0xb00327c8, 0xbf597fc7, 0xc6e00bf3, 0xd5a79147, 0x06ca6351, 0x14292967,
    0x27b70a85, 0x2e1b2138, 0x4d2c6dfc, 0x53380d13, 0x650a7354, 0x766a0abb, 0x81c2c92e, 0x92722c85,
    0xa2bfe8a1, 0xa81a664b, 0xc24b8b70, 0xc76c51a3, 0xd192e819, 0xd6990624, 0xf40e3585, 0x106aa070,
    0x19a4c116, 0x1e376c08, 0x2748774c, 0x34b0bcb5, 0x391c0cb3, 0x4ed8aa4a, 0x5b9cca4f, 0x682e6ff3,
    0x748f82ee, 0x78a5636f, 0x84c87814, 0x8cc70208, 0x90befffa, 0xa4506ceb, 0xbef9a3f7, 0xc67178f2
};

struct PowOutput {
    uint64_t nonce;
    uint32_t hash[8];
    int found;
    int pad;
};

#define POW_PARAM_TRAIL 6

struct PowParams {
    uint32_t midstate[8];
    uint32_t tpl[32];
    uint32_t kw[POW_PARAM_TRAIL * 64];
    const uint32_t *trailKW;
    PowOutput *out;
    uint64_t unitStart;
    uint64_t highBase;
    uint32_t unitCount;
    int nTrail;
    int highDigits;
    int target;
    int firstTarget;
};

__device__ __forceinline__ uint32_t sha_rotr(uint32_t x, uint32_t n) {
    return __funnelshift_r(x, x, n);
}

__device__ __forceinline__ uint32_t sha_bsig0(uint32_t x) { return sha_rotr(x, 2) ^ sha_rotr(x, 13) ^ sha_rotr(x, 22); }
__device__ __forceinline__ uint32_t sha_bsig1(uint32_t x) { return sha_rotr(x, 6) ^ sha_rotr(x, 11) ^ sha_rotr(x, 25); }
__device__ __forceinline__ uint32_t sha_ssig0(uint32_t x) { return sha_rotr(x, 7) ^ sha_rotr(x, 18) ^ (x >> 3); }
__device__ __forceinline__ uint32_t sha_ssig1(uint32_t x) { return sha_rotr(x, 17) ^ sha_rotr(x, 19) ^ (x >> 10); }
__device__ __forceinline__ uint32_t sha_ch(uint32_t x, uint32_t y, uint32_t z) { return z ^ (x & (y ^ z)); }
__device__ __forceinline__ uint32_t sha_maj(uint32_t x, uint32_t y, uint32_t z) { return (x & y) | (z & (x | y)); }

__device__ __forceinline__ void sha_round(uint32_t (&v)[8], int i, uint32_t kw) {
    uint32_t &a = v[(0 - i) & 7], &b = v[(1 - i) & 7], &c = v[(2 - i) & 7], &d = v[(3 - i) & 7];
    uint32_t &e = v[(4 - i) & 7], &f = v[(5 - i) & 7], &g = v[(6 - i) & 7], &h = v[(7 - i) & 7];
    uint32_t t1 = h + sha_bsig1(e) + sha_ch(e, f, g) + kw;
    uint32_t t2 = sha_bsig0(a) + sha_maj(a, b, c);
    d += t1;
    h = t1 + t2;
}

__device__ __forceinline__ void compress(uint32_t (&st)[8], uint32_t (&w)[16]) {
    uint32_t v[8];
    #pragma unroll
    for (int i = 0; i < 8; i++) v[i] = st[i];
    #pragma unroll
    for (int i = 0; i < 16; i++) sha_round(v, i, c_K[i] + w[i]);
    #pragma unroll
    for (int i = 16; i < 64; i++) {
        w[i & 15] += sha_ssig1(w[(i - 2) & 15]) + w[(i - 7) & 15] + sha_ssig0(w[(i - 15) & 15]);
        sha_round(v, i, c_K[i] + w[i & 15]);
    }
    #pragma unroll
    for (int i = 0; i < 8; i++) st[i] += v[i];
}

template <int B0, int WV>
__device__ __forceinline__ void PowKernel(const PowParams &p) {
    const uint32_t gid = blockIdx.x * POW_THREADS_PER_BLOCK + threadIdx.x;
    if (gid >= p.unitCount) return;

    const uint64_t unit = p.unitStart + gid;
    const uint64_t high = p.highBase + unit / UNITS_PER_HIGH;
    const uint32_t thousands = (uint32_t)(unit % UNITS_PER_HIGH);

    uint32_t msg[(B0 + 1) * 16];
    #pragma unroll
    for (int i = 0; i < (B0 + 1) * 16; i++) msg[i] = p.tpl[i];
    uint64_t rest = high;
    #pragma unroll
    for (int j = 0; j < MAX_NONCE_DIGITS - LOW_DIGITS; j++) {
        const int word = B0 * 16 + WV - 1 - j / 4;
        if (word >= 0 && j < p.highDigits) {
            msg[word] |= (uint32_t)('0' + (int)(rest % 10)) << (8 * (j & 3));
            rest /= 10;
        }
    }

    uint32_t st[8];
    #pragma unroll
    for (int i = 0; i < 8; i++) st[i] = p.midstate[i];
    if (B0 == 1) {
        uint32_t w0[16];
        #pragma unroll
        for (int i = 0; i < 16; i++) w0[i] = msg[i];
        compress(st, w0);
    }

    uint32_t w[16];
    #pragma unroll
    for (int i = 0; i < 16; i++) w[i] = msg[B0 * 16 + i];

    uint32_t pre[8];
    #pragma unroll
    for (int i = 0; i < 8; i++) pre[i] = st[i];
    #pragma unroll
    for (int i = 0; i < WV; i++) sha_round(pre, i, c_K[i] + w[i]);

    const uint32_t lowBase = 0x30303030u | (thousands << 24);
    volatile int *found = &p.out->found;

    for (uint32_t hd = 0; hd < 10; hd++) {
        if (*found) return;
        for (uint32_t td = 0; td < 10; td++) {
            for (uint32_t od = 0; od < 10; od++) {
                uint32_t m[16];
                #pragma unroll
                for (int i = 0; i < 16; i++) m[i] = w[i];
                m[WV] = lowBase | (hd << 16) | (td << 8) | od;

                uint32_t v[8];
                #pragma unroll
                for (int i = 0; i < 8; i++) v[i] = pre[i];
                #pragma unroll
                for (int i = WV; i < 16; i++) sha_round(v, i, c_K[i] + m[i]);
                #pragma unroll
                for (int i = 16; i < 64; i++) {
                    m[i & 15] += sha_ssig1(m[(i - 2) & 15]) + m[(i - 7) & 15] + sha_ssig0(m[(i - 15) & 15]);
                    sha_round(v, i, c_K[i] + m[i & 15]);
                }

                uint32_t s[8];
                #pragma unroll
                for (int i = 0; i < 8; i++) s[i] = st[i] + v[i];

                for (int b = 0; b < p.nTrail; b++) {
                    uint32_t y[8];
                    #pragma unroll
                    for (int i = 0; i < 8; i++) y[i] = s[i];
                    if (b < POW_PARAM_TRAIL) {
                        #pragma unroll
                        for (int i = 0; i < 64; i++) sha_round(y, i, p.kw[b * 64 + i]);
                    } else {
                        const uint32_t *kw = p.trailKW + b * 64;
                        #pragma unroll
                        for (int i = 0; i < 64; i++) sha_round(y, i, __ldg(kw + i));
                    }
                    #pragma unroll
                    for (int i = 0; i < 8; i++) s[i] += y[i];
                }

                if (__clz(s[0]) >= p.firstTarget) {
                    int bits = 0;
                    #pragma unroll
                    for (int i = 0; i < 8; i++) {
                        if (s[i] != 0) { bits += __clz(s[i]); break; }
                        bits += 32;
                    }
                    if (bits >= p.target) {
                        if (atomicCAS(&p.out->found, 0, 1) == 0) {
                            p.out->nonce = high * 10000ull + thousands * 1000u + hd * 100u + td * 10u + od;
                            #pragma unroll
                            for (int i = 0; i < 8; i++) p.out->hash[i] = s[i];
                            __threadfence();
                        }
                        return;
                    }
                }
            }
        }
    }
}


#define NGRIND_POW_ENTRY(B0, WV) \
    extern "C" __global__ void __launch_bounds__(POW_THREADS_PER_BLOCK) \
    ngrind_pow_##B0##_##WV(const PowParams p) { PowKernel<B0, WV>(p); }

#define NGRIND_POW_ROW(B0) \
    NGRIND_POW_ENTRY(B0, 0) NGRIND_POW_ENTRY(B0, 1) NGRIND_POW_ENTRY(B0, 2) NGRIND_POW_ENTRY(B0, 3) \
    NGRIND_POW_ENTRY(B0, 4) NGRIND_POW_ENTRY(B0, 5) NGRIND_POW_ENTRY(B0, 6) NGRIND_POW_ENTRY(B0, 7) \
    NGRIND_POW_ENTRY(B0, 8) NGRIND_POW_ENTRY(B0, 9) NGRIND_POW_ENTRY(B0, 10) NGRIND_POW_ENTRY(B0, 11) \
    NGRIND_POW_ENTRY(B0, 12) NGRIND_POW_ENTRY(B0, 13) NGRIND_POW_ENTRY(B0, 14) NGRIND_POW_ENTRY(B0, 15)

NGRIND_POW_ROW(0)
NGRIND_POW_ROW(1)
