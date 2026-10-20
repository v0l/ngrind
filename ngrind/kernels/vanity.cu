#include <stdint.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <curand_kernel.h>

#ifndef NOSTR_BLOCKS_PER_GRID
#define NOSTR_BLOCKS_PER_GRID 3072
#endif
#ifndef NOSTR_THREADS_PER_BLOCK
#define NOSTR_THREADS_PER_BLOCK 256
#endif
#ifndef KEYS_PER_THREAD_BATCH
#define KEYS_PER_THREAD_BATCH 256
#endif

#define NOSTR_COUNT_CUDA_THREADS (NOSTR_BLOCKS_PER_GRID * NOSTR_THREADS_PER_BLOCK)
#define NOSTR_IDX_CUDA_THREAD ((blockIdx.x * blockDim.x) + threadIdx.x)

#define MAX_VANITY_HEX_LEN 16
#define MAX_VANITY_BECH32_LEN 52

#define SIZE_PRIV_KEY_NOSTR 32
#define SIZE_PUBKEY_NOSTR 32
#define SIZE_LONG 8

#define NUM_GTABLE_CHUNK 16
#define NUM_GTABLE_VALUE 65536
#define SIZE_GTABLE_POINT 32
#define COUNT_GTABLE_POINTS (NUM_GTABLE_CHUNK * NUM_GTABLE_VALUE)

__constant__ int NOSTR_CHUNK_FIRST_ELEMENT[NUM_GTABLE_CHUNK] = {
  65536*0,  65536*1,  65536*2,  65536*3,
  65536*4,  65536*5,  65536*6,  65536*7,
  65536*8,  65536*9,  65536*10, 65536*11,
  65536*12, 65536*13, 65536*14, 65536*15,
};

#include "GPUMath.h"

static_assert(sizeof(curandState) == 48, "the host allocates 48 bytes of generator state per thread");

// Bech32 charset (32 characters + null terminator)
__constant__ char BECH32_CHARSET[33] = "qpzry9x8gf2tvdw0s3jn54khce6mua7l";

// Bech32 HRP expansion for "npub"
__constant__ uint32_t BECH32_HRP_EXPAND[5] = {3, 3, 3, 3, 16}; // npub expansion

// 256-bit starting offset for sequential search mode
__constant__ uint8_t d_startOffset[32];
__constant__ uint8_t d_vanitySplit;

// Convert hex character to integer
__device__ uint8_t hexCharToInt(uint8_t c) {
    if (c >= '0' && c <= '9') return c - '0';
    if (c >= 'a' && c <= 'f') return c - 'a' + 10;
    if (c >= 'A' && c <= 'F') return c - 'A' + 10;
    return 0;
}

// Convert byte to hex characters
__device__ void byteToHex(uint8_t byte, char *hex) {
    const char hexChars[] = "0123456789abcdef";
    hex[0] = hexChars[(byte >> 4) & 0xF];
    hex[1] = hexChars[byte & 0xF];
}

// Bech32 polymod function for checksum
__device__ uint32_t bech32_polymod(uint8_t *values, int len) {
    uint32_t chk = 1;
    uint32_t GEN[5] = {0x3b6a57b2, 0x26508e6d, 0x1ea119fa, 0x3d4233dd, 0x2a1462b3};

    for (int i = 0; i < len; i++) {
        uint8_t top = chk >> 25;
        chk = (chk & 0x1ffffff) << 5 ^ values[i];
        for (int j = 0; j < 5; j++) {
            if ((top >> j) & 1) {
                chk ^= GEN[j];
            }
        }
    }
    return chk;
}

// Convert 8-bit array to 5-bit array for bech32
__device__ void convert_bits(uint8_t *out, int *outlen, uint8_t *in, int inlen, int frombits, int tobits, bool pad) {
    uint32_t acc = 0;
    int bits = 0;
    int maxv = (1 << tobits) - 1;
    int max_acc = (1 << (frombits + tobits - 1)) - 1;
    *outlen = 0;

    for (int i = 0; i < inlen; i++) {
        acc = ((acc << frombits) | in[i]) & max_acc;
        bits += frombits;
        while (bits >= tobits) {
            bits -= tobits;
            out[(*outlen)++] = (acc >> bits) & maxv;
        }
    }

    if (pad) {
        if (bits > 0) {
            out[(*outlen)++] = (acc << (tobits - bits)) & maxv;
        }
    }
}

// Encode pubkey to bech32 npub format
// Returns the npub string without "npub1" prefix (just the encoded part)
__device__ void encode_npub(uint8_t *pubkey_32bytes, char *npub_out) {
    // Convert pubkey to 5-bit groups
    uint8_t data5[52]; // 256 bits / 5 bits per group = 51.2, rounded up = 52
    int data5_len;
    convert_bits(data5, &data5_len, pubkey_32bytes, 32, 8, 5, true);

    // Create values array for checksum: HRP expansion + data + 6 zeros
    uint8_t values[63]; // 5 (HRP) + 52 (data) + 6 (checksum placeholder)

    // Add HRP expansion for "npub"
    values[0] = 3;  // 'n' >> 5
    values[1] = 3;  // 'p' >> 5
    values[2] = 3;  // 'u' >> 5
    values[3] = 3;  // 'b' >> 5
    values[4] = 16; // separator

    // Add data
    for (int i = 0; i < data5_len; i++) {
        values[5 + i] = data5[i];
    }

    // Add 6 zeros for checksum calculation
    for (int i = 0; i < 6; i++) {
        values[5 + data5_len + i] = 0;
    }

    // Calculate checksum
    uint32_t polymod = bech32_polymod(values, 5 + data5_len + 6) ^ 1;

    // Extract checksum (6 characters)
    uint8_t checksum[6];
    for (int i = 0; i < 6; i++) {
        checksum[i] = (polymod >> (5 * (5 - i))) & 31;
    }

    // Encode data to bech32 charset
    for (int i = 0; i < data5_len; i++) {
        npub_out[i] = BECH32_CHARSET[data5[i]];
    }

    // Append checksum
    for (int i = 0; i < 6; i++) {
        npub_out[data5_len + i] = BECH32_CHARSET[checksum[i]];
    }

    // Null terminate
    npub_out[data5_len + 6] = '\0';
}

// Check if bech32 string matches pattern
__device__ bool matchesBech32Pattern(char *npub, uint8_t *pattern, uint8_t patternLen, bool isPrefix) {
    if (isPrefix) {
        // Check prefix match (skip "npub1" part, match against encoded data)
        for (uint8_t i = 0; i < patternLen; i++) {
            if (npub[i] != pattern[i]) return false;
        }
    } else {
        // Check suffix match (before checksum - last 6 chars are checksum)
        int data_len = 52;  // Length without checksum
        int start_pos = data_len - patternLen;

        for (uint8_t i = 0; i < patternLen; i++) {
            if (npub[start_pos + i] != pattern[i]) return false;
        }
    }
    return true;
}

// Check if public key matches hex vanity pattern
__device__ bool matchesHexPattern(uint64_t *pubkey, uint8_t *pattern, uint8_t patternLen, bool isPrefix) {
    uint8_t *pubkeyBytes = (uint8_t *)pubkey;
    char hex[2];

    // NOTE: pubkey is stored in little-endian format, but we display/search in big-endian
    // So for prefix search, we need to check from the END of the byte array (reversed)
    // And for suffix search, we check from the START (reversed)

    if (isPrefix) {
        // Check prefix match - read from END of array backwards (big-endian display)
        for (uint8_t i = 0; i < patternLen; i++) {
            int byteIdx = 31 - (i / 2);  // Start from byte 31 and work backwards
            byteToHex(pubkeyBytes[byteIdx], hex);
            if (i % 2 == 0) {
                if (hex[0] != pattern[i]) return false;
            } else {
                if (hex[1] != pattern[i]) return false;
            }
        }
    } else {
        // Check suffix match - read from START of array (big-endian display)
        int startByte = (patternLen + 1) / 2 - 1;  // Start from beginning
        int startChar = (patternLen % 2 == 1) ? 1 : 0;

        for (uint8_t i = 0; i < patternLen; i++) {
            int byteIdx = startByte - (i + startChar) / 2;  // Work from start
            byteToHex(pubkeyBytes[byteIdx], hex);
            if ((i + startChar) % 2 == 0) {
                if (hex[0] != pattern[i]) return false;
            } else {
                if (hex[1] != pattern[i]) return false;
            }
        }
    }

    return true;
}

//Cuda Secp256k1 Point Multiplication (from GPUSecp.cu)
//Takes 32-byte privKey + gTable and outputs 64-byte public key [qx,qy]
__device__ void _PointMultiSecp256k1Nostr(uint64_t *qx, uint64_t *qy, uint16_t *privKey, uint8_t *gTableX, uint8_t *gTableY) {
    int chunk = 0;
    uint64_t qz[5] = {1, 0, 0, 0, 0};

    //Find the first non-zero point [qx,qy]
    for (; chunk < NUM_GTABLE_CHUNK; chunk++) {
        if (privKey[chunk] > 0) {
            int index = (NOSTR_CHUNK_FIRST_ELEMENT[chunk] + (privKey[chunk] - 1)) * SIZE_GTABLE_POINT;
            memcpy(qx, gTableX + index, SIZE_GTABLE_POINT);
            memcpy(qy, gTableY + index, SIZE_GTABLE_POINT);
            chunk++;
            break;
        }
    }

    //Add the remaining chunks together
    for (; chunk < NUM_GTABLE_CHUNK; chunk++) {
        if (privKey[chunk] > 0) {
            uint64_t gx[4];
            uint64_t gy[4];

            int index = (NOSTR_CHUNK_FIRST_ELEMENT[chunk] + (privKey[chunk] - 1)) * SIZE_GTABLE_POINT;

            memcpy(gx, gTableX + index, SIZE_GTABLE_POINT);
            memcpy(gy, gTableY + index, SIZE_GTABLE_POINT);

            _PointAddSecp256k1(qx, qy, qz, gx, gy);
        }
    }

    //Performing modular inverse on qz to obtain the public key [qx,qy]
    _ModInv(qz);
    _ModMult(qx, qz);
    _ModMult(qy, qz);
}

//GPU kernel to initialize cuRAND states (called once at startup)
extern "C" __global__ void ngrind_rand_init(curandState *randStates, uint64_t seed) {
    int idxThread = NOSTR_IDX_CUDA_THREAD;

    // Initialize this thread's RNG state with unique seed
    curand_init(seed + idxThread, 0, 0, &randStates[idxThread]);
}

//GPU kernel function for Nostr vanity key mining (BATCHED)
//GPU kernel for Sequential vanity key mining (exhaustive search)
extern "C" __global__ void ngrind_vanity_sequential(
    uint64_t globalIteration,  // Current global iteration
    uint8_t *gTableXGPU,
    uint8_t *gTableYGPU,
    uint8_t *vanityPatternGPU,
    uint8_t vanityLen,
    int vanityMode,
    uint8_t *outputFoundGPU,
    uint8_t *outputPrivKeysGPU,
    uint8_t *outputPubKeysGPU)
{
    int idxThread = NOSTR_IDX_CUDA_THREAD;

    // Calculate sequential key index for this thread in this batch
    // Each thread handles KEYS_PER_THREAD_BATCH keys per global iteration
    uint64_t baseKeyIndex = globalIteration * NOSTR_COUNT_CUDA_THREADS * KEYS_PER_THREAD_BATCH;
    uint64_t threadBaseIndex = baseKeyIndex + (idxThread * KEYS_PER_THREAD_BATCH);

    // Generate and check MULTIPLE keys sequentially
    for (int batch = 0; batch < KEYS_PER_THREAD_BATCH; batch++) {
        uint64_t keyIndex = threadBaseIndex + batch;

        // Start with the base offset from constant memory
        uint8_t privKey[SIZE_PRIV_KEY_NOSTR];
        for (int i = 0; i < SIZE_PRIV_KEY_NOSTR; i++) {
            privKey[i] = d_startOffset[i];
        }

        // Add keyIndex to the offset (256-bit addition with carry)
        uint64_t carry = keyIndex;
        for (int i = SIZE_PRIV_KEY_NOSTR - 1; i >= 0 && carry > 0; i--) {
            uint64_t sum = privKey[i] + (carry & 0xFF);
            privKey[i] = sum & 0xFF;
            carry = (carry >> 8) + (sum >> 8);
        }

        // Ensure private key is valid (not zero)
        bool isZero = true;
        for (int i = 0; i < SIZE_PRIV_KEY_NOSTR; i++) {
            if (privKey[i] != 0) {
                isZero = false;
                break;
            }
        }
        if (isZero) continue;  // Skip zero key

        // Compute secp256k1 public key
        uint64_t qx[4];
        uint64_t qy[4];
        _PointMultiSecp256k1Nostr(qx, qy, (uint16_t *)privKey, gTableXGPU, gTableYGPU);

        // Check if x-coordinate matches vanity pattern (same logic as random mode)
        bool matched = false;

        if (vanityMode == 0) { // VANITY_HEX_PREFIX
            matched = matchesHexPattern(qx, vanityPatternGPU, vanityLen, true);
        } else if (vanityMode == 1) { // VANITY_HEX_SUFFIX
            matched = matchesHexPattern(qx, vanityPatternGPU, vanityLen, false);
        } else if (vanityMode == 2) { // VANITY_HEX_BOTH
            uint8_t halfLen = d_vanitySplit;
            matched = matchesHexPattern(qx, vanityPatternGPU, halfLen, true) &&
                      matchesHexPattern(qx, vanityPatternGPU + halfLen, vanityLen - halfLen, false);
        } else if (vanityMode == 3 || vanityMode == 4 || vanityMode == 5) {
            // VANITY_BECH32_PREFIX, VANITY_BECH32_SUFFIX, VANITY_BECH32_BOTH
            // Convert qx from little-endian to big-endian for Nostr
            uint8_t pubkey_be[32];
            uint8_t *qxBytes = (uint8_t *)qx;
            for (int i = 0; i < 32; i++) {
                pubkey_be[i] = qxBytes[31 - i];
            }
            char npub[64];
            encode_npub(pubkey_be, npub);

            if (vanityMode == 3) {
                matched = matchesBech32Pattern(npub, vanityPatternGPU, vanityLen, true);
            } else if (vanityMode == 4) {
                matched = matchesBech32Pattern(npub, vanityPatternGPU, vanityLen, false);
            } else if (vanityMode == 5) {
                uint8_t halfLen = d_vanitySplit;
                matched = matchesBech32Pattern(npub, vanityPatternGPU, halfLen, true) &&
                          matchesBech32Pattern(npub, vanityPatternGPU + halfLen, vanityLen - halfLen, false);
            }
        }

        // If matched and we haven't already found one, store it
        if (matched && outputFoundGPU[idxThread] == 0) {
            outputFoundGPU[idxThread] = 1;

            // Copy private key to output
            for (int i = 0; i < SIZE_PRIV_KEY_NOSTR; i++) {
                outputPrivKeysGPU[(idxThread * SIZE_PRIV_KEY_NOSTR) + i] = privKey[i];
            }

            // Copy public key (x-only) to output
            // NOTE: qx is stored as uint64_t[4] in little-endian format, but Nostr
            // expects big-endian byte order, so we reverse the bytes
            uint8_t *qxBytes = (uint8_t *)qx;
            for (int i = 0; i < SIZE_PUBKEY_NOSTR; i++) {
                outputPubKeysGPU[(idxThread * SIZE_PUBKEY_NOSTR) + i] = qxBytes[31 - i];
            }

            break;  // Found a match, stop checking this batch
        }
    }
}

//GPU kernel for Random vanity key mining (original random search)
extern "C" __global__ void ngrind_vanity_random(
    curandState *randStates,
    uint8_t *gTableXGPU,
    uint8_t *gTableYGPU,
    uint8_t *vanityPatternGPU,
    uint8_t vanityLen,
    int vanityMode,
    uint8_t *outputFoundGPU,
    uint8_t *outputPrivKeysGPU,
    uint8_t *outputPubKeysGPU)
{
    int idxThread = NOSTR_IDX_CUDA_THREAD;

    // Use pre-initialized RNG state
    curandState localState = randStates[idxThread];

    // Generate and check MULTIPLE keys per thread (BATCHING!)
    for (int batch = 0; batch < KEYS_PER_THREAD_BATCH; batch++) {
        // Generate random 32-byte private key
        uint8_t privKey[SIZE_PRIV_KEY_NOSTR];
        uint32_t *privKey32 = (uint32_t *)privKey;

        for (int i = 0; i < 8; i++) {
            privKey32[i] = curand(&localState);
        }

        // Ensure private key is valid (not zero)
        bool isZero = true;
        for (int i = 0; i < SIZE_PRIV_KEY_NOSTR; i++) {
            if (privKey[i] != 0) {
                isZero = false;
                break;
            }
        }

        if (isZero) {
            privKey[31] = 1;
        }

        // Compute secp256k1 public key
        uint64_t qx[4];
        uint64_t qy[4];

        _PointMultiSecp256k1Nostr(qx, qy, (uint16_t *)privKey, gTableXGPU, gTableYGPU);

        // Check if x-coordinate matches vanity pattern
        bool matched = false;

        if (vanityMode == 0) { // VANITY_HEX_PREFIX
            matched = matchesHexPattern(qx, vanityPatternGPU, vanityLen, true);
        } else if (vanityMode == 1) { // VANITY_HEX_SUFFIX
            matched = matchesHexPattern(qx, vanityPatternGPU, vanityLen, false);
        } else if (vanityMode == 2) { // VANITY_HEX_BOTH
            uint8_t halfLen = d_vanitySplit;
            matched = matchesHexPattern(qx, vanityPatternGPU, halfLen, true) &&
                      matchesHexPattern(qx, vanityPatternGPU + halfLen, vanityLen - halfLen, false);
        } else if (vanityMode == 3 || vanityMode == 4 || vanityMode == 5) {
            // VANITY_BECH32_PREFIX, VANITY_BECH32_SUFFIX, VANITY_BECH32_BOTH

            // Encode public key to bech32
            // Convert qx from little-endian to big-endian for Nostr
            uint8_t pubkey_be[32];
            uint8_t *qxBytes = (uint8_t *)qx;
            for (int i = 0; i < 32; i++) {
                pubkey_be[i] = qxBytes[31 - i];
            }
            char npub[64]; // 52 data + 6 checksum + null terminator
            encode_npub(pubkey_be, npub);

            // Check pattern match
            if (vanityMode == 3) { // VANITY_BECH32_PREFIX
                matched = matchesBech32Pattern(npub, vanityPatternGPU, vanityLen, true);
            } else if (vanityMode == 4) { // VANITY_BECH32_SUFFIX
                matched = matchesBech32Pattern(npub, vanityPatternGPU, vanityLen, false);
            } else if (vanityMode == 5) { // VANITY_BECH32_BOTH
                uint8_t halfLen = d_vanitySplit;
                matched = matchesBech32Pattern(npub, vanityPatternGPU, halfLen, true) &&
                          matchesBech32Pattern(npub, vanityPatternGPU + halfLen, vanityLen - halfLen, false);
            }
        }

        // If matched and we haven't already found one, store it
        if (matched && outputFoundGPU[idxThread] == 0) {
            // Mark that we found a match
            outputFoundGPU[idxThread] = 1;

            // Copy private key to output
            for (int i = 0; i < SIZE_PRIV_KEY_NOSTR; i++) {
                outputPrivKeysGPU[(idxThread * SIZE_PRIV_KEY_NOSTR) + i] = privKey[i];
            }

            // Copy public key (x-only) to output
            // NOTE: qx is stored as uint64_t[4] in little-endian format, but Nostr
            // expects big-endian byte order, so we reverse the bytes
            uint8_t *qxBytes = (uint8_t *)qx;
            for (int i = 0; i < SIZE_PUBKEY_NOSTR; i++) {
                outputPubKeysGPU[(idxThread * SIZE_PUBKEY_NOSTR) + i] = qxBytes[31 - i];
            }

            // Break after first match to save time
            break;
        }
    } // End of batch loop

    // Save updated RNG state back to global memory
    randStates[idxThread] = localState;
}

