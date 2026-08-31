/* Portable scalar Q6_K x Q8_K reference/runtime bridge for the Qwen3.8 lab.
 * Semantics mirror llama.cpp pin 557614e0296ff4a5b6f649737a65ae2076eea2fd.
 */
#include <assert.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define QWEN_QK_K 256
#define QWEN_BLOCK_Q6_K 210
#define QWEN_BLOCK_Q8_K 292

static float qwen_f16_to_f32(uint16_t h) {
    const uint32_t sign = ((uint32_t)h & 0x8000u) << 16;
    uint32_t exp = ((uint32_t)h >> 10) & 0x1fu;
    uint32_t mant = (uint32_t)h & 0x03ffu;
    uint32_t bits;
    if (exp == 0) {
        if (mant == 0) {
            bits = sign;
        } else {
            int shift = 0;
            while ((mant & 0x0400u) == 0) {
                mant <<= 1;
                ++shift;
            }
            mant &= 0x03ffu;
            const uint32_t fexp = (uint32_t)(127 - 15 - shift);
            bits = sign | (fexp << 23) | (mant << 13);
        }
    } else if (exp == 0x1fu) {
        bits = sign | 0x7f800000u | (mant << 13);
    } else {
        const uint32_t fexp = exp + (127u - 15u);
        bits = sign | (fexp << 23) | (mant << 13);
    }
    float out;
    memcpy(&out, &bits, sizeof(out));
    return out;
}

static int qwen_host_little_endian(void) {
    const uint16_t one = 1;
    return *((const uint8_t *)&one) == 1;
}

static uint16_t qwen_load_u16_le(const uint8_t *p) {
    return (uint16_t)p[0] | ((uint16_t)p[1] << 8);
}

static float qwen_load_f32_le(const uint8_t *p) {
    uint32_t bits = (uint32_t)p[0] | ((uint32_t)p[1] << 8) | ((uint32_t)p[2] << 16) | ((uint32_t)p[3] << 24);
    float out;
    memcpy(&out, &bits, sizeof(out));
    return out;
}

static void qwen_store_f32_le(uint8_t *p, float x) {
    uint32_t bits;
    memcpy(&bits, &x, sizeof(bits));
    p[0] = (uint8_t)(bits & 0xffu);
    p[1] = (uint8_t)((bits >> 8) & 0xffu);
    p[2] = (uint8_t)((bits >> 16) & 0xffu);
    p[3] = (uint8_t)((bits >> 24) & 0xffu);
}

int qwen_quantize_q8_k_scalar(const float *x, size_t n, uint8_t *dst, size_t dst_bytes) {
    if (!x || !dst || n == 0 || n % QWEN_QK_K != 0) return -1;
    const size_t nb = n / QWEN_QK_K;
    if (dst_bytes != nb * QWEN_BLOCK_Q8_K) return -2;

    for (size_t ib = 0; ib < nb; ++ib) {
        const float *xb = x + ib * QWEN_QK_K;
        uint8_t *out = dst + ib * QWEN_BLOCK_Q8_K;
        float amax = 0.0f;
        float maxv = 0.0f;
        for (size_t j = 0; j < QWEN_QK_K; ++j) {
            const float ax = fabsf(xb[j]);
            if (ax > amax) {
                amax = ax;
                maxv = xb[j];
            }
        }
        if (amax == 0.0f) {
            memset(out, 0, QWEN_BLOCK_Q8_K);
            continue;
        }
        const float iscale = -127.0f / maxv;
        const float d = 1.0f / iscale;
        qwen_store_f32_le(out, d);
        int8_t *qs = (int8_t *)(out + 4);
        for (size_t j = 0; j < QWEN_QK_K; ++j) {
            long q = lrintf(iscale * xb[j]);
            if (q > 127) q = 127;
            if (q < -128) q = -128;
            qs[j] = (int8_t)q;
        }
        uint8_t *bs = out + 4 + QWEN_QK_K;
        for (size_t j = 0; j < QWEN_QK_K / 16; ++j) {
            int sum = 0;
            for (size_t ii = 0; ii < 16; ++ii) sum += qs[j * 16 + ii];
            const int16_t s = (int16_t)sum;
            bs[2*j + 0] = (uint8_t)((uint16_t)s & 0xffu);
            bs[2*j + 1] = (uint8_t)(((uint16_t)s >> 8) & 0xffu);
        }
    }
    return 0;
}

float qwen_vec_dot_q6_k_q8_k_scalar(const uint8_t *q6, size_t q6_bytes, const uint8_t *q8k, size_t q8k_bytes, size_t n) {
    if (!q6 || !q8k || n == 0 || n % QWEN_QK_K != 0) return NAN;
    const size_t nb = n / QWEN_QK_K;
    if (q6_bytes != nb * QWEN_BLOCK_Q6_K || q8k_bytes != nb * QWEN_BLOCK_Q8_K) return NAN;

    float sums[8] = {0};
    int8_t aux8[QWEN_QK_K];
    for (size_t ib = 0; ib < nb; ++ib) {
        const uint8_t *xb = q6 + ib * QWEN_BLOCK_Q6_K;
        const uint8_t *yb = q8k + ib * QWEN_BLOCK_Q8_K;
        const uint8_t *ql = xb;
        const uint8_t *qh = xb + 128;
        const int8_t *scales = (const int8_t *)(xb + 192);
        const int8_t *q8 = (const int8_t *)(yb + 4);

        int8_t *a = aux8;
        for (int j = 0; j < QWEN_QK_K; j += 128) {
            for (int l = 0; l < 32; ++l) {
                a[l + 0]  = (int8_t)(((ql[l + 0]  & 0x0f) | (((qh[l] >> 0) & 3) << 4)) - 32);
                a[l + 32] = (int8_t)(((ql[l + 32] & 0x0f) | (((qh[l] >> 2) & 3) << 4)) - 32);
                a[l + 64] = (int8_t)(((ql[l + 0]  >> 4)   | (((qh[l] >> 4) & 3) << 4)) - 32);
                a[l + 96] = (int8_t)(((ql[l + 32] >> 4)   | (((qh[l] >> 6) & 3) << 4)) - 32);
            }
            a += 128;
            ql += 64;
            qh += 32;
        }

        int32_t lanes[8] = {0};
        a = aux8;
        int is = 0;
        for (int j = 0; j < QWEN_QK_K/16; ++j) {
            const int scale = scales[is++];
            for (int l = 0; l < 8; ++l) lanes[l] += scale * (int)q8[l] * (int)a[l];
            q8 += 8; a += 8;
            for (int l = 0; l < 8; ++l) lanes[l] += scale * (int)q8[l] * (int)a[l];
            q8 += 8; a += 8;
        }
        const float d = qwen_f16_to_f32(qwen_load_u16_le(xb + 208)) * qwen_load_f32_le(yb);
        for (int l = 0; l < 8; ++l) sums[l] += d * (float)lanes[l];
    }
    float sumf = 0.0f;
    for (int l = 0; l < 8; ++l) sumf += sums[l];
    return sumf;
}

#ifdef QWEN_QUANT_DOT_SELFTEST
static void pack_q6_fixture(uint8_t out[QWEN_BLOCK_Q6_K], uint16_t f16_scale, const int8_t scales[16], const int8_t quants[256]) {
    memset(out, 0, QWEN_BLOCK_Q6_K);
    uint8_t *ql = out;
    uint8_t *qh = out + 128;
    for (int n = 0; n < 256; n += 128) {
        const int ql_base = n == 0 ? 0 : 64;
        const int qh_base = n == 0 ? 0 : 32;
        for (int l = 0; l < 32; ++l) {
            const uint8_t q1 = (uint8_t)(quants[n+l+0] + 32);
            const uint8_t q2 = (uint8_t)(quants[n+l+32] + 32);
            const uint8_t q3 = (uint8_t)(quants[n+l+64] + 32);
            const uint8_t q4 = (uint8_t)(quants[n+l+96] + 32);
            ql[ql_base+l] = (uint8_t)((q1 & 15) | ((q3 & 15) << 4));
            ql[ql_base+l+32] = (uint8_t)((q2 & 15) | ((q4 & 15) << 4));
            qh[qh_base+l] = (uint8_t)(((q1 >> 4) & 3) | (((q2 >> 4)&3)<<2) | (((q3>>4)&3)<<4) | (((q4>>4)&3)<<6));
        }
    }
    memcpy(out + 192, scales, 16);
    out[208] = (uint8_t)(f16_scale & 0xffu);
    out[209] = (uint8_t)(f16_scale >> 8);
}

static double independent_dot(const uint8_t *q6, const uint8_t *q8k) {
    const float q6d = qwen_f16_to_f32(qwen_load_u16_le(q6 + 208));
    const float q8d = qwen_load_f32_le(q8k);
    const int8_t *sc = (const int8_t *)(q6 + 192);
    const int8_t *q8 = (const int8_t *)(q8k + 4);
    double sum = 0.0;
    for (int idx = 0; idx < 256; ++idx) {
        const int half = idx >= 128;
        const int r = idx - half * 128;
        const int group = r / 32;
        const int l = r % 32;
        const uint8_t *ql = q6 + half * 64;
        const uint8_t *qh = q6 + 128 + half * 32;
        int q;
        if (group == 0) q = (ql[l] & 15) | (((qh[l] >> 0)&3)<<4);
        else if (group == 1) q = (ql[l+32] & 15) | (((qh[l] >> 2)&3)<<4);
        else if (group == 2) q = (ql[l] >> 4) | (((qh[l] >> 4)&3)<<4);
        else q = (ql[l+32] >> 4) | (((qh[l] >> 6)&3)<<4);
        q -= 32;
        sum += (double)q6d * (double)sc[idx/16] * (double)q * (double)q8d * (double)q8[idx];
    }
    return sum;
}

int main(void) {
    if (!qwen_host_little_endian()) {
        fprintf(stderr, "selftest currently requires little-endian host\n");
        return 2;
    }
    uint8_t q6[2 * QWEN_BLOCK_Q6_K];
    int8_t scales[16];
    int8_t quants[256];
    for (int i = 0; i < 16; ++i) scales[i] = (int8_t)((i % 2 ? 1 : -1) * (i + 1));
    for (int i = 0; i < 256; ++i) quants[i] = (int8_t)(((i * 17 + 3) % 64) - 32);
    pack_q6_fixture(q6, 0x3400u /* 0.25 */, scales, quants);
    for (int i = 0; i < 16; ++i) scales[i] = (int8_t)(8 - i);
    for (int i = 0; i < 256; ++i) quants[i] = (int8_t)(31 - ((i * 29 + 11) % 64));
    pack_q6_fixture(q6 + QWEN_BLOCK_Q6_K, 0xb000u /* -0.125 */, scales, quants);

    float x[512];
    for (int i = 0; i < 256; ++i) x[i] = (float)((i % 255) - 127) / 127.0f;
    for (int i = 0; i < 256; ++i) x[256+i] = 0.75f * (float)(((i * 7) % 251) - 125) / 125.0f;
    uint8_t q8k[2 * QWEN_BLOCK_Q8_K];
    if (qwen_quantize_q8_k_scalar(x, 512, q8k, sizeof(q8k)) != 0) return 3;

    const double expected = independent_dot(q6, q8k) + independent_dot(q6 + QWEN_BLOCK_Q6_K, q8k + QWEN_BLOCK_Q8_K);
    const float got = qwen_vec_dot_q6_k_q8_k_scalar(q6, sizeof(q6), q8k, sizeof(q8k), 512);
    const double err = fabs((double)got - expected);
    const double limit = 2e-5 * fmax(1.0, fabs(expected));
    if (!(err <= limit)) {
        fprintf(stderr, "dot mismatch got=%.9g expected=%.17g err=%.9g limit=%.9g\n", got, expected, err, limit);
        return 4;
    }

    float zero[256] = {0};
    uint8_t zq8[QWEN_BLOCK_Q8_K];
    if (qwen_quantize_q8_k_scalar(zero, 256, zq8, sizeof(zq8)) != 0) return 5;
    for (size_t i = 0; i < sizeof(zq8); ++i) if (zq8[i] != 0) return 6;
    const float zdot = qwen_vec_dot_q6_k_q8_k_scalar(q6, QWEN_BLOCK_Q6_K, zq8, sizeof(zq8), 256);
    if (zdot != 0.0f) return 7;

    if (!isnan(qwen_vec_dot_q6_k_q8_k_scalar(q6, QWEN_BLOCK_Q6_K - 1, zq8, sizeof(zq8), 256))) return 8;

    printf("{\n");
    printf("  \"schema\": \"qwen38-q6k-q8k-scalar-sanity-v1\",\n");
    printf("  \"status\": \"PASS\",\n");
    printf("  \"llama_cpp_reference_revision\": \"557614e0296ff4a5b6f649737a65ae2076eea2fd\",\n");
    printf("  \"model_weights_downloaded\": false,\n");
    printf("  \"q6_k_block_bytes\": %d,\n", QWEN_BLOCK_Q6_K);
    printf("  \"q8_k_block_bytes\": %d,\n", QWEN_BLOCK_Q8_K);
    printf("  \"tested_elements\": 512,\n");
    printf("  \"dot_abs_error\": %.9g,\n", err);
    printf("  \"dot_error_limit\": %.9g,\n", limit);
    printf("  \"zero_activation_dot\": %.9g,\n", zdot);
    printf("  \"invalid_size_rejected\": true\n");
    printf("}\n");
    return 0;
}
#endif
