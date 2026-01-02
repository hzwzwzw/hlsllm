#include "typedefs.h"
#include "config.h"
#include <math.h>
#include <cstring>
extern "C" void forward(Transformer<dim, hidden_dim, n_layers, n_heads, n_kv_heads, vocab_size, seq_len, GS> *transformer, int token, int pos, float key_cache[n_layers * seq_len * ((dim * n_kv_heads) / n_heads)], float value_cache[n_layers * seq_len * ((dim * n_kv_heads) / n_heads)], float *out);
template <int S>
void dequantize(QuantizedTensor<S> *qx, float x[S], int GS)
{
  for (int i = 0; i < S; i++)
  {
    x[i] = qx->q[i] * qx->s[i / GS];
  }
}

template <int S>
void quantize(QuantizedTensor<S> *qx, float x[S], int GS)
{
    constexpr int num_groups = S / 64;
    constexpr float Q_MAX = 127.0f;

    // 我们直接计算并写出到 AXI 接口，不留在本地做缓冲

    main_loop:
    for (int group = 0; group < num_groups; group++)
    {
        float wmax = 0.0;
        int base_idx = group * GS;

        // 第一步：找最大值（计算 Scale）
        max_loop:
        for (int i = 0; i < GS; i++) {
            #pragma HLS PIPELINE
            #pragma HLS UNROLL factor=16
            float val = fabs(x[base_idx + i]);
            if (val > wmax) wmax = val;
        }

        float inv_scale = Q_MAX / (wmax + 1e-9f); // 使用倒数，避开循环内的除法
        qx->s[group] = 1.0f / inv_scale; // 写回 scale

        // 第二步：直接量化并写回到接口 qx->q
        // 这样就不需要本地的 quantized_buffer 了
        quant_write_loop:
        for (int i = 0; i < GS; i++) {
            #pragma HLS PIPELINE
            #pragma HLS UNROLL factor=16
            float quant_value = x[base_idx + i] * inv_scale;
            int8_t q_val = (int8_t)round(quant_value);
            qx->q[base_idx + i] = q_val; // 直接写出
        }
    }
}