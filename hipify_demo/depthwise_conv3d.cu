/*
 * Hipify demo: portable depthwise Conv3D BF16 reference kernel with CUDA APIs.
 * Run ./run_hipify.sh then ./build_kernel.sh to produce a HIP binary.
 *
 * Args: [niter] [nwarmup] [--no-check]
 */

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#include <cuda_bf16.h>
#include <cuda_runtime.h>

#ifndef KD
#define KD 3
#endif
#ifndef KH
#define KH 5
#endif
#ifndef KW
#define KW 5
#endif
#ifndef PaddingD
#define PaddingD 0
#endif
#ifndef PaddingH
#define PaddingH 2
#endif
#ifndef PaddingW
#define PaddingW 2
#endif
#ifndef BLOCK_H
#define BLOCK_H 45
#endif
#ifndef BLOCK_W
#define BLOCK_W 80
#endif

#define CUDA_KERNEL_LOOP(i, n)                                                \
  for (int64_t i = (int64_t)(blockIdx.x) * blockDim.x + threadIdx.x; i < (n); \
       i += (int64_t)(blockDim.x) * gridDim.x)

#define CUDA_CHECK(call)                                                                  \
  do {                                                                                    \
    cudaError_t err = (call);                                                             \
    if (err != cudaSuccess) {                                                             \
      fprintf(stderr, "%s:%d CUDA error: %s\n", __FILE__, __LINE__, cudaGetErrorString(err)); \
      std::exit(1);                                                                       \
    }                                                                                     \
  } while (0)

// BFloat16 reference kernel (grid-stride 1D launch).
__global__ void conv_depthwise3d_cuda_kernel_reference_bf16(
    const void* input_void,
    void* output_void,
    const void* kernel_void,
    const void* bias_void,
    int batch,
    int iC,
    int oC,
    int iT,
    int iH,
    int iW,
    int oT,
    int oH,
    int oW,
    int kT,
    int kH,
    int kW,
    int strideT,
    int strideH,
    int strideW,
    int paddingT,
    int paddingH,
    int paddingW,
    int dilationT,
    int dilationH,
    int dilationW)
{
  const __nv_bfloat16* input = static_cast<const __nv_bfloat16*>(input_void);
  __nv_bfloat16* output = static_cast<__nv_bfloat16*>(output_void);
  const __nv_bfloat16* kernel = static_cast<const __nv_bfloat16*>(kernel_void);
  const __nv_bfloat16* bias = static_cast<const __nv_bfloat16*>(bias_void);

  const int channel_multiplier = oC / iC;
  const int num_output = batch * oC * oT * oH * oW;

  CUDA_KERNEL_LOOP(index, num_output) {
    const int out_col = index % oW;
    const int out_row = (index / oW) % oH;
    const int out_frame = (index / oW / oH) % oT;
    const int out_channel = (index / oW / oH / oT) % oC;
    const int b = index / oW / oH / oT / oC;

    const int in_channel = out_channel / channel_multiplier;

    const int in_col_start = out_col * strideW - paddingW;
    const int in_row_start = out_row * strideH - paddingH;
    const int in_frame_start = out_frame * strideT - paddingT;

    float sum = 0.0f;
    const __nv_bfloat16* kernel_ptr = kernel + out_channel * kT * kH * kW;
    const int input_stride_c = iT * iH * iW;
    const int input_stride_t = iH * iW;
    const __nv_bfloat16* input_ptr = input + b * iC * input_stride_c + in_channel * input_stride_c +
                                     in_frame_start * input_stride_t + in_row_start * iW + in_col_start;

    for (int k_frame = 0; k_frame < kT; ++k_frame) {
      const int in_frame = in_frame_start + k_frame * dilationT;
      for (int k_row = 0; k_row < kH; ++k_row) {
        const int in_row = in_row_start + k_row * dilationH;
        for (int k_col = 0; k_col < kW; ++k_col) {
          const float op1 = (float)*(kernel_ptr++);
          const int in_col = in_col_start + k_col * dilationW;
          if (in_frame >= 0 && in_row >= 0 && in_col >= 0 && in_frame < iT && in_row < iH && in_col < iW) {
            sum += op1 * (float)*input_ptr;
          }
          input_ptr += dilationW;
        }
        input_ptr += iW * dilationH - kW * dilationW;
      }
      input_ptr += iW * (iH * dilationT - kH * dilationH);
    }
    if (bias != nullptr) {
      sum += (float)bias[out_channel];
    }

    const int output_stride_c = oT * oH * oW;
    const int output_stride_t = oH * oW;
    output[b * oC * output_stride_c + out_channel * output_stride_c + out_frame * output_stride_t +
           out_row * oW + out_col] = (__nv_bfloat16)sum;
  }
}

static void bf16_fill_pattern(__nv_bfloat16* dst, size_t n, unsigned seed) {
  for (size_t i = 0; i < n; ++i) {
    uint16_t u = static_cast<uint16_t>((seed * 1103515245u + static_cast<unsigned>(i)) & 0x7fffu);
    u |= 0x3f00u;
    std::memcpy(dst + i, &u, sizeof(u));
  }
}

static float bf16_to_float(__nv_bfloat16 x) {
  uint16_t bits = 0;
  std::memcpy(&bits, &x, sizeof(bits));
  const uint32_t u32 = static_cast<uint32_t>(bits) << 16u;
  float f = 0.f;
  std::memcpy(&f, &u32, sizeof(f));
  return f;
}

static size_t lin5(int B, int C, int D, int H, int W, int b, int c, int d, int h, int w) {
  return (((static_cast<size_t>(b) * static_cast<size_t>(C) + static_cast<size_t>(c)) * static_cast<size_t>(D) +
           static_cast<size_t>(d)) *
              static_cast<size_t>(H) +
          static_cast<size_t>(h)) *
             static_cast<size_t>(W) +
         static_cast<size_t>(w);
}

static size_t lin_w(int oC, int kd, int kh, int kw) {
  const size_t per_oc = static_cast<size_t>(KD) * KH * KW;
  return static_cast<size_t>(oC) * per_oc + static_cast<size_t>(kd) * static_cast<size_t>(KH * KW) +
         static_cast<size_t>(kh) * static_cast<size_t>(KW) + static_cast<size_t>(kw);
}

static void cpu_depthwise_conv3d_ref(const std::vector<__nv_bfloat16>& in_bf16,
                                     const std::vector<__nv_bfloat16>& w_bf16,
                                     const std::vector<__nv_bfloat16>& bias_bf16, std::vector<float>& out_f32, int B,
                                     int C, int D, int H, int W, int C_out, int D_out, int H_out, int W_out) {
  out_f32.assign(static_cast<size_t>(B) * C_out * D_out * H_out * W_out, 0.f);
  const int stride_d = 1, stride_h = 1, stride_w = 1;
  const int dil_d = 1, dil_h = 1, dil_w = 1;

  for (int b = 0; b < B; ++b) {
    for (int oc = 0; oc < C_out; ++oc) {
      const int ic = oc;
      const float bias = bf16_to_float(bias_bf16[static_cast<size_t>(oc)]);
      for (int od = 0; od < D_out; ++od) {
        for (int oh = 0; oh < H_out; ++oh) {
          for (int ow = 0; ow < W_out; ++ow) {
            float acc = bias;
            for (int kd = 0; kd < KD; ++kd) {
              for (int kh = 0; kh < KH; ++kh) {
                for (int kw = 0; kw < KW; ++kw) {
                  const int id = od * stride_d - PaddingD + kd * dil_d;
                  const int ih = oh * stride_h - PaddingH + kh * dil_h;
                  const int iw = ow * stride_w - PaddingW + kw * dil_w;
                  if (id < 0 || id >= D || ih < 0 || ih >= H || iw < 0 || iw >= W) {
                    continue;
                  }
                  const float iv = bf16_to_float(in_bf16[lin5(B, C, D, H, W, b, ic, id, ih, iw)]);
                  const float wv = bf16_to_float(w_bf16[lin_w(oc, kd, kh, kw)]);
                  acc += iv * wv;
                }
              }
            }
            out_f32[lin5(B, C_out, D_out, H_out, W_out, b, oc, od, oh, ow)] = acc;
          }
        }
      }
    }
  }
}

static void print_bytes_line(const char* label, size_t bytes) {
  const double mib = static_cast<double>(bytes) / (1024.0 * 1024.0);
  printf("  %-14s %10zu B  (%8.2f MiB)\n", label, bytes, mib);
}

static void print_run_header(int device_id) {
  cudaDeviceProp prop{};
  CUDA_CHECK(cudaGetDeviceProperties(&prop, device_id));
  printf("\n");
  printf("=== Depthwise Conv3D BF16 (portable reference kernel; hipify CUDA->HIP demo) ===\n");
  printf("Device id %d: %s\n", device_id, prop.name);
  printf("  multiProcessorCount=%d  maxThreadsPerBlock=%d  warpSize=%d\n", prop.multiProcessorCount,
         prop.maxThreadsPerBlock, prop.warpSize);
}

static void print_problem_and_memory(int B, int C, int D, int H, int W, int C_out, int D_out, int H_out, int W_out,
                                     size_t n_in, size_t n_out, size_t n_w, size_t n_bias) {
  const size_t bsz = sizeof(__nv_bfloat16);
  const size_t bytes_in = n_in * bsz;
  const size_t bytes_out = n_out * bsz;
  const size_t bytes_w = n_w * bsz;
  const size_t bytes_bias = n_bias * bsz;
  const size_t bytes_dev = bytes_in + bytes_out + bytes_w + bytes_bias;

  printf("\n--- Problem (case3-style depthwise) ---\n");
  printf("  Input NCHW:   [%d, %d, %d, %d, %d]  BF16\n", B, C, D, H, W);
  printf("  Weight:       [%d, 1, %d, %d, %d]  BF16  (groups=%d depthwise)\n", C_out, KD, KH, KW, C);
  printf("  Output NCHW:  [%d, %d, %d, %d, %d]  BF16\n", B, C_out, D_out, H_out, W_out);
  printf("  stride=(1,1,1)  dilation=(1,1,1)  padding=(%d,%d,%d)\n", PaddingD, PaddingH, PaddingW);

  printf("\n--- Device buffers ---\n");
  print_bytes_line("input", bytes_in);
  print_bytes_line("output", bytes_out);
  print_bytes_line("weights", bytes_w);
  print_bytes_line("bias", bytes_bias);
  print_bytes_line("total alloc", bytes_dev);
}

static void print_kernel_config(int blocks, dim3 block, int64_t num_output) {
  const long long total_threads = static_cast<long long>(blocks) * block.x;

  printf("\n--- Kernel configuration ---\n");
  printf("  Kernel: conv_depthwise3d_cuda_kernel_reference_bf16 (portable reference)\n");
  printf("  Compile-time: KD=%d KH=%d KW=%d  PaddingD/H/W=%d/%d/%d  BLOCK_H=%d BLOCK_W=%d  BF16\n", KD, KH, KW,
         PaddingD, PaddingH, PaddingW, BLOCK_H, BLOCK_W);
  printf("  Block:  (%d, %d, %d) -> %d threads / block\n", block.x, block.y, block.z,
         block.x * block.y * block.z);
  printf("  Grid:   (%d, 1, 1) -> %d blocks  (num_output=%lld, ~%lld thread slots / launch)\n", blocks, blocks,
         (long long)num_output, (long long)total_threads);
}

static void print_performance(double avg_ms, int niter, int B, int C_out, int D_out, int H_out, int W_out, size_t n_in,
                              size_t n_out, size_t n_w, size_t n_bias) {
  const double sec = static_cast<double>(avg_ms) / 1000.0;
  const double gflops =
      (2.0 * static_cast<double>(B) * C_out * D_out * H_out * W_out * 1.0 * KD * KH * KW) / 1e9;
  const double tflops = (gflops / 1000.0) / sec;
  const size_t bsz = sizeof(__nv_bfloat16);
  const double bytes_hbm = static_cast<double>((n_in + n_w + n_bias + n_out) * bsz);
  const double gbps = bytes_hbm / sec / 1e9;

  printf("\n--- Performance (same style as gemm samples) ---\n");
  printf("  niter (timed):        %d\n", niter);
  printf("  Average kernel time:  %.4f ms  (total %.4f ms)\n", avg_ms, avg_ms * niter);
  printf("  Workload:             %.4f GFLOP / iter\n", gflops);
  printf("  Throughput:           %.4f TFLOPS\n", tflops);
  printf("  Efficive Bandwidth:   %.1f GB/s \n", gbps);
}

static bool check_output_bf16_vs_ref(const std::vector<__nv_bfloat16>& gpu_out, const std::vector<float>& ref_f32,
                                     float atol, float rtol, int B, int C_out, int D_out, int H_out, int W_out) {
  size_t bad = 0;
  float max_abs = 0.f;
  size_t first_i = 0;
  for (size_t i = 0; i < ref_f32.size(); ++i) {
    const float g = bf16_to_float(gpu_out[i]);
    const float r = ref_f32[i];
    const float diff = std::fabs(g - r);
    const float tol = atol + rtol * std::fabs(r);
    if (diff > tol) {
      if (bad == 0) {
        first_i = i;
      }
      ++bad;
      max_abs = std::fmax(max_abs, diff);
    }
  }
  if (bad > 0) {
    fprintf(stderr,
            "\n=== Correctness ===\n"
            "FAIL: %zu / %zu elements outside tol (atol=%g rtol=%g)\n"
            "  max_abs_err=%g  first_idx=%zu  gpu=%g  ref=%g\n",
            bad, ref_f32.size(), atol, rtol, max_abs, first_i, bf16_to_float(gpu_out[first_i]), ref_f32[first_i]);
    return false;
  }
  printf("\n=== Correctness ===\n");
  printf("PASS: all %zu outputs within atol=%g rtol=%g (CPU float32 reference, zero pad like conv3d)\n", ref_f32.size(),
         atol, rtol);
  return true;
}

static void launch_reference(__nv_bfloat16* d_in, __nv_bfloat16* d_out, __nv_bfloat16* d_w, __nv_bfloat16* d_bias,
                             int B, int C, int D, int H, int W, int C_out, int D_out, int H_out, int W_out, int blocks,
                             int threads_per_block) {
  conv_depthwise3d_cuda_kernel_reference_bf16<<<blocks, threads_per_block>>>(
      d_in, d_out, d_w, d_bias, B, C, C_out, D, H, W, D_out, H_out, W_out, KD, KH, KW, 1, 1, 1, PaddingD, PaddingH,
      PaddingW, 1, 1, 1);
  CUDA_CHECK(cudaGetLastError());
}

int main(int argc, char** argv) {
  int niter = 10;
  int warmup = 10;
  bool do_check = true;
  int pos = 1;
  if (pos < argc && argv[pos][0] != '-') {
    niter = std::atoi(argv[pos++]);
  }
  if (pos < argc && argv[pos][0] != '-') {
    warmup = std::atoi(argv[pos++]);
  }
  for (; pos < argc; ++pos) {
    if (std::strcmp(argv[pos], "--no-check") == 0) {
      do_check = false;
    }
  }
  if (niter < 1) {
    niter = 1;
  }
  if (warmup < 0) {
    warmup = 0;
  }

  const int B = 1;
  const int C = 512;
  const int D = 61;
  const int H = 45;
  const int W = 80;
  const int C_out = C;
  const int D_out = (D + 2 * PaddingD - (KD - 1) - 1) + 1;
  const int H_out = BLOCK_H;
  const int W_out = BLOCK_W;

  const size_t n_in = static_cast<size_t>(B) * C * D * H * W;
  const size_t n_out = static_cast<size_t>(B) * C_out * D_out * H_out * W_out;
  const size_t n_w = static_cast<size_t>(C_out) * 1 * KD * KH * KW;
  const size_t n_bias = static_cast<size_t>(C_out);

  const int64_t num_output = static_cast<int64_t>(B) * C_out * D_out * H_out * W_out;
  const int threads_per_block = 256;
  const int blocks = static_cast<int>((num_output + threads_per_block - 1) / threads_per_block);
  dim3 block(threads_per_block);

  std::vector<__nv_bfloat16> h_in(n_in), h_w(n_w), h_bias(n_bias), h_out(n_out);
  bf16_fill_pattern(h_in.data(), n_in, 1u);
  bf16_fill_pattern(h_w.data(), n_w, 2u);
  bf16_fill_pattern(h_bias.data(), n_bias, 3u);
  std::memset(h_out.data(), 0, n_out * sizeof(__nv_bfloat16));

  __nv_bfloat16 *d_in = nullptr, *d_out = nullptr, *d_w = nullptr, *d_bias = nullptr;
  CUDA_CHECK(cudaMalloc(&d_in, n_in * sizeof(__nv_bfloat16)));
  CUDA_CHECK(cudaMalloc(&d_out, n_out * sizeof(__nv_bfloat16)));
  CUDA_CHECK(cudaMalloc(&d_w, n_w * sizeof(__nv_bfloat16)));
  CUDA_CHECK(cudaMalloc(&d_bias, n_bias * sizeof(__nv_bfloat16)));

  CUDA_CHECK(cudaMemcpy(d_in, h_in.data(), n_in * sizeof(__nv_bfloat16), cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_w, h_w.data(), n_w * sizeof(__nv_bfloat16), cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_bias, h_bias.data(), n_bias * sizeof(__nv_bfloat16), cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemset(d_out, 0, n_out * sizeof(__nv_bfloat16)));

  int dev_id = 0;
  CUDA_CHECK(cudaGetDevice(&dev_id));
  print_run_header(dev_id);
  print_problem_and_memory(B, C, D, H, W, C_out, D_out, H_out, W_out, n_in, n_out, n_w, n_bias);
  print_kernel_config(blocks, block, num_output);

  printf("\nRunning warmup (%d iterations)...\n", warmup);
  for (int i = 0; i < warmup; ++i) {
    launch_reference(d_in, d_out, d_w, d_bias, B, C, D, H, W, C_out, D_out, H_out, W_out, blocks,
                       threads_per_block);
  }
  CUDA_CHECK(cudaDeviceSynchronize());

  cudaEvent_t ev0, ev1;
  CUDA_CHECK(cudaEventCreate(&ev0));
  CUDA_CHECK(cudaEventCreate(&ev1));
  printf("Running benchmark (niter=%d)...\n", niter);
  CUDA_CHECK(cudaEventRecord(ev0));
  for (int i = 0; i < niter; ++i) {
    launch_reference(d_in, d_out, d_w, d_bias, B, C, D, H, W, C_out, D_out, H_out, W_out, blocks,
                       threads_per_block);
  }
  CUDA_CHECK(cudaEventRecord(ev1));
  CUDA_CHECK(cudaEventSynchronize(ev1));
  float ms = 0.f;
  CUDA_CHECK(cudaEventElapsedTime(&ms, ev0, ev1));

  CUDA_CHECK(cudaMemcpy(h_out.data(), d_out, n_out * sizeof(__nv_bfloat16), cudaMemcpyDeviceToHost));

  const float avg_ms = ms / static_cast<float>(niter);
  print_performance(avg_ms, niter, B, C_out, D_out, H_out, W_out, n_in, n_out, n_w, n_bias);

  if (do_check) {
    std::vector<float> ref_f32;
    cpu_depthwise_conv3d_ref(h_in, h_w, h_bias, ref_f32, B, C, D, H, W, C_out, D_out, H_out, W_out);
    const float atol = 0.02f;
    const float rtol = 0.02f;
    if (!check_output_bf16_vs_ref(h_out, ref_f32, atol, rtol, B, C_out, D_out, H_out, W_out)) {
      CUDA_CHECK(cudaEventDestroy(ev0));
      CUDA_CHECK(cudaEventDestroy(ev1));
      CUDA_CHECK(cudaFree(d_in));
      CUDA_CHECK(cudaFree(d_out));
      CUDA_CHECK(cudaFree(d_w));
      CUDA_CHECK(cudaFree(d_bias));
      return 1;
    }
  } else {
    printf("correctness skipped (--no-check)\n");
  }

  CUDA_CHECK(cudaEventDestroy(ev0));
  CUDA_CHECK(cudaEventDestroy(ev1));
  CUDA_CHECK(cudaFree(d_in));
  CUDA_CHECK(cudaFree(d_out));
  CUDA_CHECK(cudaFree(d_w));
  CUDA_CHECK(cudaFree(d_bias));
  return 0;
}
