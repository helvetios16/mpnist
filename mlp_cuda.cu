// Version CUDA del perceptron multiclase (0-9) de mnist.cc.
// NO compila/corre en esta maquina (Apple M5, sin GPU NVIDIA/nvcc); pensado
// para portarse a una maquina con toolkit CUDA instalado. Build: `make cuda`.

#include "cnpy.h"
#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>
#include <iostream>
#include <vector>

#define CUDA_CHECK(call)                                                       \
  do {                                                                         \
    cudaError_t err = (call);                                                  \
    if (err != cudaSuccess) {                                                  \
      std::cerr << "CUDA error: " << cudaGetErrorString(err) << " at "         \
                << __FILE__ << ":" << __LINE__ << "\n";                        \
      exit(1);                                                                 \
    }                                                                          \
  } while (0)

static const int NUM_CLASSES = 10;
static const int BLOCK_SIZE = 256;

// Un bloque por clase. Cada bloque calcula el producto punto W[k]·x + B[k]
// (reduccion en memoria compartida) y, si update=true, aplica la regla del
// perceptron (one-vs-all) actualizando W[k] y B[k] en paralelo sobre los
// pixeles del bloque.
__global__ void perceptron_step_kernel(const float *x, float *W, float *B,
                                       float *scores, int pixel_per_img,
                                       float lr, int label, bool update) {
  const int k = blockIdx.x;
  const int tid = threadIdx.x;
  extern __shared__ float shared[];

  float partial = 0.0f;
  for (int v = tid; v < pixel_per_img; v += blockDim.x)
    partial += W[k * pixel_per_img + v] * x[v];
  shared[tid] = partial;
  __syncthreads();

  for (int s = blockDim.x / 2; s > 0; s >>= 1) {
    if (tid < s)
      shared[tid] += shared[tid + s];
    __syncthreads();
  }

  if (tid == 0) {
    float z = shared[0] + B[k];
    scores[k] = z;
    shared[0] = z;
  }
  __syncthreads();
  const float z = shared[0];

  if (update) {
    const int target = (label == k) ? 1 : 0;
    const int pred = (z >= 0.0f) ? 1 : 0;
    const int error = target - pred;
    if (error != 0) {
      for (int v = tid; v < pixel_per_img; v += blockDim.x)
        W[k * pixel_per_img + v] += lr * error * x[v];
      if (tid == 0)
        B[k] += lr * error;
    }
  }
}

// Normaliza todas las imagenes [0,255] -> [0,1] en un solo buffer contiguo,
// igual que build_dataset en mnist.cc.
static std::vector<float> build_dataset(const unsigned char *imgs, size_t count,
                                        size_t pixel_per_img) {
  std::vector<float> X(count * pixel_per_img);
  for (size_t i = 0; i < count * pixel_per_img; i++)
    X[i] = imgs[i] / 255.0f;
  return X;
}

static int argmax(const float *scores, int n) {
  int best = 0;
  for (int k = 1; k < n; k++)
    if (scores[k] > scores[best])
      best = k;
  return best;
}

int main() {
  cnpy::npz_t data = cnpy::npz_load("mnist.npz");
  cnpy::NpyArray x_train = data["x_train"];
  cnpy::NpyArray y_train = data["y_train"];
  cnpy::NpyArray x_test = data["x_test"];
  cnpy::NpyArray y_test = data["y_test"];

  const unsigned char *imgs_train = x_train.data<unsigned char>();
  const unsigned char *labels_train = y_train.data<unsigned char>();
  const unsigned char *imgs_test = x_test.data<unsigned char>();
  const unsigned char *labels_test = y_test.data<unsigned char>();

  const size_t count_train = x_train.shape[0];
  const size_t height = x_train.shape[1];
  const size_t width = x_train.shape[2];
  const size_t pixel_per_img = height * width;
  const size_t count_test = x_test.shape[0];

  std::cout << "x_train: " << count_train << " imagenes de " << height << "x"
            << width << "\n";
  std::cout << "x_test:  " << count_test << " imagenes\n";

  std::vector<float> X_train =
      build_dataset(imgs_train, count_train, pixel_per_img);
  std::vector<float> X_test =
      build_dataset(imgs_test, count_test, pixel_per_img);

  const float lr = 0.1f;
  const int epochs = 5;

  float *X_train_dev, *X_test_dev, *W_dev, *B_dev, *scores_dev;
  CUDA_CHECK(cudaMalloc(&X_train_dev, X_train.size() * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&X_test_dev, X_test.size() * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&W_dev, NUM_CLASSES * pixel_per_img * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&B_dev, NUM_CLASSES * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&scores_dev, NUM_CLASSES * sizeof(float)));

  CUDA_CHECK(cudaMemcpy(X_train_dev, X_train.data(),
                        X_train.size() * sizeof(float),
                        cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(X_test_dev, X_test.data(),
                        X_test.size() * sizeof(float), cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemset(W_dev, 0, NUM_CLASSES * pixel_per_img * sizeof(float)));
  CUDA_CHECK(cudaMemset(B_dev, 0, NUM_CLASSES * sizeof(float)));

  const size_t shared_bytes = BLOCK_SIZE * sizeof(float);
  float scores_host[NUM_CLASSES];

  for (int epoch = 1; epoch <= epochs; epoch++) {
    int successes = 0;
    for (size_t i = 0; i < count_train; i++) {
      const float *x = X_train_dev + i * pixel_per_img;
      const int label = static_cast<int>(labels_train[i]);

      perceptron_step_kernel<<<NUM_CLASSES, BLOCK_SIZE, shared_bytes>>>(
          x, W_dev, B_dev, scores_dev, pixel_per_img, lr, label, true);
      CUDA_CHECK(cudaGetLastError());

      CUDA_CHECK(cudaMemcpy(scores_host, scores_dev,
                            NUM_CLASSES * sizeof(float),
                            cudaMemcpyDeviceToHost));
      if (argmax(scores_host, NUM_CLASSES) == label)
        successes++;
    }
    std::cout << "Epochs " << epoch << ": " << 100.0 * successes / count_train
              << "%\n";
  }

  int successes = 0;
  for (size_t i = 0; i < count_test; i++) {
    const float *x = X_test_dev + i * pixel_per_img;
    const int label = static_cast<int>(labels_test[i]);

    perceptron_step_kernel<<<NUM_CLASSES, BLOCK_SIZE, shared_bytes>>>(
        x, W_dev, B_dev, scores_dev, pixel_per_img, lr, label, false);
    CUDA_CHECK(cudaGetLastError());

    CUDA_CHECK(cudaMemcpy(scores_host, scores_dev, NUM_CLASSES * sizeof(float),
                          cudaMemcpyDeviceToHost));
    if (argmax(scores_host, NUM_CLASSES) == label)
      successes++;
  }
  std::cout << "Results: " << 100.0 * successes / count_test << "%\n";

  cudaFree(X_train_dev);
  cudaFree(X_test_dev);
  cudaFree(W_dev);
  cudaFree(B_dev);
  cudaFree(scores_dev);

  return 0;
}
