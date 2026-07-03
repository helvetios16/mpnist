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

// Dibuja la imagen 'n' como arte ASCII con su etiqueta y un marco (igual que
// draw_image en mnist.cc).
static void draw_image(const unsigned char *imgs, const unsigned char *labels,
                       size_t n, size_t height, size_t width) {
  static const char *ramp[] = {" ", "░", "▒", "▓", "█"};
  const int levels = 5;
  const size_t pixel_per_img = height * width;
  const unsigned char *img = imgs + n * pixel_per_img;

  auto horizontal_border = [width](const char *left, const char *mid,
                                   const char *right) {
    std::cout << left;
    for (size_t col = 0; col < width; col++)
      std::cout << mid;
    std::cout << right << "\n";
  };

  std::cout << "Label = " << static_cast<int>(labels[n]) << "\n";
  horizontal_border("┌", "─", "┐");
  for (size_t row = 0; row < height; row++) {
    std::cout << "│";
    for (size_t col = 0; col < width; col++) {
      unsigned char p = img[row * width + col];
      int level = (p * (levels - 1)) / 255;
      std::cout << ramp[level];
    }
    std::cout << "│\n";
  }
  horizontal_border("└", "─", "┘");
}

// Dibuja los pesos de una clase como mapa de calor ASCII (igual que
// draw_weights en mnist.cc).
static void draw_weights(const std::vector<float> &weights, size_t width,
                         size_t height) {
  float wmin = 1e9f, wmax = -1e9f;
  for (float w : weights) {
    if (w > wmax)
      wmax = w;
    if (w < wmin)
      wmin = w;
  }

  auto horizontal_border = [width](const char *left, const char *mid,
                                   const char *right) {
    std::cout << left;
    for (size_t col = 0; col < width; col++)
      std::cout << mid;
    std::cout << right << "\n";
  };

  static const char *ramp[] = {" ", "░", "▒", "▓", "█"};
  const int levels = 5;

  std::cout << "weights\n";
  horizontal_border("┌", "─", "┐");
  for (size_t row = 0; row < height; row++) {
    std::cout << "│";
    for (size_t col = 0; col < width; col++) {
      float w = weights[row * width + col];
      int level = static_cast<int>((w - wmin) / (wmax - wmin) * (levels - 1));
      std::cout << ramp[level];
    }
    std::cout << "│\n";
  }
  horizontal_border("└", "─", "┘");
}

// Corre el kernel en modo prediccion (sin actualizar pesos) y devuelve la
// clase con mayor score.
static int predict_gpu(const float *x_dev, float *W_dev, float *B_dev,
                       float *scores_dev, float *scores_host,
                       size_t pixel_per_img, size_t shared_bytes) {
  perceptron_step_kernel<<<NUM_CLASSES, BLOCK_SIZE, shared_bytes>>>(
      x_dev, W_dev, B_dev, scores_dev, pixel_per_img, 0.0f, -1, false);
  CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(cudaMemcpy(scores_host, scores_dev, NUM_CLASSES * sizeof(float),
                        cudaMemcpyDeviceToHost));
  return argmax(scores_host, NUM_CLASSES);
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

  // Pruebas pequenas: dibuja unas imagenes de test y compara la prediccion
  // del modelo ya entrenado contra la etiqueta real.
  std::cout << "\n--- Pruebas ---\n";
  const size_t num_demo = 5;
  for (size_t i = 0; i < num_demo && i < count_test; i++) {
    draw_image(imgs_test, labels_test, i, height, width);
    const int pred =
        predict_gpu(X_test_dev + i * pixel_per_img, W_dev, B_dev, scores_dev,
                    scores_host, pixel_per_img, shared_bytes);
    std::cout << "Prediccion: " << pred
              << " | Real: " << static_cast<int>(labels_test[i]) << "\n\n";
  }

  int successes = 0;
  for (size_t i = 0; i < count_test; i++) {
    const int label = static_cast<int>(labels_test[i]);
    const int pred =
        predict_gpu(X_test_dev + i * pixel_per_img, W_dev, B_dev, scores_dev,
                    scores_host, pixel_per_img, shared_bytes);
    if (pred == label)
      successes++;
  }
  std::cout << "Results: " << 100.0 * successes / count_test << "%\n";

  // Pesos aprendidos por cada clase, igual que al final de mnist.cc.
  std::vector<float> W_host(NUM_CLASSES * pixel_per_img);
  CUDA_CHECK(cudaMemcpy(W_host.data(), W_dev,
                        NUM_CLASSES * pixel_per_img * sizeof(float),
                        cudaMemcpyDeviceToHost));
  for (int k = 0; k < NUM_CLASSES; k++) {
    std::vector<float> Wk(W_host.begin() + k * pixel_per_img,
                          W_host.begin() + (k + 1) * pixel_per_img);
    std::cout << k << " ";
    draw_weights(Wk, width, height);
  }

  cudaFree(X_train_dev);
  cudaFree(X_test_dev);
  cudaFree(W_dev);
  cudaFree(B_dev);
  cudaFree(scores_dev);

  return 0;
}
