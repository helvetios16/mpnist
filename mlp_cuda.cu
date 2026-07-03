// Version CUDA del perceptron multiclase (0-9) y del MLP de mnist.cc.
// NO compila/corre en esta maquina (Apple M5, sin GPU NVIDIA/nvcc); pensado
// para portarse a una maquina con toolkit CUDA instalado. Build: `make cuda`.

#include "cnpy.h"
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>
#include <iostream>
#include <random>
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

// Perceptron multiclase (GPU): 10 unidades lineales one-vs-all, sin capa
// oculta. Un bloque por clase, reduccion en memoria compartida.

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

// Normaliza todas las imagenes [0,255] -> [0,1] en un solo buffer contiguo
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

// Dibuja la imagen 'n' como arte ASCII con su etiqueta y un marco
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

// Dibuja los pesos de una clase como mapa de calor ASCII
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

// MLP (GPU): 784 -> 128 (ReLU) -> 10 (softmax), backprop con un kernel por
// etapa. Cada neurona de una capa es un bloque que reduce su producto punto en
// memoria compartida (igual patron que perceptron_step_kernel).

static const int HIDDEN = 128;

// z1 = W1*x + b1 ; a1 = ReLU(z1). Un bloque por neurona oculta (HIDDEN
// bloques), reduccion sobre las `input` entradas.
__global__ void mlp_hidden_forward_kernel(const float *x, const float *W1,
                                          const float *b1, float *z1, float *a1,
                                          int input) {
  const int i = blockIdx.x;
  const int tid = threadIdx.x;
  extern __shared__ float shared[];

  float partial = 0.0f;
  for (int j = tid; j < input; j += blockDim.x)
    partial += W1[i * input + j] * x[j];
  shared[tid] = partial;
  __syncthreads();

  for (int s = blockDim.x / 2; s > 0; s >>= 1) {
    if (tid < s)
      shared[tid] += shared[tid + s];
    __syncthreads();
  }

  if (tid == 0) {
    float z = shared[0] + b1[i];
    z1[i] = z;
    a1[i] = (z > 0.0f) ? z : 0.0f;
  }
}

// z2 = W2*a1 + b2. Un bloque por neurona de salida (NUM_CLASSES bloques),
// reduccion sobre las `hidden` activaciones ocultas.
__global__ void mlp_output_forward_kernel(const float *a1, const float *W2,
                                          const float *b2, float *z2,
                                          int hidden) {
  const int k = blockIdx.x;
  const int tid = threadIdx.x;
  extern __shared__ float shared[];

  float partial = 0.0f;
  for (int j = tid; j < hidden; j += blockDim.x)
    partial += W2[k * hidden + j] * a1[j];
  shared[tid] = partial;
  __syncthreads();

  for (int s = blockDim.x / 2; s > 0; s >>= 1) {
    if (tid < s)
      shared[tid] += shared[tid + s];
    __syncthreads();
  }

  if (tid == 0)
    z2[k] = shared[0] + b2[k];
}

// softmax(z2) -> y (resta el maximo por estabilidad) y, si se pide,
// dz2 = y - onehot(label) (el "regalo" de combinar softmax + cross-entropy).
// NUM_CLASSES es tan chico (10) que un solo hilo alcanza.
__global__ void mlp_softmax_kernel(const float *z2, float *y, float *dz2,
                                   int output, int label, bool compute_dz2) {
  float max_z = z2[0];
  for (int k = 1; k < output; k++)
    if (z2[k] > max_z)
      max_z = z2[k];

  float sum = 0.0f;
  for (int k = 0; k < output; k++) {
    y[k] = expf(z2[k] - max_z);
    sum += y[k];
  }
  for (int k = 0; k < output; k++)
    y[k] /= sum;

  if (compute_dz2) {
    for (int k = 0; k < output; k++)
      dz2[k] = y[k] - ((k == label) ? 1.0f : 0.0f);
  }
}

// Gradiente + update de la capa de salida: dW2[k][j] = dz2[k]*a1[j],
// db2[k] = dz2[k]. Ademas acumula da1[j] += W2[k][j] * dz2[k] (con el peso
// *antes* de actualizarlo, como exige backprop) para propagar el error
// hacia la capa oculta. da1 debe estar puesto a 0 antes de este kernel.
__global__ void mlp_output_backward_kernel(const float *a1, float *W2,
                                           float *b2, const float *dz2,
                                           float *da1, int hidden, float lr) {
  const int k = blockIdx.x;
  const int tid = threadIdx.x;

  for (int j = tid; j < hidden; j += blockDim.x) {
    const float w = W2[k * hidden + j];
    atomicAdd(&da1[j], w * dz2[k]);
    W2[k * hidden + j] = w - lr * dz2[k] * a1[j];
  }
  if (tid == 0)
    b2[k] -= lr * dz2[k];
}

// Cruza la ReLU (dz1 = z1>0 ? da1 : 0) y aplica el gradiente + update de la
// capa oculta: dW1[i][j] = dz1[i]*x[j], db1[i] = dz1[i]. Un bloque por
// neurona oculta.
__global__ void mlp_hidden_backward_kernel(const float *x, float *W1, float *b1,
                                           const float *da1, const float *z1,
                                           int input, float lr) {
  const int i = blockIdx.x;
  const int tid = threadIdx.x;
  __shared__ float dz1;

  if (tid == 0) {
    dz1 = (z1[i] > 0.0f) ? da1[i] : 0.0f;
    b1[i] -= lr * dz1;
  }
  __syncthreads();

  for (int j = tid; j < input; j += blockDim.x)
    W1[i * input + j] -= lr * dz1 * x[j];
}

// Inicializacion He (misma que init_mlp en mnist.cc): pesos ~ N(0,
// sqrt(2/fan_in)) generados en host y copiados a la GPU; sesgos en 0.
static void mlp_init(float *W1_dev, float *b1_dev, float *W2_dev, float *b2_dev,
                     size_t input, size_t hidden, size_t output) {
  std::mt19937 gen(42);
  std::normal_distribution<float> d1(0.0f, std::sqrt(2.0f / input));
  std::normal_distribution<float> d2(0.0f, std::sqrt(2.0f / hidden));

  std::vector<float> W1_host(hidden * input);
  for (size_t i = 0; i < W1_host.size(); i++)
    W1_host[i] = d1(gen);
  std::vector<float> W2_host(output * hidden);
  for (size_t i = 0; i < W2_host.size(); i++)
    W2_host[i] = d2(gen);

  CUDA_CHECK(cudaMemcpy(W1_dev, W1_host.data(), W1_host.size() * sizeof(float),
                        cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(W2_dev, W2_host.data(), W2_host.size() * sizeof(float),
                        cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemset(b1_dev, 0, hidden * sizeof(float)));
  CUDA_CHECK(cudaMemset(b2_dev, 0, output * sizeof(float)));
}

// Forward completo (oculta + salida); no incluye softmax porque
// argmax(z2) == argmax(softmax(z2)) y prediccion/evaluacion solo necesitan
// el argmax.
static void mlp_forward_gpu(const float *x_dev, const float *W1_dev,
                            const float *b1_dev, const float *W2_dev,
                            const float *b2_dev, float *z1_dev, float *a1_dev,
                            float *z2_dev, int input, int hidden,
                            size_t shared_bytes) {
  mlp_hidden_forward_kernel<<<hidden, BLOCK_SIZE, shared_bytes>>>(
      x_dev, W1_dev, b1_dev, z1_dev, a1_dev, input);
  CUDA_CHECK(cudaGetLastError());
  mlp_output_forward_kernel<<<NUM_CLASSES, BLOCK_SIZE, shared_bytes>>>(
      a1_dev, W2_dev, b2_dev, z2_dev, hidden);
  CUDA_CHECK(cudaGetLastError());
}

// Forward + argmax(z2) sobre el host.
static int mlp_predict_gpu(const float *x_dev, const float *W1_dev,
                           const float *b1_dev, const float *W2_dev,
                           const float *b2_dev, float *z1_dev, float *a1_dev,
                           float *z2_dev, float *z2_host, int input, int hidden,
                           size_t shared_bytes) {
  mlp_forward_gpu(x_dev, W1_dev, b1_dev, W2_dev, b2_dev, z1_dev, a1_dev, z2_dev,
                  input, hidden, shared_bytes);
  CUDA_CHECK(cudaMemcpy(z2_host, z2_dev, NUM_CLASSES * sizeof(float),
                        cudaMemcpyDeviceToHost));
  return argmax(z2_host, NUM_CLASSES);
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

  float lr = 0.1f;
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

  cudaFree(W_dev);
  cudaFree(B_dev);
  cudaFree(scores_dev);

  // MLP (GPU): 784 -> 128 (ReLU) -> 10 (softmax)
  std::cout << "\n--- MLP ---\n";

  const int input = static_cast<int>(pixel_per_img);
  const int hidden = HIDDEN;
  lr = 0.01f;

  float *W1_dev, *b1_dev, *W2_dev, *b2_dev;
  float *z1_dev, *a1_dev, *z2_dev, *y_dev, *dz2_dev, *da1_dev;
  CUDA_CHECK(cudaMalloc(&W1_dev, hidden * input * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&b1_dev, hidden * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&W2_dev, NUM_CLASSES * hidden * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&b2_dev, NUM_CLASSES * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&z1_dev, hidden * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&a1_dev, hidden * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&z2_dev, NUM_CLASSES * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&y_dev, NUM_CLASSES * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&dz2_dev, NUM_CLASSES * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&da1_dev, hidden * sizeof(float)));

  mlp_init(W1_dev, b1_dev, W2_dev, b2_dev, input, hidden, NUM_CLASSES);

  float y_host[NUM_CLASSES];
  float z2_host_mlp[NUM_CLASSES];

  for (int epoch = 1; epoch <= epochs; epoch++) {
    int correct = 0;
    float loss_sum = 0.0f;

    for (size_t i = 0; i < count_train; i++) {
      const float *x = X_train_dev + i * pixel_per_img;
      const int label = static_cast<int>(labels_train[i]);

      // 1. Forward
      mlp_forward_gpu(x, W1_dev, b1_dev, W2_dev, b2_dev, z1_dev, a1_dev, z2_dev,
                      input, hidden, shared_bytes);

      // 2. Softmax + dz2 = y - onehot(label), y para medir acierto/perdida.
      mlp_softmax_kernel<<<1, 1>>>(z2_dev, y_dev, dz2_dev, NUM_CLASSES, label,
                                   true);
      CUDA_CHECK(cudaGetLastError());
      CUDA_CHECK(cudaMemcpy(y_host, y_dev, NUM_CLASSES * sizeof(float),
                            cudaMemcpyDeviceToHost));
      if (argmax(y_host, NUM_CLASSES) == label)
        correct++;
      loss_sum += -std::log(y_host[label] + 1e-9f);

      // 3. Backward: da1 se acumula por atomicAdd, hay que resetearlo.
      CUDA_CHECK(cudaMemset(da1_dev, 0, hidden * sizeof(float)));
      mlp_output_backward_kernel<<<NUM_CLASSES, BLOCK_SIZE>>>(
          a1_dev, W2_dev, b2_dev, dz2_dev, da1_dev, hidden, lr);
      CUDA_CHECK(cudaGetLastError());

      // 4. Update capa oculta (cruza la ReLU dentro del kernel).
      mlp_hidden_backward_kernel<<<hidden, BLOCK_SIZE>>>(
          x, W1_dev, b1_dev, da1_dev, z1_dev, input, lr);
      CUDA_CHECK(cudaGetLastError());
    }
    std::cout << "Epoch " << epoch << ": " << 100.0 * correct / count_train
              << "%  loss " << loss_sum / count_train << "\n";
  }

  int mlp_successes = 0;
  for (size_t i = 0; i < count_test; i++) {
    const int label = static_cast<int>(labels_test[i]);
    const int pred = mlp_predict_gpu(
        X_test_dev + i * pixel_per_img, W1_dev, b1_dev, W2_dev, b2_dev, z1_dev,
        a1_dev, z2_dev, z2_host_mlp, input, hidden, shared_bytes);
    if (pred == label)
      mlp_successes++;
  }
  std::cout << "Results: " << 100.0 * mlp_successes / count_test << "%\n";

  cudaFree(X_train_dev);
  cudaFree(X_test_dev);
  cudaFree(W1_dev);
  cudaFree(b1_dev);
  cudaFree(W2_dev);
  cudaFree(b2_dev);
  cudaFree(z1_dev);
  cudaFree(a1_dev);
  cudaFree(z2_dev);
  cudaFree(y_dev);
  cudaFree(dz2_dev);
  cudaFree(da1_dev);

  return 0;
}
