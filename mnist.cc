#include "cnpy.h"
#include <cerrno>
#include <cstddef>
#include <cstdio>
#include <iostream>
#include <set>
#include <vector>

// Dibuja la imagen 'n' como arte ASCII con su etiqueta y un marco.
// 'count' es el numero total de imagenes disponibles (p.ej. 60000): si 'n' se
// sale de ese rango la funcion avisa y devuelve false sin dibujar nada.
bool draw_image(const unsigned char *imgs, const unsigned char *labels,
                size_t n, size_t count, size_t height, size_t width) {
  // Contingencia: el indice debe estar dentro de [0, count).
  if (n >= count) {
    std::cerr << "Error: indice " << n << " fuera de rango (maximo "
              << count - 1 << ")\n";
    return false;
  }

  // Rampa de 10 niveles: de mas oscuro (espacio) a mas claro (@).
  // El indice se obtiene escalando el valor 0-255 al rango 0-9.
  static const char *ramp[] = {" ", "░", "▒", "▓", "█"};
  const int levels = 5;

  const size_t pixel_per_img = height * width;
  const unsigned char *img = imgs + n * pixel_per_img;

  // Borde horizontal "+----...----+" que encaja con los lados "|" de cada fila
  // (mide width + 2 para incluir las dos columnas de los bordes laterales).
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
  return true;
}

bool draw_weights(const std::vector<float> &weights, const size_t &width,
                  const size_t &height) {
  float wmin = 1e9, wmax = -1e9;
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

  std::cout << "weights\n";
  static const char *ramp[] = {" ", "░", "▒", "▓", "█"};
  const int levels = 5;

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

  return true;
}

std::vector<float> normalize_image(const unsigned char *imgs, size_t n,
                                   size_t pixel_per_img) {
  std::vector<float> normalize(pixel_per_img, 0.0);
  const unsigned char *img = imgs + n * pixel_per_img;

  for (size_t i = 0; i < pixel_per_img; i++)
    normalize[i] = img[i] / 255.0;

  return normalize;
}

int predict(const std::vector<float> &weight, float bias, const float *x) {
  float z = bias;
  for (size_t i = 0; i < weight.size(); i++) {
    z += weight[i] * x[i];
  }
  return (z >= 0) ? 1 : 0;
}

void training(const std::vector<float> &X, std::vector<int> &x_train_bin,
              const unsigned char *labels, float &bias,
              const size_t pixel_per_img, std::vector<float> &weights, float lr,
              int epochs) {
  int total = epochs;
  while (epochs > 0) {
    int succeses = 0;

    for (size_t i = 0; i < x_train_bin.size(); i++) {
      const float *x = &X[i * pixel_per_img];
      const int target = static_cast<int>(labels[x_train_bin[i]]);
      const int predi = predict(weights, bias, x);
      if (predi == target)
        succeses++;
      int error = target - predi;

      if (error != 0) {
        for (size_t v = 0; v < weights.size(); v++) {
          weights[v] += lr * error * x[v];
        }
        bias += lr * error;
      }
    }
    std::cout << "Epochs " << total - epochs + 1 << ": "
              << 100.0 * succeses / x_train_bin.size() << "%\n";
    epochs--;
  }
}

void evaluate(const std::vector<float> &X, const std::vector<int> &x_test_bin,
              const unsigned char *labels_test,
              const std::vector<float> &weights, const float &bias,
              const size_t pixel_per_img) {
  int succeses = 0;
  for (size_t i = 0; i < x_test_bin.size(); i++) {
    const float *x = &X[i * pixel_per_img];
    const int target = static_cast<int>(labels_test[x_test_bin[i]]);
    const int predi = predict(weights, bias, x);
    if (predi == target)
      succeses++;
  }
  std::cout << "Results: " << 100.0 * succeses / x_test_bin.size() << "%\n";
}

std::vector<float> build_dataset(const std::vector<int> &data,
                                 const unsigned char *imgs,
                                 size_t pixel_per_img) {
  std::vector<float> X(data.size() * pixel_per_img);
  for (size_t i = 0; i < data.size(); i++) {
    const unsigned char *img = imgs + data[i] * pixel_per_img;
    for (size_t p = 0; p < pixel_per_img; p++) {
      X[i * pixel_per_img + p] = img[p] / 255.0;
    }
  }
  return X;
}

float score(const std::vector<float> &weights, float bias, const float *x) {
  float z = bias;
  for (size_t i = 0; i < weights.size(); i++) {
    z += weights[i] * x[i];
  }
  return z;
}

int predict_class(const std::vector<std::vector<float>> &W,
                  const std::vector<float> &B, const float *x) {
  size_t best_class = 0;
  float best_score = -1e9;
  for (size_t i = 0; i < B.size(); i++) {
    float z = score(W[i], B[i], x);
    if (z > best_score) {
      best_score = z;
      best_class = i;
    }
  }
  return best_class;
}

void training_multi(const std::vector<float> &X,
                    const std::vector<int> &X_train,
                    const unsigned char *labels,
                    std::vector<std::vector<float>> &W, std::vector<float> &B,
                    const size_t pixel_per_img, float lr, int epochs) {
  int total = epochs;
  while (epochs > 0) {
    int succeses = 0;

    for (size_t i = 0; i < X_train.size(); i++) {
      const float *x = &X[i * pixel_per_img];
      const int label = static_cast<int>(labels[X_train[i]]);

      if (predict_class(W, B, x) == label)
        succeses++;

      for (size_t k = 0; k < B.size(); k++) {
        const int target = (label == (int)k) ? 1 : 0;
        const int pred = (score(W[k], B[k], x) >= 0) ? 1 : 0;
        const int error = target - pred;

        if (error != 0) {
          for (size_t v = 0; v < W[k].size(); v++) {
            W[k][v] += lr * error * x[v];
          }
          B[k] += lr * error;
        }
      }
    }

    std::cout << "Epochs " << total - epochs + 1 << ": "
              << 100.0 * succeses / X_train.size() << "%\n";
    epochs--;
  }
}

void evaluate_multi(const std::vector<float> &X, const std::vector<int> &X_test,
                    const unsigned char *labels,
                    const std::vector<std::vector<float>> &W,
                    const std::vector<float> &B, const size_t pixel_per_img) {
  int succeses = 0;
  for (size_t i = 0; i < X_test.size(); i++) {
    const float *x = &X[i * pixel_per_img];
    const int label = static_cast<int>(labels[X_test[i]]);

    if (predict_class(W, B, x) == label)
      succeses++;
  }
  std::cout << "Results: " << 100.0 * succeses / X_test.size() << "%\n";
}

int main() {
  // mnist.npz es un ZIP que contiene 4 arrays .npy: x_train, y_train,
  // x_test, y_test. Cargamos el archivo completo como un mapa
  // clave -> NpyArray y accedemos por nombre.
  cnpy::npz_t data = cnpy::npz_load("mnist.npz");
  cnpy::NpyArray x_train = data["x_train"];
  cnpy::NpyArray y_train = data["y_train"];
  cnpy::NpyArray x_test = data["x_test"];
  cnpy::NpyArray y_test = data["y_test"];

  // Mostramos la forma (shape) de cada array para verificar la carga.
  std::cout << "Data verifity:\n";
  std::cout << "x_train shape: ";
  for (size_t d : x_train.shape)
    std::cout << d << " ";
  std::cout << "\n";

  std::cout << "y_train shape: ";
  for (size_t d : y_train.shape)
    std::cout << d << " ";
  std::cout << "\n";

  std::cout << "x_test shape: ";
  for (size_t d : x_test.shape)
    std::cout << d << " ";
  std::cout << "\n";

  std::cout << "y_test shape: ";
  for (size_t d : y_test.shape)
    std::cout << d << " ";
  std::cout << "\n";

  // El dtype real es uint8 (0-255), por eso accedemos como unsigned char.
  const unsigned char *imgs = x_train.data<unsigned char>();
  const unsigned char *labels = y_train.data<unsigned char>();

  // shape[0] es el numero total de imagenes (60000 en x_train).
  const size_t count = x_train.shape[0];
  const size_t height = x_train.shape[1];
  const size_t width = x_train.shape[2];

  const size_t pixel_per_img = height * width;

  // Dibujamos las primeras 3 imagenes como arte ASCII junto a su etiqueta.
  // for (size_t n = 0; n < 3; n++)
  //   draw_image(imgs, labels, n, count, height, width);

  draw_image(imgs, labels, 50, count, height, width);

  std::set<int> numbers = {0, 1};
  std::vector<int> x_train_bin;

  for (size_t n = 0; n < count; n++) {
    if (numbers.count(labels[n]))
      x_train_bin.push_back(n);
  }

  std::cout << x_train_bin.size() << "\n";

  std::vector<float> X_train = build_dataset(x_train_bin, imgs, pixel_per_img);

  // 0 - 1
  std::vector<float> weights(pixel_per_img, 0.0);
  float bias = 0.0;

  // Prueba
  std::vector<float> foo = normalize_image(imgs, x_train_bin[0], pixel_per_img);
  float min = 100.0, max = -1.0;
  for (auto v : foo) {
    if (v < min)
      min = v;
    if (v > max)
      max = v;
  }
  std::cout << min << " " << max << "\n";
  std::cout << predict(weights, bias, foo.data()) << "\n";

  training(X_train, x_train_bin, labels, bias, pixel_per_img, weights, 0.1, 5);

  const unsigned char *imgs_test = x_test.data<unsigned char>();
  const unsigned char *labels_test = y_test.data<unsigned char>();

  const size_t count_test = x_test.shape[0];

  std::vector<int> x_test_bin;

  for (size_t n = 0; n < count_test; n++) {
    if (numbers.count(labels_test[n]))
      x_test_bin.push_back(n);
  }

  std::cout << x_test_bin.size() << "\n";

  std::vector<float> X_test =
      build_dataset(x_test_bin, imgs_test, pixel_per_img);

  evaluate(X_test, x_test_bin, labels_test, weights, bias, pixel_per_img);

  // draw_weights(weights, width, height);

  std::vector<int> XA_train;
  for (int i = 0; i < count; i++)
    XA_train.push_back(i);

  std::vector<float> X_train_all = build_dataset(XA_train, imgs, pixel_per_img);

  // 0 - 9
  std::vector<std::vector<float>> W(10, std::vector<float>(pixel_per_img, 0.0));
  std::vector<float> B(10, 0.0);

  training_multi(X_train_all, XA_train, labels, W, B, pixel_per_img, 0.1, 5);

  std::vector<int> XA_test;
  for (int i = 0; i < count_test; i++)
    XA_test.push_back(i);

  std::vector<float> X_test_all =
      build_dataset(XA_test, imgs_test, pixel_per_img);

  evaluate_multi(X_test_all, XA_test, labels_test, W, B, pixel_per_img);

  for (int i = 0; i < W.size(); i++) {
    std::cout << i << " ";
    draw_weights(W[i], width, height);
  }

  return 0;
}
