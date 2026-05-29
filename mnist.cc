#include "cnpy.h"
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
  static const char ramp[] = " .:-=+*#%@";

  const size_t pixel_per_img = height * width;
  const unsigned char *img = imgs + n * pixel_per_img;

  // Borde horizontal "+----...----+" que encaja con los lados "|" de cada fila
  // (mide width + 2 para incluir las dos columnas de los bordes laterales).
  auto horizontal_border = [width]() {
    std::cout << '+';
    for (size_t col = 0; col < width; col++)
      std::cout << '-';
    std::cout << "+\n";
  };

  std::cout << "Label = " << static_cast<int>(labels[n]) << "\n";
  horizontal_border();
  for (size_t row = 0; row < height; row++) {
    std::cout << '|';
    for (size_t col = 0; col < width; col++) {
      unsigned char p = img[row * width + col];
      int level = (p * 9) / 255;
      std::cout << ramp[level];
    }
    std::cout << "|\n";
  }
  horizontal_border();
  return true;
}

bool draw_weights(const std::vector<double> &weights, const size_t &width,
                  const size_t &height) {
  double wmin = 1e9, wmax = -1e9;
  for (double w : weights) {
    if (w > wmax)
      wmax = w;
    if (w < wmin)
      wmin = w;
  }

  auto horizontal_border = [width]() {
    std::cout << '+';
    for (size_t col = 0; col < width; col++)
      std::cout << '-';
    std::cout << "+\n";
  };

  std::cout << "weights\n";
  static const char ramp[] = " .:-=+*#%@";
  horizontal_border();
  for (size_t row = 0; row < height; row++) {
    std::cout << '|';
    for (size_t col = 0; col < width; col++) {
      double w = weights[row * width + col];
      int level = static_cast<int>((w - wmin) / (wmax - wmin) * 9);
      std::cout << ramp[level];
    }
    std::cout << "|\n";
  }
  horizontal_border();

  return true;
}

std::vector<double> normalize_image(const unsigned char *imgs, size_t n,
                                    size_t pixel_per_img) {
  std::vector<double> normalize(pixel_per_img, 0.0);
  const unsigned char *img = imgs + n * pixel_per_img;

  for (size_t i = 0; i < pixel_per_img; i++)
    normalize[i] = img[i] / 255.0;

  return normalize;
}

int predict(const std::vector<double> &weight, double bias,
            const std::vector<double> &x) {
  double z = bias;
  for (size_t i = 0; i < weight.size(); i++) {
    z += weight[i] * x[i];
  }
  return (z >= 0) ? 1 : 0;
}

void training(std::vector<int> &x_train_bin, const unsigned char *imgs,
              const unsigned char *labels, double &bias,
              const size_t pixel_per_img, std::vector<double> &weights,
              double lr, int epochs) {
  int total = epochs;
  while (epochs > 0) {
    int succeses = 0;

    for (size_t i = 0; i < x_train_bin.size(); i++) {
      std::vector<double> x =
          normalize_image(imgs, x_train_bin[i], pixel_per_img);
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

void evaluate(const std::vector<int> &x_test_bin,
              const unsigned char *labels_test,
              const std::vector<double> &weights, const double &bias,
              const unsigned char *imgs, const size_t pixel_per_img) {
  int succeses = 0;
  for (size_t i = 0; i < x_test_bin.size(); i++) {
    std::vector<double> x = normalize_image(imgs, x_test_bin[i], pixel_per_img);
    const int target = static_cast<int>(labels_test[x_test_bin[i]]);
    const int predi = predict(weights, bias, x);
    if (predi == target)
      succeses++;
  }
  std::cout << "Results: " << 100.0 * succeses / x_test_bin.size() << "%\n";
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

  std::vector<double> weights(pixel_per_img, 0.0);
  double bias = 0.0;

  // Prueba
  std::vector<double> foo =
      normalize_image(imgs, x_train_bin[0], pixel_per_img);
  double min = 100.0, max = -1.0;
  for (auto v : foo) {
    if (v < min)
      min = v;
    if (v > max)
      max = v;
  }
  std::cout << min << " " << max << "\n";
  std::cout << predict(weights, bias, foo) << "\n";

  training(x_train_bin, imgs, labels, bias, pixel_per_img, weights, 0.1, 5);

  const unsigned char *imgs_test = x_test.data<unsigned char>();
  const unsigned char *labels_test = y_test.data<unsigned char>();

  const size_t count_test = x_test.shape[0];

  std::vector<int> x_test_bin;

  for (size_t n = 0; n < count_test; n++) {
    if (numbers.count(labels_test[n]))
      x_test_bin.push_back(n);
  }

  std::cout << x_test_bin.size() << "\n";

  evaluate(x_test_bin, labels_test, weights, bias, imgs_test, pixel_per_img);

  draw_weights(weights, width, height);

  return 0;
}
