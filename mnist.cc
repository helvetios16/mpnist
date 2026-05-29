#include "cnpy.h"
#include <cstdio>
#include <iostream>

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

  // Dibujamos las primeras 3 imagenes como arte ASCII junto a su etiqueta.
  for (size_t n = 0; n < 3; n++)
    draw_image(imgs, labels, n, count, height, width);

  return 0;
}
