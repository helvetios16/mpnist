#include "cnpy.h"
#include <algorithm>
#include <cerrno>
#include <cmath>
#include <cstddef>
#include <cstdio>
#include <iostream>
#include <random>
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

// Perceptron multiclase (0-9): 10 unidades lineales one-vs-all, sin capa
// oculta. Cada clase tiene su propio W/b y compite por argmax(score).

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

// MLP (perceptron multicapa): 784 -> 128 (ReLU) -> 10 (softmax), entrenado
// con backpropagation + descenso de gradiente.

// Agrupa los 4 grupos de parametros del MLP.
struct MLP {
  std::vector<std::vector<float>> W1; // hidden x input  (entrada -> oculta)
  std::vector<float> b1;              // hidden          (sesgos oculta)
  std::vector<std::vector<float>> W2; // output x hidden (oculta -> salida)
  std::vector<float> b2;              // output          (sesgos salida)
};

// Valores intermedios de un forward pass (se reutilizan en el backward).
struct Forward {
  std::vector<float> z1, a1; // capa oculta: suma ponderada y activacion ReLU
  std::vector<float> z2, y;  // capa salida: logits y probabilidades softmax
};

// Gradientes de los 4 grupos de parametros, misma forma que MLP.
struct Grads {
  std::vector<std::vector<float>> dW1, dW2;
  std::vector<float> db1, db2;
};

// Inicializacion: pesos ~ N(0, sqrt(2/fan_in)), sesgos en 0. Necesaria
// porque, a diferencia del perceptron, iniciar todo en cero deja a las
// neuronas ocultas identicas para siempre (nunca se diferencian).
void init_mlp(MLP &net, size_t input, size_t hidden, size_t output) {
  std::mt19937 gen(42); // semilla fija -> reproducible
  std::normal_distribution<float> d1(0.0f, std::sqrt(2.0f / input));
  std::normal_distribution<float> d2(0.0f, std::sqrt(2.0f / hidden));

  net.W1.assign(hidden, std::vector<float>(input));
  for (size_t i = 0; i < hidden; i++)
    for (size_t j = 0; j < input; j++)
      net.W1[i][j] = d1(gen);
  net.b1.assign(hidden, 0.0f);

  net.W2.assign(output, std::vector<float>(hidden));
  for (size_t k = 0; k < output; k++)
    for (size_t j = 0; j < hidden; j++)
      net.W2[k][j] = d2(gen);
  net.b2.assign(output, 0.0f);
}

Forward forward(const MLP &net, const float *x) {
  Forward out;
  const size_t hidden = net.b1.size();
  const size_t output = net.b2.size();
  const size_t input = net.W1[0].size();

  // Capa oculta: z1 = W1*x + b1, luego a1 = ReLU(z1)
  out.z1.assign(hidden, 0.0f);
  out.a1.assign(hidden, 0.0f);
  for (size_t i = 0; i < hidden; i++) {
    float z = net.b1[i];
    for (size_t j = 0; j < input; j++)
      z += net.W1[i][j] * x[j];
    out.z1[i] = z;
    out.a1[i] = (z > 0) ? z : 0.0f; // ReLU
  }

  // Capa salida: z2 = W2*a1 + b2
  out.z2.assign(output, 0.0f);
  for (size_t k = 0; k < output; k++) {
    float z = net.b2[k];
    for (size_t j = 0; j < hidden; j++)
      z += net.W2[k][j] * out.a1[j];
    out.z2[k] = z;
  }

  // Softmax sobre z2 (resta del maximo por estabilidad numerica)
  out.y.assign(output, 0.0f);
  float max_z = *std::max_element(out.z2.begin(), out.z2.end());
  float sum = 0.0f;
  for (size_t k = 0; k < output; k++) {
    out.y[k] = std::exp(out.z2[k] - max_z);
    sum += out.y[k];
  }
  for (size_t k = 0; k < output; k++)
    out.y[k] /= sum;

  return out;
}

// Cross-entropy de un ejemplo: -log de la probabilidad dada al correcto.
float cross_entropy(const std::vector<float> &y, int label) {
  float p = y[label];
  return -std::log(p + 1e-9f); // +epsilon para no calcular log(0)
}

// Backprop para un ejemplo. Reparte la culpa del error hacia atras,
// capa por capa, usando la regla de la cadena.
Grads backward(const MLP &net, const Forward &fwd, const float *x, int label) {
  Grads g;
  const size_t hidden = net.b1.size();
  const size_t output = net.b2.size();
  const size_t input = net.W1[0].size();

  g.dW2.assign(output, std::vector<float>(hidden));
  g.db2.assign(output, 0.0f);
  g.dW1.assign(hidden, std::vector<float>(input));
  g.db1.assign(hidden, 0.0f);

  // dz2 = y - t   (t = one-hot del label correcto). Es el "regalo" de
  // combinar softmax con cross-entropy: el gradiente se simplifica a esto.
  std::vector<float> dz2(output);
  for (size_t k = 0; k < output; k++)
    dz2[k] = fwd.y[k];
  dz2[label] -= 1.0f;

  // Gradientes capa salida: dW2[k][j] = dz2[k] * a1[j] ; db2[k] = dz2[k]
  for (size_t k = 0; k < output; k++) {
    for (size_t j = 0; j < hidden; j++)
      g.dW2[k][j] = dz2[k] * fwd.a1[j];
    g.db2[k] = dz2[k];
  }

  // Propagar el error hacia la capa oculta: da1 = W2^T * dz2
  std::vector<float> da1(hidden, 0.0f);
  for (size_t k = 0; k < output; k++)
    for (size_t j = 0; j < hidden; j++)
      da1[j] += net.W2[k][j] * dz2[k];

  // Cruzar la ReLU: si z1 <= 0 el gradiente no fluye (se corta a 0).
  std::vector<float> dz1(hidden);
  for (size_t j = 0; j < hidden; j++)
    dz1[j] = (fwd.z1[j] > 0) ? da1[j] : 0.0f;

  // Gradientes capa oculta: dW1[i][j] = dz1[i] * x[j] ; db1[i] = dz1[i]
  for (size_t i = 0; i < hidden; i++) {
    for (size_t j = 0; j < input; j++)
      g.dW1[i][j] = dz1[i] * x[j];
    g.db1[i] = dz1[i];
  }

  return g;
}

// Entrena por descenso de gradiente sobre el dataset (un ejemplo a la vez):
// forward -> medir (acierto + perdida) -> backward -> update.
void train_mlp(MLP &net, const std::vector<float> &X,
               const std::vector<int> &idx, const unsigned char *labels,
               size_t pixel_per_img, float lr, int epochs) {
  const size_t hidden = net.b1.size();
  const size_t output = net.b2.size();
  const size_t input = net.W1[0].size();

  for (int e = 0; e < epochs; e++) {
    int correct = 0;
    float loss_sum = 0.0f;

    for (size_t i = 0; i < idx.size(); i++) {
      const float *x = &X[i * pixel_per_img];
      const int label = static_cast<int>(labels[idx[i]]);

      // 1. Forward: predice las probabilidades.
      Forward fwd = forward(net, x);

      // 2. Medir: acierto + perdida del ejemplo.
      int pred = 0;
      for (size_t k = 1; k < output; k++)
        if (fwd.y[k] > fwd.y[pred])
          pred = k;
      if (pred == label)
        correct++;
      loss_sum += cross_entropy(fwd.y, label);

      // 3. Backward: gradiente de cada parametro.
      Grads g = backward(net, fwd, x, label);

      // 4. Update: parametro -= lr * gradiente.
      for (size_t k = 0; k < output; k++) {
        for (size_t j = 0; j < hidden; j++)
          net.W2[k][j] -= lr * g.dW2[k][j];
        net.b2[k] -= lr * g.db2[k];
      }
      for (size_t h = 0; h < hidden; h++) {
        for (size_t j = 0; j < input; j++)
          net.W1[h][j] -= lr * g.dW1[h][j];
        net.b1[h] -= lr * g.db1[h];
      }
    }
    std::cout << "Epoch " << e + 1 << ": " << 100.0 * correct / idx.size()
              << "%  loss " << loss_sum / idx.size() << "\n";
  }
}

// Prediccion: argmax de las probabilidades del forward.
int predict_mlp(const MLP &net, const float *x) {
  Forward fwd = forward(net, x);
  int best = 0;
  for (size_t k = 1; k < fwd.y.size(); k++)
    if (fwd.y[k] > fwd.y[best])
      best = k;
  return best;
}

// Evaluacion: cuenta aciertos sobre el set de prueba.
void evaluate_mlp(const MLP &net, const std::vector<float> &X,
                  const std::vector<int> &idx, const unsigned char *labels,
                  size_t pixel_per_img) {
  int correct = 0;
  for (size_t i = 0; i < idx.size(); i++) {
    const float *x = &X[i * pixel_per_img];
    if (predict_mlp(net, x) == labels[idx[i]])
      correct++;
  }
  std::cout << "Results: " << 100.0 * correct / idx.size() << "%\n";
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

  // MLP: 784 -> 128 -> 10
  std::cout << "\nMLP\n";
  MLP net;
  init_mlp(net, pixel_per_img, 128, 10);
  train_mlp(net, X_train_all, XA_train, labels, pixel_per_img, 0.01f, 5);
  evaluate_mlp(net, X_test_all, XA_test, labels_test, pixel_per_img);

  return 0;
}
