# MNIST — Perceptrón y MLP (CPU/CUDA)

Proyecto que implementa, desde cero y en distintas variantes, clasificadores de dígitos escritos a mano sobre el dataset **MNIST**: desde un perceptrón binario simple hasta un perceptrón multiclase y un MLP (perceptrón multicapa) con backpropagation, tanto en CPU (C++) como en GPU (CUDA).

## Integrantes

- Salim Adrian Jorge Rodríguez
- Andrea Lucía Cuela Morales
- Kristopher Rospigliosi Gonzales
- Sebastián Andrés Mendoza Fernández

## Contenido del repo

### Código C++ (CPU)

- **`mnist.cc`** — Programa principal. Carga el dataset MNIST (`mnist.npz`) usando `cnpy`, dibuja imágenes como arte ASCII en terminal, y entrena/evalúa dos modelos:
  - Un **perceptrón multiclase** (0–9), 10 unidades lineales one-vs-all.
  - Un **MLP** con arquitectura `784 → 128 → 10`, inicialización He, activación ReLU, softmax + cross-entropy y descenso de gradiente (backpropagation).
- **`cnpy.cc` / `cnpy.h`** — Librería de terceros para leer archivos `.npy`/`.npz` (formato de NumPy) desde C++.
- **`mnist.npz`** — Dataset MNIST (imágenes y etiquetas de entrenamiento/test) en formato NumPy comprimido.
- **`Makefile`** — Reglas de compilación:
  - `make` / `make run` — compila y ejecuta la versión CPU (`mnist`).
  - `make cuda` / `make run-cuda` — compila y ejecuta la versión GPU (requiere `nvcc` y GPU NVIDIA).
  - `make clean` — limpia binarios y objetos.

### Código CUDA (GPU)

- **`mlp_cuda.cu`** — Port a CUDA del perceptrón multiclase y del MLP de `mnist.cc`: kernels para el forward (ReLU + softmax), el backward del MLP, y el entrenamiento/predicción en paralelo en GPU. No compila en esta máquina de desarrollo (Apple Silicon, sin GPU NVIDIA), por lo que su ejecución y verificación se hicieron en un entorno con GPU real:

  **Ejecución en GPU (Google Colab):** https://colab.research.google.com/drive/1-7CVT8dKPcduWH9IaOqNJRLBEMGTa4JU?hl=es#scrollTo=RDx6pA9OhMbj

## Cómo correr la versión CPU

```bash
make run
```

## Cómo correr la versión CUDA

Requiere `nvcc` y una GPU NVIDIA (no disponible en esta máquina de desarrollo):

```bash
make run-cuda
```

Para verla en acción sin GPU local, usar el notebook de Colab enlazado arriba.
