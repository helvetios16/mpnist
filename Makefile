# Makefile para el lector de MNIST (mnist.cc + cnpy)

CXX      = g++
CXXFLAGS = -std=c++17 -Wall -O2
LDLIBS   = -lz

NVCC      = nvcc
NVCCFLAGS = -std=c++17 -O2

TARGET   = mnist
OBJS     = mnist.o cnpy.o

CUDA_TARGET = mlp_cuda
CUDA_OBJS   = mlp_cuda.o cnpy.o

# Objetivo por defecto: compila el ejecutable
all: $(TARGET)

# Enlaza los objetos en el ejecutable final
$(TARGET): $(OBJS)
	$(CXX) $(CXXFLAGS) $(OBJS) $(LDLIBS) -o $(TARGET)

# Nuestro codigo: se compila con todas las advertencias activadas
mnist.o: mnist.cc cnpy.h
	$(CXX) $(CXXFLAGS) -c mnist.cc -o mnist.o

# Libreria de terceros: silenciamos sus advertencias con -w
cnpy.o: cnpy.cc cnpy.h
	$(CXX) $(CXXFLAGS) -w -c cnpy.cc -o cnpy.o

# Compila y ejecuta
run: $(TARGET)
	./$(TARGET)

# Version CUDA (requiere nvcc y una GPU NVIDIA; no corre en Apple Silicon)
cuda: $(CUDA_TARGET)

$(CUDA_TARGET): $(CUDA_OBJS)
	$(NVCC) $(NVCCFLAGS) $(CUDA_OBJS) $(LDLIBS) -o $(CUDA_TARGET)

mlp_cuda.o: mlp_cuda.cu cnpy.h
	$(NVCC) $(NVCCFLAGS) -c mlp_cuda.cu -o mlp_cuda.o

run-cuda: $(CUDA_TARGET)
	./$(CUDA_TARGET)

# Borra los objetos y los ejecutables generados
clean:
	rm -f $(TARGET) $(CUDA_TARGET) $(OBJS) mlp_cuda.o

.PHONY: all run cuda run-cuda clean
