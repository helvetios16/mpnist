# Makefile para el lector de MNIST (mnist.cc + cnpy)

CXX      = g++
CXXFLAGS = -std=c++17 -Wall -O2
LDLIBS   = -lz

TARGET   = mnist
OBJS     = mnist.o cnpy.o

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

# Borra los objetos y el ejecutable generados
clean:
	rm -f $(TARGET) $(OBJS)

.PHONY: all run clean
