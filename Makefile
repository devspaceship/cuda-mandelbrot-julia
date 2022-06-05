./build/core: ./build/core.o ./build/base64.o
	nvcc ./build/core.o ./build/base64.o -o ./build/core -ccbin=g++-10

./build/core.o: core.cu
	nvcc -c core.cu -o ./build/core.o -ccbin=g++-10

./build/base64.o: ./libs/base64.cpp
	nvcc -c ./libs/base64.cpp -o ./build/base64.o -ccbin=g++-10