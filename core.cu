// MIT License

// Copyright (c) 2022 Thomas Saint-Gérand

// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:

// The above copyright notice and this permission notice shall be included in all
// copies or substantial portions of the Software.

// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.

#include <iostream>
#include <thrust/complex.h>
#include "libs/CLI11.hpp"
#include "libs/base64.h"

__global__ void computation_kernel(float x_min, float x_max, float y_min, float y_max, int width, int height, int iterations, unsigned char *grid_vector)
{
  int idx = threadIdx.x + blockIdx.x * blockDim.x;
  if (idx < width * height)
  {
    int i = idx / width;
    int j = idx % width;
    int x = j * (x_max - x_min) / (width - 1) + x_min;
    int y = i * (y_max - y_min) / (height - 1) + y_min;

    thrust::complex<float> c = thrust::complex<float>(x, y);
    thrust::complex<float> z = thrust::complex<float>(0, 0);
    for (int k = 0; k <= iterations; k++)
    {
      z = z * z + c;
      if (thrust::norm(z) > 2)
      {
        grid_vector[idx] = k;
        break;
      }
    }
  }
}

void set_sizes(int *grid_size, int *block_size, int num_threads)
{
  int min_grid_size;
  cudaOccupancyMaxPotentialBlockSize(&min_grid_size, block_size, computation_kernel, 0, 0);
  *grid_size = (num_threads + *block_size - 1) / *block_size;
}

void print_b64_grid_vector(unsigned char *grid_vector, int width, int height)
{
  std::string encoded_string = base64_encode(grid_vector, width * height);
  std::cout << encoded_string << std::endl;
}

void mandelbrot_set(float x_min, float x_max, float y_min, float y_max, int width, int height, int iterations)
{
  float x_step = (x_max - x_min) / (width - 1);
  float y_step = (y_max - y_min) / (height - 1);

  unsigned char *grid_vector;
  cudaMallocManaged(&grid_vector, width * height * sizeof(int));
  for (int i = 0; i < width * height; i++)
  {
    grid_vector[i] = 0xff;
  }

  int grid_size, block_size;
  set_sizes(&grid_size, &block_size, width * height);

  computation_kernel<<<grid_size, block_size>>>(x_min, x_max, y_min, y_max, width, height, iterations, grid_vector);
  cudaDeviceSynchronize();

  print_b64_grid_vector(grid_vector, width, height);

  cudaFree(grid_vector);
}

int main(int argc, char *argv[])
{
  float x_min = -2;
  float x_max = 2;
  float y_min = -2;
  float y_max = 2;
  int w = 400;
  int h = 400;
  unsigned char iterations = 0xfe;

  CLI::App app{"Compute the number of iterations before leaving bounds"};
  app.add_option("--x_min", x_min, "the minimum x coordinate");
  app.add_option("--x_max", x_max, "the maximum x coordinate");
  app.add_option("--y_min", y_min, "the minimum y coordinate");
  app.add_option("--y_max", y_max, "the maximum y coordinate");
  app.add_option("--width", w, "the horizontal resolution");
  app.add_option("--height", h, "the vertical resolution");
  // app.add_option("--iterations", iterations, "the number of iterations");
  CLI11_PARSE(app, argc, argv);

  mandelbrot_set(x_min, x_max, y_min, y_max, w, h, iterations);

  return 0;
}
