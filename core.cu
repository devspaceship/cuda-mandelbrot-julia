#include <iostream>
#include "libs/CLI11.hpp"

#include <thrust/complex.h>

__global__ void computation_kernel(float x_min, float x_max, float y_min, float y_max, int width, int height, int iterations, int *grid_vector)
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
    for (int k = 0; k < iterations; k++)
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

void mandelbrot_set(float x_min, float x_max, float y_min, float y_max, int width, int height, int iterations)
{
  float x_step = (x_max - x_min) / (width - 1);
  float y_step = (y_max - y_min) / (height - 1);

  int *grid_vector;
  cudaMallocManaged(&grid_vector, width * height * sizeof(int));
  for (int i = 0; i < width * height; i++)
  {
    grid_vector[i] = -1;
  }

  int grid_size, block_size;
  set_sizes(&grid_size, &block_size, width * height);

  computation_kernel<<<grid_size, block_size>>>(x_min, x_max, y_min, y_max, width, height, iterations, grid_vector);

  for (int i = 0; i < height; i++)
  {
    for (int j = 0; j < width; j++)
    {
      int idx = i * width + j;
      if (grid_vector[idx] == -1)
      {
        std::cout << ". ";
      }
      else
      {
        std::cout << grid_vector[idx] << " ";
      }
    }
    std::cout << std::endl;
  }

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
  int iterations = 100;

  CLI::App app{"Compute the number of iterations before leaving bounds"};
  app.add_option("--x_min", x_min, "the minimum x coordinate");
  app.add_option("--x_max", x_max, "the maximum x coordinate");
  app.add_option("--y_min", y_min, "the minimum y coordinate");
  app.add_option("--y_max", y_max, "the maximum y coordinate");
  app.add_option("--width", w, "the horizontal resolution");
  app.add_option("--height", h, "the vertical resolution");
  app.add_option("--iterations", iterations, "the number of iterations");
  CLI11_PARSE(app, argc, argv);

  mandelbrot_set(x_min, x_max, y_min, y_max, w, h, iterations);

  return 0;
}
