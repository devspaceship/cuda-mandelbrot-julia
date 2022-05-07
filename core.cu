#include <iostream>
#include "libs/CLI11.hpp"

__global__ void computation_kernel(float x_min, float x_max, float y_min, float y_max, int width, int height, int iterations)
{
  int idx = threadIdx.x + blockIdx.x * blockDim.x;
  if (idx < width * height)
  {
    // magic happens here
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
  float x_step = (x_max - x_min) / width;
  float y_step = (y_max - y_min) / height;
}

int main(int argc, char *argv[])
{
  float x_min = -2;
  float x_max = 2;
  float y_min = -2;
  float y_max = 2;
  int w = 1920;
  int h = 1080;
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

  return 0;
}
