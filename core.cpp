#include <iostream>
#include "libs/CLI11.hpp"

int main(int argc, char *argv[])
{
    float x_min = -2;
    float x_max = 2;
    float y_min = -2;
    float y_max = 2;
    int w = 1920;
    int h = 1080;

    CLI::App app{"Compute the number of iterations before leaving bounds"};
    app.add_option("--x_min", x_min, "the minimum x coordinate");
    app.add_option("--x_max", x_max, "the maximum x coordinate");
    app.add_option("--y_min", y_min, "the minimum y coordinate");
    app.add_option("--y_max", y_max, "the maximum y coordinate");
    app.add_option("--width", w, "the horizontal resolution of the computation");
    app.add_option("--height", h, "the vertical resolution of the computation");
    CLI11_PARSE(app, argc, argv);

    return 0;
}
