#include <iostream>
#include "libs/CLI11.hpp"

int main(int argc, char *argv[])
{
    float x_a = -2;
    float y_a = 2;
    float x_b = 2;
    float y_b = -2;
    int w = 1920;
    int h = 1080;

    CLI::App app{"Compute the number of iterations before leaving bounds"};
    app.add_option("--x_a", x_a, "the x coordinate of the top left corner");
    app.add_option("--y_a", y_a, "the y coordinate of the top left corner");
    app.add_option("--x_b", x_b, "the x coordinate of the bottom right corner");
    app.add_option("--y_b", y_b, "the y coordinate of the bottom right corner");
    app.add_option("--width", w, "the horizontal resolution of the computation");
    app.add_option("--height", h, "the vertical resolution of the computation");
    CLI11_PARSE(app, argc, argv);

    return 0;
}
