#include <iostream>
#include <getopt.h>

void usage()
{
    std::cout
        << "Usage: ./core [OPTION]...\n"
        << "Compute the number of iterations before leaving bounds\n\n"
        << "  --x_a             the x coordinate of the top left corner\n"
        << "  --y_a             the y coordinate of the top left corner\n"
        << "  --x_b             the x coordinate of the bottom right corner\n"
        << "  --y_b             the y coordinate of the bottom right corner\n"
        << "  --width           the horizontal resolution of the computation\n"
        << "  --height          the vertical resolution of the computation\n"
        << "  -h, --help        display this help and exit\n"
        << std::endl;
}

int main(int argc, char *argv[])
{
    float x_a = -2;
    float y_a = 2;
    float x_b = 2;
    float y_b = -2;
    int w = 1920;
    int h = 1080;

    static struct option long_options[] = {
        {"x_a", required_argument, NULL, 0},
        {"y_a", required_argument, NULL, 0},
        {"x_b", required_argument, NULL, 0},
        {"y_b", required_argument, NULL, 0},
        {"width", required_argument, NULL, 0},
        {"height", required_argument, NULL, 0},
        {"help", no_argument, NULL, 'h'},
        {NULL, 0, NULL, 0},
    };

    int option_index = 0;
    char c;
    while ((c = getopt_long(argc, argv, "h", long_options, &option_index)) != -1)
    {
        switch (c)
        {
        case 0:
            std::cout << "option " << long_options[option_index].name;
            if (optarg)
            {
                std::cout << " with arg " << optarg;
            }
            std::cout << std::endl;
            break;
        case 'h':
            usage();
            return 0;
        }
    }

    return 0;
}
