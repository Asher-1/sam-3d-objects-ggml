#include "moge_inference.hpp"

#include <cstdio>
#include <string>

int main() {
    std::string error;
    int width = 0;
    int height = 0;
    if (!sam3d::moge_resized_dimensions(6720, 4480, 2500, width, height, error)) {
        std::fprintf(stderr, "MoGe target-dimension calculation failed: %s\n", error.c_str());
        return 1;
    }
    if (width != 857 || height != 571) {
        std::fprintf(stderr, "unexpected MoGe target dimensions: %dx%d\n", width, height);
        return 1;
    }
    if (sam3d::moge_resized_dimensions(0, 10, 2500, width, height, error)) {
        std::fprintf(stderr, "MoGe accepted an invalid input dimension\n");
        return 1;
    }
    return 0;
}
