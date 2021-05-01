// Standard
#include <stdio.h>

// Third-party
#include <cuda.h>

// Local
#include "placeholder.cu.h"
#include "image_io.cu.h"

int main(int argc, char *argv[]) {
    printf("Hello from program main!\n");
    printf("Testing included cuda header: %d.\n", (int)placeholder());

    int status;
    unsigned char *data = load_img_carefully(
        "data/chinese_mnist/data/input_1_1_1.jpg",
        64, 64, 1, &status
    );
    img_to_ascii(data, 64, 64, 1);
    return 0;
}
