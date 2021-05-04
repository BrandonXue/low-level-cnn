// Standard
#include <stdio.h>

// Third-party
#include <cuda.h>

// Local
#include "conv2d.cu.h"
#include "image_io.cu.h"

int main(int argc, char *argv[]) {
    /*int status;
    unsigned char *data = load_img_carefully(
        "data/chinese_mnist/data/input_1_1_1.jpg",
        64, 64, 1, &status
    );*/
    //set_locale_lc_all();
    //img_to_unicode(data, 64, 64, 1);
    /*double weights_k1[3] = {-479, 479, 999};
    Double2D kern1 = {.width = 5, .height = 5, .data = weights_k1};
    UChar2D img = {.width = 64, .height = 64, .data = data};
    Double2D out = {.width = 0, .height = 0};
    conv_2d_input(&kern1, 1, 1, PAD_VALID, &img, &out);
    printf("Output dimensions: %d, %d.\n", out.width, out.height);
    */
    InputLabel *inputs = load_chinese_mnist_info();
    /*for (int i = 0; i < 14999; ++i) {
        printf("Input file %s has label %d\n", inputs[i].input, inputs[i].label);
    }*/
    return 0;
}
