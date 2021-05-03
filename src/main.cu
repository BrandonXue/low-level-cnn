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
    unsigned char img_in1[36] = {
        1, 3, 3, 2, 1, 1,
        1, 2, 2, 1, 1, 1,
        0, 1, 2, 1, 1, 0,
        0, 1, 1, 1, 0, 1,
        0, 0, 0, 1, 0, 0,
        1, 0, 1, 1, 1, 0
    };
    UChar2D conv_in1 = {.width = 6, .height = 6, .data = img_in1};

    double kern1[9] = {
         2,  2,  2,
         0,  0,  0,
        -2, -2, -2
    };
    Double2D kernel1 = {.width = 3, .height = 3, .data = kern1};
    
    // Valid padding with stride of 2 in both axes
    // expect last row and col to be ignored, so output 2x2
    double expected1[4] = {
        8, 4,
        6, 6
    };

    Double2D output1 = {.width = 0, .height = 0, .data = NULL};
    conv_2d_input(&kernel1, 2, 2, PAD_VALID, &conv_in1, &output1);
    printf("%lf %lf\n%lf %lf\n",
           output1.data[0], output1.data[1],
           output1.data[2], output1.data[3]
    );
    
    unsigned char img2[16] = {
        1, 3, 3, 2,
        1, 2, 2, 1,
        0, 1, 2, 1,
        0, 1, 1, 1
    };
    UChar2D in2 = {.width = 4, .height = 4, .data = img2};

    double ker2[16] = {
         2,  2,  2,
         0,  0,  0,
        -2, -2, -2
    };
    Double2D kernel2 = {.width = 3, .height = 3, .data = ker2};

    // Valid padding with stride of 1 in both axes
    // output 2x2
    double expected2[4] = {
        8, 8,
        6, 4
    };

    Double2D out2 = {.width = 0, .height = 0, .data = NULL};
    conv_2d_input(&kernel2, 1, 1, PAD_VALID, &in2, &out2);
    printf("%lf %lf\n%lf %lf\n",
           out2.data[0], out2.data[1],
           out2.data[2], out2.data[3]
    );
    return 0;
}
