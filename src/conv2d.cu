#include <stdio.h>

#include "image_io.cu.h"
#include "conv2d.cu.h"

/*
struct UChar2D {
    int width;
    int height;
    unsigned char *data;
};

struct Double2D {
    int width;
    int height;
    double *data;
};
*/

/**
 * Perform 2d convolution with "valid" padding. This means no padding,
 * is performed and extra bits on the higher index dimensions of each
 * axis are dropped.
 *
 * Example 1:
 * 0 1 2 3 4 5 6 7 8
 * --1--
 *   --2--
 *     --3--
 *       --4--
 *         --5-- 
 *           --6--
 *             --7--
 * (9 - 3) / (int)1 + 1 = 6 / (int)1 + 1 = 7
 *
 * Example 2:
 * 0 1 2 3 4 5 6 7 8 9 a b c d e
 * --1--
 *       --2--
 *             --3--
 *                   --4--
 *                         --5--
 *
 * (15 - 3) / (int)3 + 1 = 12  / (int)3 + 1 = 5 
 *
 * @param wid_stride The horizontal stride across the width axis.
 * @param hgt_stride The vertical stride across the height axis.
 */
__host__
void conv_2d(
    Double2D *kernel, int wid_stride, int hgt_stride, PadType padding,
    Double2D *in, Double2D *out
) {
    if (padding == PAD_VALID) {
        out->width = (in->width - kernel->width) / (int)wid_stride + 1;
        out->height = (in->height - kernel->height) / (int)hgt_stride + 1;
    }
    out->data = (double*)malloc(out->width * out->height * sizeof(double));
    //img_to_ascii(input->data, 64, 64, 1);
    return;
}

__host__
void conv_2d_input(
    Double2D *kernel, int wid_stride, int hgt_stride, PadType padding,
    UChar2D *in, Double2D *out
) {
    if (padding == PAD_VALID) {
        out->width = calc_dims_pad_valid(
            in->width, kernel->width, wid_stride
        );
        out->height = calc_dims_pad_valid(
            in->height, kernel->height, hgt_stride
        );
    }
    out->data = (double*)malloc(out->width * out->height * sizeof(double));
    if (padding == PAD_VALID) {

    }
    return;
}

__host__
int calc_dims_pad_valid(int in_x, int kernel_x, int stride_x) {
    return (in_x - kernel_x) / (int)stride_x + 1;
}
