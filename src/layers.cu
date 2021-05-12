#include <stdio.h>

#include "image_io.cu.h"
#include "layers.cu.h"
#include "math.cu.h"

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
 * axis are dropped. See test cases for examples
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
    //img_to_ascii(input.data, 64, 64, 1);
    return;
}

/**
 * Same as the conv_2d() function, but takes unsigned char input data.
 */
__host__
void conv_2d_input(
    Double2D *kernel, int wid_stride, int hgt_stride, PadType padding,
    UChar2D *in, Double2D *out
) {
    // Ensure valid padding size
    if (wid_stride <= 0) {
        wid_stride = 1;
    }
    if (hgt_stride <= 0) {
        hgt_stride = 1;
    }

    // Calculate output dimensions based on input dimensions, kernel dimensions, and stride
    if (padding == PAD_VALID) {
        out->width = calc_dims_pad_valid(
            in->width, kernel->width, wid_stride
        );
        out->height = calc_dims_pad_valid(
            in->height, kernel->height, hgt_stride
        );
    }

    // Allocate memory (first free if memory exists)
    if (out->data != NULL) {
        free(out->data);
    }
    out->data = (double*)malloc(out->width * out->height * sizeof(double));

    // Perform element-wise multiply then sum for each output cell
    if (padding == PAD_VALID) {
        for (int outrow = 0; outrow < out->height; ++outrow) {
            for (int outcol = 0; outcol < out->width; ++outcol) {
                for (int kerrow = 0; kerrow < kernel->height; ++kerrow) {
                    for (int kercol = 0; kercol < kernel->width; ++kercol) {
                        int i_out = outrow * out->width + outcol;
                        int i_ker = kerrow * kernel->width + kercol;
                        int in_row = outrow * hgt_stride + kerrow;
                        int in_col = outcol * wid_stride + kercol;
                        int i_in = in_row * in->width + in_col;
                        //printf("Adding from input[%d][%d] to output[%d][%d]\n",
                        //in_row, in_col, outrow, outcol);
                        out->data[i_out] += 
                            kernel->data[i_ker] * (double)in->data[i_in];
                    }
                }
            }
        }
    }
}

__host__
int calc_dims_pad_valid(int in_x, int kernel_x, int stride_x) {
    return (in_x - kernel_x) / (int)stride_x + 1;
}

__host__
void Conv2D_forward(
    float *outs, int o_rows, int o_cols, // output
    float *vals, // the values before activation; same dimensions as outs
    float *do_dv, // change in outs w.r.t. vals; same sims as outs
    float *ins, int i_rows, int i_cols, // input
    float *weights, int w_rows, int w_cols, // the filters
    int s_rows, int s_cols, int filters, // stride and filters
    int activation // 0 = sigmoid, 1 = ReLU
) {
    for (int filter = 0; filter < filters; ++filter) {
        for (int o_row = 0; o_row < o_rows; ++o_row) {
            for (int o_col = 0; o_col < o_cols; ++o_col) {
                int out_index = o_cols * (filter * o_rows + o_row) + o_col;
                vals[out_index] = 0;
                for (int k_row = 0; k_row < w_rows; ++k_row) {
                    for (int k_col = 0; k_col < w_cols; ++k_col) {
                        // The input row is the output row * stride in the row axis
                        // plus the row in the kernel we're currently in.
                        int i_row = o_row * s_rows + k_row;
                        // Same formula applies for columns
                        int i_col = o_col * s_cols + k_col;
                        int in_index = i_row * i_rows + i_col;
                        // each activation map in vals has o_rows * o_cols elements
                        // each row has o_cols elements
                        int weight_index = w_cols * (filter * w_rows + k_row) + k_col;
                        vals[out_index] += weights[weight_index] * ins[in_index];
                    }
                }
            }
        }
        int act_map_offset = filter * o_rows * o_cols;
        // Calculate the activated outputs and the local gradient between vals and outs
        // vec_sigmoid_and_deriv(float *out, float *out_deriv, float *in, int len);
        if (activation == 0) {
            vec_sigmoid_and_deriv(
                outs + act_map_offset, do_dv + act_map_offset,
                vals + act_map_offset, o_rows * o_cols
            );
        } else if (activation == 1) {
            vec_relu_and_deriv(
                outs + act_map_offset, do_dv + act_map_offset,
                vals + act_map_offset, o_rows * o_cols
            );
        }
    }
}
