#include <stdio.h>

#include "image_io.cu.h"
#include "nn_layers.cu.h"
#include "math.cu.h"

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

__host__
void Conv2D_backward(
    int o_rows, int o_cols, // output rows and cols, same dims for pdL_pdouts, pdL_pdvals, douts_dvals
    float *pdL_pdouts, // global gradient for the outs of this layer
    float *douts_dvals, // change in outs w.r.t. vals; same dims as outs
    float *pdL_pdvals, // gloabl gradient for the vals of this layer
    float *pdL_pdouts_pred, // global gradient for the outputs of previous layer
    float *ins, int i_rows, int i_cols, // input 
    float *weights, int w_rows, int w_cols, // the filters
    float *grads, // the gradients
    int s_rows, int s_cols, int filters // stride and filters
) {
    // Using the global gradient of the outs and the local
    // gradient between outs and vals, we can calculate the global
    // gradient for the vals by multiplying element-wise.
    vec_vec_multiply(pdL_pdvals, pdL_pdouts, douts_dvals, filters * o_rows * o_cols);
    //for (int i = 0; i < filters * o_rows * o_cols; ++i) {
    //    printf("index %d, pdL_pdouts=%f, douts_dvals=%f, pdL_pdvals=%f\n", i, pdL_pdouts[i], douts_dvals[i], pdL_pdvals[i]);
    //}
    // calculate gradients, this is a shortcut to convolute a dilated matrix
    // on another matrix without having to actually perform dilation.
    for (int filter = 0; filter < filters; ++filter) {
        // lg = local gradient. gg = global gradient
        for (int lg_row = 0; lg_row < w_rows; ++lg_row) {
            for (int lg_col = 0; lg_col < w_cols; ++lg_col) {
                int lg_index = w_cols * (filter * w_rows + lg_row) + lg_col;
                grads[lg_index] = 0;
                for (int gg_row = 0; gg_row < o_rows; ++gg_row) {
                    for (int gg_col = 0; gg_col < o_cols; ++gg_col) {
                        int in_row = lg_row + s_rows * gg_row;
                        int in_col = lg_col + s_cols * gg_col;
                        int in_index = in_row * i_cols + in_col;
                        int gg_index = o_cols * (filter * o_rows + gg_row) + gg_col;
                        grads[lg_index] += pdL_pdvals[gg_index] * ins[in_index];
                        //printf("in_index %d, in = %f\n", in_index, ins[in_index]);
                        //if (pdL_pdvals[gg_index] != 0) {
                        //    printf("pdL_pdvals[%d] = %f\n", gg_index, pdL_pdvals[gg_index]);
                        //}
                    }
                }
                //printf("in loop grads[%d]=%f\n", lg_index, grads[lg_index]); 
            }
        }
    }
}

__host__
void Dense_forward(
    float *outs, int o_len, // output
    float *vals, // the values before activation; same lenth as output
    float *do_dv, // change in outs w.r.t. vals; same length as output
    float *ins, int i_len, // input
    float *weights, // the weights; same length as input
    int activation // 0 = sigmoid, 1 = ReLU, 2 = None
) {
    // debug
    //for (int i = 0; i < i_len; ++i) {
    //    printf("ins[%d]=%f", i, ins[i]);
    //}
    vec_mat_dot(vals, ins, weights, i_len, o_len);
    if (activation == 0) {
        vec_sigmoid_and_deriv(outs, do_dv, vals, o_len);
    } else if (activation == 1) {
        vec_relu_and_deriv(outs, do_dv, vals, o_len);
    } 
    // implied else, no activation results in empty do_dv and outs
} 

__host__
void Dense_backward(
    int o_len,
    float *pdL_pdouts, // global gradient for the outs of this layer
    float *douts_dvals, // change in outs w.r.t. vals; same dims as outs
    float *pdL_pdvals, // gloabl gradient for the vals of this layer
    float *pdL_pdouts_pred, // global gradient for the outputs of previous layer
    float *ins, int i_len, // input
    float *weights, // the dimensions should be i_len x o_len
    float *grads, // the gradients; same dims as weights
    int activation // 0 = sigmoid, 1 = ReLU, 2 = None
) {
    // For categorical cross-entropy, no activation is used.
    // The activation is calculated in the same function as the cross entropy,
    // because this skips the computation of the Jacobian.
    if (activation != 2) {
        // Compute global gradient to the values before activation
        vec_vec_multiply(pdL_pdvals, pdL_pdouts, douts_dvals, o_len); 

    }
    // Compute global gradient to the weights
    vec_vec_outer(grads, ins, pdL_pdvals, i_len, o_len);

    mat_vec_multiply_reduce_sum(pdL_pdouts_pred, weights, i_len, o_len, pdL_pdvals);
}

/**
 * Update parameters using stochastic gradient descent.
 */
__host__
void SGD_update_params(float alpha, float *weights, float *grads, int len) {
    for (int i = 0; i < len; ++i) {
        weights[i] -= alpha * grads[i];
    }
}
