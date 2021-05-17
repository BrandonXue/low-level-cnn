#include <math.h>
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
    
    all_convolution_2d(
        vals, 
        ins, i_rows, i_cols,
        weights, filters, w_rows, w_cols,
        s_rows, s_cols
    );

    for (int filter = 0; filter < filters; ++filter) {
        int act_map_offset = filter * o_rows * o_cols;
        // Calculate the activated outputs and the local gradient between vals and outs
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
    

    all_back_convolution_2d(
        grads, w_rows, w_cols,
        ins, i_rows, i_cols,
        pdL_pdvals, filters, o_rows, o_cols,
        s_rows, s_cols
    );
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

    mat_vec_dot(pdL_pdouts_pred, weights, i_len, o_len, pdL_pdvals);
}

__global__
void kernel_SGD_update_params(float alpha, float *weights, float *grads, int len) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < len) {
        weights[idx] -= alpha * grads[idx];
    }
}

/**
 * Update parameters using stochastic gradient descent.
 */
__host__
void SGD_update_params(float alpha, float *weights, float *grads, int len) {
    int blocks = ceil(len / 32.0);
    kernel_SGD_update_params<<<blocks, 32>>>(alpha, weights, grads, len);
}
