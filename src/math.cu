#include <float.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>

#include "math.cu.h"

/**
 * Pretty print a vector. Useful for debug.
 */
__host__
void print_vec(float *vec, int len, int precision) {
    if (precision < 0) {
        precision = 0;
    }

    printf("[");
    for (int i = 0; i < len; ++i) {
        printf("%0.*f", precision, vec[i]);
        if (i + 1 < len) {
            printf("  ");
        }
    }
    printf("]\n");
}


/**
 * Find the index within the vector of the highest-valued float.
 */
__host__
int argmax(float *vec, int len) {
    int arg_max = 0;
    float max = vec[0];
    for (int i = 0; i < len; ++i) {
        if (vec[i] > max) {
            max = vec[i];
            arg_max = i;
        }
    }
    return arg_max;
}

__host__
float rand_float(float min, float max) {
    double range = max - min;
    double zero_to_one = (double)rand() / (double)(RAND_MAX);
    return float(zero_to_one * range + (double)min);
}

__host__
void random_init(float *arr, int len, float min, float max) {
    for (int i = 0; i < len; ++i ) {
        arr[i] = rand_float(min, max);
    }
}

__host__
double round_digits(double x, int digits) {
    double mul_factor = pow(10, digits);
    return round(x * mul_factor) / mul_factor;
}

__host__
bool fuzzy_equals_digits(double a, double b, int digits) {
    return round_digits(a, digits) == round_digits(b, digits);
}


__global__
void kernel_vec_vec_multiply(float *out, float *a, float *b, int len) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < len) {
        out[idx] = a[idx] * b[idx];
    }
}

/**
 * Element-wise multiplication of two vectors.
 * Output does not need zero-initialization.
 */
__host__
void vec_vec_multiply(float *out, float *a, float*b, int len) {
    int blocks = ceil(len / 32.0);
    kernel_vec_vec_multiply<<<blocks, 32>>>(out, a, b, len);

    /*HOST VERSION
    for (int i = 0; i < len; ++i) {
        out[i] = a[i] * b[i];
    }
    */
}

/**
 * Dot product between matrix A and column vector.
 */
__global__
void kernel_mat_vec_dot(float *out, float *A, int M, int N, float *v) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    // One thread per row of the matrix
    if (idx < M) {
        out[idx] = 0;
        for (int mat_col = 0; mat_col < N; ++mat_col) {
            out[idx] += A[idx * N + mat_col] * v[mat_col];
        }
    }
}

__host__
void mat_vec_dot(float *out, float *A, int M, int N, float *v) {
    int blocks = ceil(M / 32.0);
    kernel_mat_vec_dot<<<blocks, 32>>>(out, A, M, N, v);
}


__global__
void kernel_zero(float *v, int len) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < len) {
        v[idx] = 0;
    }
}

#define COLS_PER_BLOCK 32
#define SEGMENT_SIZE 32

// out needs to be zero-ed before this kernel launch.
// this dot product should only be used for a number of rows that
// is perfectly divisible by "segment size"
__global__
void kernel_vmm_full(float *out, float *v, float *A, int N) {
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int seg = blockIdx.y * blockDim.y + threadIdx.y;

    // each column is handled by multiple full segments
    if (col < N) {
        float loc_sum = 0;

        // "By default, the compiler unrolls small loops with a known trip count."
#pragma unroll
        for (int seg_row = 0; seg_row < SEGMENT_SIZE; ++seg_row) {
            int mat_row = seg * SEGMENT_SIZE + seg_row;
            loc_sum += v[mat_row] * A[mat_row * N + col];
        }

        // Add to the output vector at the index of this thread's matrix column
        // Must be atomic because of race conditions between segments.
        atomicAdd(out + col, loc_sum);
    }
}

__global__
void kernel_vmm_partial(float *out, float *offset_v, float *offset_A, int rows, int N) {
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (col < N) {
        for (int seg_row = 0; seg_row < rows; ++seg_row) {
            out[col] += offset_v[seg_row] * offset_A[seg_row * N + col];
        }
    }
}

/**
 * Dot product between row vector v and matrix A.
 * Output does not need zero-initialization.
 */
__host__
void vec_mat_dot(float *out, float *v, float *A, int M, int N) {

    int gridX = ceil((float)N / (float)COLS_PER_BLOCK);

    // Zero the output vector first.
    kernel_zero<<<gridX, COLS_PER_BLOCK>>>(out, N);

    // Handle full vertical segments of the matrix (if any)
    int full_segments = M / SEGMENT_SIZE; // int division
    if (full_segments > 0) {
        dim3 gridLaunch(gridX, full_segments);
        dim3 blockLaunch(COLS_PER_BLOCK, 1);
        kernel_vmm_full<<<gridLaunch, blockLaunch>>>(out, v, A, N);
    }

    // Handle the remaining partial segment (if any)
    int remaining_rows = M % SEGMENT_SIZE;
    if (remaining_rows != 0) {
        kernel_vmm_partial<<<gridX, COLS_PER_BLOCK>>>
            (out,
             v + full_segments * SEGMENT_SIZE,
             A + full_segments * SEGMENT_SIZE * N,
             remaining_rows, N);
    }

    /* HOST VERSION
    for (int mat_col = 0; mat_col < N; ++mat_col) {
        out[mat_col] = 0;
        for (int mat_row = 0; mat_row < M; ++mat_row) {
            out[mat_col] += v[mat_row] * A[mat_row * N + mat_col];
        }
    }
    */
}

__global__
void kernel_vec_vec_outer(float *out, float *a, float *b, int a_len, int b_len) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    int col = blockIdx.y * blockDim.y + threadIdx.y;
    if (row < a_len && col < b_len) {
        out[row * b_len + col] = a[row] * b[col];
    }
}

/**
 * Outer product of two vectors. The output has rows of b, each multiplied by
 * a factor of an element of a.    a = [a1  a2  a3 ... aM]    b = [b1  b2  b3 ... bN]
 * out = [a1b1 a1b2 a1b3 ... a1bN
 *        a2b1 a2b2 a2b3 ... a2bN
           .    .    .    .    .
          aMb1 aMb2 aMb3 ... aMbN]
 */
__host__
void vec_vec_outer(float *out, float *a, float *b, int a_len, int b_len) {

    int blockX = ceil(a_len / 32.0);
    int blockY = ceil(b_len / 32.0);
    dim3 gridLaunch(blockX, blockY);
    dim3 blockLaunch(32, 32);
    kernel_vec_vec_outer<<<gridLaunch, blockLaunch>>>(out, a, b, a_len, b_len);
    /* HOST VERSION
    for (int row = 0; row < a_len; ++row) {
        for (int col = 0; col < b_len; ++col) {
            out[row * b_len + col] = a[row] * b[col];
        }
    }
    */
}

/**
 * Normalize an image with unsigned char (1 byte) to floating point (4 bytes),
 * with min 0 and range 1.
 */
__host__
void normalize_ctf(float *res, unsigned char *arr, int len) {
    int max = 0;
    int min = 255;
    for (int i = 0; i < len; ++i) {
        if ((int)arr[i] > max) {
            max = (int)arr[i];
        }
        if ((int)arr[i] < min) {
            min = (int)arr[i];
        }
    }
    if (max <= min) {
        printf("Warning: normalize had invalid denominator.\n");
    }
    for (int i = 0; i < len; ++i) {
        res[i] = ((float)arr[i] - min) / (float)(max - min);
    }
}


__global__
void kernel_vec_sigmoid_and_deriv(float *out, float *out_deriv, float *in, int len) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < len) {
        if (in[idx] < 0) {
            float exp_x = exp(in[idx]);
            out[idx] = exp_x / (1 + exp_x);
        } else {
            out[idx] = 1 / (1 + exp(-in[idx]));
        }
        out_deriv[idx] = out[idx] * (1 - out[idx]);
    }
}

/**
 * Vectorized sigmoid and its first derivative.
 */
__host__
void vec_sigmoid_and_deriv(float *out, float *out_deriv, float *in, int len) {
    int blocks = ceil(len / 32.0);
    kernel_vec_sigmoid_and_deriv<<<blocks, 32>>>(out, out_deriv, in, len);

    /* HOST VERSION
    for (int i = 0; i < len; ++i) {
        if (in[i] < 0) {
            float exp_x = exp(in[i]);
            out[i] = exp_x / (1 + exp_x);
        } else {
            out[i] = 1 / (1 + exp(-in[i]));
        }
        //out[i] = 1 / (1 + exp(-in[i]));
        out_deriv[i] = out[i] * (1 - out[i]);
    }
    */
}

__global__
void kernel_vec_relu_and_deriv(float *out, float *out_deriv, float *in, int len) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < len) {
        if (in[idx] > 0) {
            out[idx] = in[idx];
            out_deriv[idx] = 1;
        } else {
            out[idx] = 0;
            out_deriv[idx] = 0;
        }
    }
}

/**
 * Vectorized ReLU and its first derivative.
 */
__host__
void vec_relu_and_deriv(float *out, float *out_deriv, float *in, int len) {
    int blocks = ceil(len / 32.0);
    kernel_vec_relu_and_deriv<<<blocks, 32>>>(out, out_deriv, in, len);
    /* HOST VERSION
    for (int i = 0; i < len; ++i) {
        if (in[i] > 0) {
            out[i] = in[i];
            out_deriv[i] = 1;
        } else {
            out[i] = 0;
            out_deriv[i] = 0;
        }
    }
    */
}

// I am using the exact fuzz factor used by Keras and TF
static const float EPSILON_FUZZ = 0.0000001;

/**
 * Calculate the mean squared error.
 * @param y_true Pointer to an array of actual y values.
 * @param y_pred Pointer to an array of observed y values.
 * @len The length of both y_true and y_pred.
 */
__host__
float  mse(float *y_true, float *y_pred, int len) {
    float sse = 0;
    for (int i = 0; i < len; ++i) {
        // Error
        float temp  = y_true[i] - y_pred[i];
        // Squared error
        temp *= temp;
        // Sum up the squared error (TODO: overflow underflow protection, FMA)
        sse += temp;
    }
    // Mean squared error
    return sse / len;
}


/**
 * Categorical Cross Entropy Loss. Assumes that the inputs
 * have not gone through any activation function yet, and applies
 * the softmax activation as part of this function call.
 * Also calculates the change in loss wrt the values before activation.
 */
__host__
void cat_cross_entropy(
    int len, float *y_true, float *vals, float *outs, float *pdL_pdval, float *loss
) {
    // softmax(x) = softmax(x - C)
    // For numeric stability, we will subtract the max from each element.
    // First find the max of the un-activated values.
    float vals_max = vals[0];
    for (int i = 0; i < len; ++i) {
        if (vals[i] > vals_max) {
            vals_max = vals[i];
        }
        //printf("in softmax val[%d] = %f\n", i, vals[i]);
    }

    // Find the numerator and the denominator
    // Numerator is e^(x - max_x) denominator is the summation of all the numerators
    float sum_exp = 0;
    for (int i = 0; i < len; ++i) {
        outs[i] = exp(vals[i] - vals_max);
        sum_exp += outs[i];
    }
    // In the unlikely event the denom is zero, add fuzz factor
    if (sum_exp == 0) {
        sum_exp = EPSILON_FUZZ;
    }
    // Finish calculating the softmaxes by dividing each element by the denominator
    for (int i = 0; i < len; ++i) {
        outs[i] = outs[i] / sum_exp;
    }

    // Calculate the Cross Entropy Loss, -Σ (y_true * log(softmax(vals))
    *loss = 0;
    for (int i = 0; i < len; ++i) {
        *loss += y_true[i] * log(outs[i]);
    }
    // There is a negative one coefficient.
    *loss *= -1;

    // Calculate the gradient
    // For each element, the derivative is softmax(x)_i  -  y_true_i
    for (int i = 0; i < len; ++i) {
        pdL_pdval[i] = outs[i] - y_true[i];
    }
}

__host__
float convolution_2d(
    int out_row, int out_col,
    float *image, int i_rows, int i_cols,
    float *kernel, int k_rows, int k_cols,
    int stride_rows, int stride_cols
) {
    float out = 0;
    for (int k_row = 0; k_row < k_rows; ++k_row) {
        for (int k_col = 0; k_col < k_cols; ++k_col) {
            int i_row = out_row * stride_rows + k_row;
            int i_col = out_col * stride_cols + k_col;
            out += kernel[k_row * k_cols + k_col] * image[i_row * i_cols + i_col];
        }
    }
    return out;
}

__global__
void kernel_all_convolution_2d(
    float *result, int res_rows, int res_cols,
    float *image, int i_rows, int i_cols,
    float *kernel, int k_rows, int k_cols,
    int stride_rows, int stride_cols
) {
    // Find which output cell this thread is making calculations for. 
    int res_row = blockIdx.x * blockDim.x + threadIdx.x;
    int res_col = blockIdx.y * blockDim.y + threadIdx.y;

    // Only perform work for thread indices within the output bounds
    if (res_row < res_rows && res_col < res_cols) {

        // offset this thread's references to the kernel and result depending on filter index
        kernel = kernel + (blockIdx.z * k_rows * k_cols);
        result = result + (blockIdx.z * res_rows * res_cols);

        int i_row_offset = res_row * stride_rows;
        int i_col_offset = res_col * stride_cols;

        result[res_row * res_cols + res_col] = 0; // Clear cell first

        // Sum up the element-wise products.
        for (int k_row = 0; k_row < k_rows; ++k_row) {
            for (int k_col = 0; k_col < k_cols; ++k_col) {
                result[res_row * res_cols + res_col] +=
                    kernel[k_row * k_cols + k_col] * 
                    image[(i_row_offset + k_row) * i_cols + (i_col_offset + k_col)];
            }
        }
    }
}

__host__
void all_convolution_2d(
    float *result,
    float *image, int i_rows, int i_cols,
    float *weights, int filters, int k_rows, int k_cols,
    int stride_rows, int stride_cols
) {
    int res_rows = ((i_rows - k_rows) / stride_rows) + 1;
    int res_cols = ((i_cols - k_cols) / stride_cols) + 1;

    int gridX = ceil(res_rows / 32.0);
    int gridY = ceil(res_cols / 32.0);

    dim3 gridLaunch(gridX, gridY, filters);
    dim3 blockLaunch(32, 32);

    //printf("Launching for conv2d forward.\n");
    kernel_all_convolution_2d<<<gridLaunch, blockLaunch>>>(
        result, res_rows, res_cols,
        image, i_rows, i_cols,
        weights, k_rows, k_cols,
        stride_rows, stride_cols
    );

    /* HOST VERSION
    int res_rows = ((i_rows - k_rows) / stride_rows) + 1;
    int res_cols = ((i_cols - k_cols) / stride_cols) + 1;
    for (int filter = 0; filter < filters; ++filter) {
        // Each filter is its own kernel. Therefore, need to offset weights
        // depending on which kernel we're using.
        float *curr_kernel = weights + (filter * k_rows * k_cols);
        for (int res_row = 0; res_row < res_rows; ++res_row) {
            for (int res_col = 0; res_col < res_cols; ++res_col) {
                // activation map offset           + row offset           + col offset
                // (filters * res_rows * res_cols) + (res_row * res_cols) + res_col
                int res_index = res_cols * (filter * res_rows + res_row) + res_col; 
                result[res_index] = convolution_2d(
                    res_row, res_col,
                    image, i_rows, i_cols,
                    curr_kernel, k_rows, k_cols,
                    stride_rows, stride_cols);
                //printf("result[%d]=%f\n", res_index, result[res_index]);
            }
        }
    }*/
}

// This is for the host version that does not use GPU.
// dilation-from-stride convolute 2d
// used for backprop to calculate gradients for params
__host__
float back_convolution_2d(
    int out_row, int out_col,
    float *image, int i_rows, int i_cols,
    float *kernel, int k_rows, int k_cols,
    int stride_rows, int stride_cols
){
    float out = 0;
    for (int k_row = 0; k_row < k_rows; ++k_row) {
        for (int k_col = 0; k_col < k_cols; ++k_col) {
            int i_row = out_row + stride_rows * k_row;
            int i_col = out_col + stride_cols * k_col;
            out += kernel[k_row * k_cols + k_col] * image[i_row * i_cols + i_col];  
        }
    }
    return out;
}

__global__
void kernel_all_back_convolution_2d(
    float *result, int res_rows, int res_cols,
    float *image, int i_rows, int i_cols,
    float *kernel, int k_rows, int k_cols,
    int stride_rows, int stride_cols
) {
    // Find which output cell this thread is making calculations for.
    int res_row = blockIdx.x * blockDim.x + threadIdx.x;
    int res_col = blockIdx.y * blockDim.y + threadIdx.y;
    
    // Only perform work for thread indices within the output bounds
    if (res_row < res_rows && res_col < res_cols) {
        
        // offset this thread's references to the kernel and result depending on filter index
        kernel = kernel + (blockIdx.z * k_rows * k_cols);
        result = result + (blockIdx.z * res_rows * res_cols);

        result[res_row * res_cols + res_col] = 0; // Clear cell first

        // Sum up the element-wise products.
        for (int k_row = 0; k_row < k_rows; ++k_row) {
            for (int k_col = 0; k_col < k_cols; ++k_col) {
                result[res_row * res_cols + res_col] +=
                    kernel[k_row * k_cols + k_col] *
                    image[(res_row + stride_rows * k_row) * i_cols + (res_col + stride_cols * k_col)];
            }
        }
    }
}

// all dilation-from-stride convolute 2d
//used in backprop to calculate gradients for params
__host__
void all_back_convolution_2d(
    float *result, int res_rows, int res_cols,
    float *image, int i_rows, int i_cols,
    float *glob_grads, int filters, int g_rows, int g_cols,
    int stride_rows, int stride_cols
) {
    int gridX = ceil(res_rows / 32.0);
    int gridY = ceil(res_cols / 32.0);

    dim3 gridLaunch(gridX, gridY, filters);
    dim3 blockLaunch(32, 32);

    //printf("Launching for conv2d backprop.\n");
    kernel_all_back_convolution_2d<<<gridLaunch, blockLaunch>>>(
        result, res_rows, res_cols,
        image, i_rows, i_cols,
        glob_grads, g_rows, g_cols,
        stride_rows, stride_cols
    );

    /* HOST VERSION
    for (int filter = 0; filter < filters; ++filter) {
        // glob_grads is the change in loss w.r.t. pre-activation values.
        // each filter produces one activation map, which has its own
        // values for glob_grads.
        // Therefore, need to offset glob_grads based on the index
        // of the filter that produced it
        float *curr_glob_grads = glob_grads + (filter * g_rows * g_cols);
        for (int res_row = 0; res_row < res_rows; ++res_row) {
            for (int res_col = 0; res_col < res_cols; ++res_col) {
                // activation map offset           + row offset           + col offset
                // (filters * res_rows * res_cols) + (res_row * res_cols) + res_col
                int res_index = res_cols * (filter * res_rows + res_row) + res_col;
                result[res_index] = back_convolution_2d(
                    res_row, res_col,
                    image, i_rows, i_cols,
                    curr_glob_grads, g_rows, g_cols,
                    stride_rows, stride_cols);
            }
        }
    }*/
}

__host__
void conv_2d_dilate(
    float *result,
    float *A, int A_rows, int A_cols,
    float *B, int B_rows, int B_cols,
    int B_row_dil, int B_col_dil
);
