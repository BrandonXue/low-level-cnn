#include "loss_func.cu.h"

/**
 * Calculate the mean squared error.
 * @param y_true Pointer to an array of actual y values.
 * @param y_pred Pointer to an array of observed y values.
 * @len The length of both y_true and y_pred.
 */
__host__
float mse(float *y_true, float *y_pred, int len) {
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
