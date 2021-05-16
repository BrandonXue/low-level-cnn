#include <stdio.h>

#include "cn_mnist_utils.cu.h"
#include "math.cu.h"

/**
 * Converts a target integer 0 - 10, 100, 1K, 10K, 100M
 * to a one-hot vector.
 */
__host__
void cn_mnist_target_to_vec(float *vec, int target) {
    for (int i = 0; i < 15; ++i) {
        vec[i] = 0;
    }
    switch(target) {
        // integers 0 - 10 use their index as their slot in the tensor
        case 0: case 1: case 2: case 3:
        case 4: case 5: case 6: case 7:
        case 8: case 9: case 10:
            vec[target] = 1;
            break;
        case 100:
            vec[11] = 1;
            break;
        case 1000:
            vec[12] = 1;
            break;
        case 10000:
            vec[13] = 1;
            break;
        case 100000000:
            vec[14] = 1;
            break;
    }
}

const char* pred_suffix(float probability, bool is_max) {
    if (probability > 0.45) {
        if (is_max)
            return "*** (max)";
        else 
            return "***";
    } else if (probability > 0.30) {
        if (is_max)
            return "** (max)";
        else
            return "**";
    } else if (probability > 0.15) {
        if (is_max)
            return "* (max)";
        else
            return "*";
    }
    return "";
}

/**
 *
 */
__host__
void print_predictions(float *predictions) {
    int arg_max = argmax(predictions, 15);

    printf("Chinese Char\tLabel\t\tProbability:\n");
    printf("零\t\t0\t\t%0.3f %s\n",        round_digits(predictions[0], 3),
        pred_suffix(predictions[0], arg_max == 0));
    printf("一\t\t1\t\t%0.3f %s\n",        round_digits(predictions[1], 3),
        pred_suffix(predictions[1], arg_max == 1));
    printf("二\t\t2\t\t%0.3f %s\n",        round_digits(predictions[2], 3),
        pred_suffix(predictions[2], arg_max == 2));
    printf("三\t\t3\t\t%0.3f %s\n",        round_digits(predictions[3], 3),
        pred_suffix(predictions[3], arg_max == 3));
    printf("四\t\t4\t\t%0.3f %s\n",        round_digits(predictions[4], 3),
        pred_suffix(predictions[4], arg_max == 4));
    printf("五\t\t5\t\t%0.3f %s\n",        round_digits(predictions[5], 3),
        pred_suffix(predictions[5], arg_max == 5));
    printf("六\t\t6\t\t%0.3f %s\n",        round_digits(predictions[6], 3),
        pred_suffix(predictions[6], arg_max == 6));
    printf("七\t\t7\t\t%0.3f %s\n",        round_digits(predictions[7], 3),
        pred_suffix(predictions[7], arg_max == 7));
    printf("八\t\t8\t\t%0.3f %s\n",        round_digits(predictions[8], 3),
        pred_suffix(predictions[8], arg_max == 8));
    printf("九\t\t9\t\t%0.3f %s\n",        round_digits(predictions[9], 3),
        pred_suffix(predictions[9], arg_max == 9));
    printf("十\t\t10\t\t%0.3f %s\n",       round_digits(predictions[10], 3),
        pred_suffix(predictions[10], arg_max == 10));
    printf("百\t\t100\t\t%0.3f %s\n",      round_digits(predictions[11], 3),
        pred_suffix(predictions[11], arg_max == 11));
    printf("千\t\t1,000\t\t%0.3f %s\n",    round_digits(predictions[12], 3),
        pred_suffix(predictions[12], arg_max == 12));
    printf("万\t\t10,000\t\t%0.3f %s\n",   round_digits(predictions[13], 3),
        pred_suffix(predictions[13], arg_max == 13));
    printf("亿\t\t100,000,000\t%0.3f %s\n",round_digits(predictions[14], 3),
        pred_suffix(predictions[14], arg_max == 14));
}
