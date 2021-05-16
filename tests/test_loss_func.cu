#include "catch.hpp"

#include "loss_func.cu.h"
#include "math.cu.h"

TEST_CASE( "mse works", "[loss_func]" ) {
    int N = 6;
    float y_true[N] = {12.2, 19.3, 94.2, 4.5, 8.39, 28.39};
    float y_pred[N] = {14.3, 20.3, 91.3, 4.9, 9.01, 25.93};
    REQUIRE( fuzzy_equals_digits(mse(y_true, y_pred, N), 3.402666666, 4));
}

TEST_CASE( "cross entropy works", "[loss_func]" ) {
    // cat_cross_entropy(int len, float *y_true, float *y_pred, float *pdL_pdval, float *loss)
    int N = 4;
    float y_true[N] = {0, 1, 0, 0};
    float vals[N] = {1.2, 3.4, 1.1, 2.4};
    float outs[N];
    float pdL_pdval[N];
    float loss;

    float expect_outs[N] = {0.07017559697, 0.63333571, 0.06349750597, 0.2329911871};

    // These answers were calculated by hand dotting the Jacobian and the ∂L/∂outs,
    // instead of softmax(vals) - y_true
    // if they end up the same, then the math should be correct
    float expect_pdL_pdval[N] = {0.07017322697, -0.3666519068, 0.06349536151, 0.2329833184};

    cat_cross_entropy(N, y_true, vals, outs, pdL_pdval, &loss);
    
    for (int i = 0; i < N; ++i) {
        REQUIRE( fuzzy_equals_digits(outs[i], expect_outs[i], 5) );
    }

    for (int i = 0; i < N; ++i) {
        REQUIRE( fuzzy_equals_digits(pdL_pdval[i], expect_pdL_pdval[i], 4) );
    }


}

TEST_CASE( "chinese mnist targeet to one-hot works", "[loss_func]" ) {
    float target_tensor[15];
    cn_mnist_target_to_vec(target_tensor, 10000);
    float expect1[15] = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0};
    for (int i = 0; i < 15; ++i) {
        REQUIRE( target_tensor[i] == expect1[i] );
    }
    cn_mnist_target_to_vec(target_tensor, 100000000);
    float expect2[15] = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1};
    for (int i = 0; i < 15; ++i) {
        REQUIRE( target_tensor[i] == expect2[i] );
    }
    cn_mnist_target_to_vec(target_tensor, 1000);
    float expect3[15] = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0};
    for (int i = 0; i < 15; ++i) {
        REQUIRE( target_tensor[i] == expect3[i] );
    }
    cn_mnist_target_to_vec(target_tensor, 100);
    float expect4[15] = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0};
    for (int i = 0; i < 15; ++i) {
        REQUIRE( target_tensor[i] == expect4[i] );
    }
    cn_mnist_target_to_vec(target_tensor, 5);
    float expect5[15] = {0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0};
    for (int i = 0; i < 15; ++i) {
        REQUIRE( target_tensor[i] == expect5[i] );
    }   
}
