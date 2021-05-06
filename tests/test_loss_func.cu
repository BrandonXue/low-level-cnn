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
    int N = 6;
    float y_true[N] = {0, 1, 0, 0, 0, 0};
    float y_pred[N] = {1.2, 3.4, 1.1, 2.4, 0.4, 0.5};
    REQUIRE(
        fuzzy_equals_digits(
            cat_cross_entropy(y_true, y_pred, N), -0.97345, 5
        )
    );
}
