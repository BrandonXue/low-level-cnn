#include "cn_mnist_utils.cu.h"

TEST_CASE( "CN-MNIST integer label to one-hot vector", "[utils]" ) {
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
