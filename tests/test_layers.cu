#include "catch.hpp"

#include "layers.cu.h"
#include "math.cu.h"

TEST_CASE( "valid padding dimension calculation works", "[conv2d]" ) {
    /*
      0 1 2 3 4 5 6 7 8
      --1--
        --2--
          --3--
            --4--
              --5-- 
                --6--
                  --7--
      (9 - 3) / (int)1 + 1 = 6 / (int)1 + 1 = 7
     */
    REQUIRE( calc_dims_pad_valid(9, 3, 1) == 7 );

    /*
       0 1 2 3 4 5 6 7 8 9 a b c d e
       --1--
             --2--
                   --3--
                         --4--
                               --5--
       (15 - 3) / (int)3 + 1 = 12  / (int)3 + 1 = 5
     */
    REQUIRE( calc_dims_pad_valid(15, 3, 3) == 5 );

    /*
       00-04 02-06 04-08 06-10 08-12 10-14
       12-16 14-18 16-20 18-22 20-24 22-26
       24-28 26-30 28-32 30-34 32-36 34-38
       36-40 38-42 40-44 42-46 44-48 46-50
       48-52 50-54 52-56 54-58 56-60 58-62
     */
    REQUIRE( calc_dims_pad_valid(64, 5, 2) == 30 );
}


TEST_CASE( "conv 2d valid padding on larger input", "[conv2d]" ) {
    // Setup the tests and require that the setup is correct.

    // Setup the input "image"
    unsigned char img_dat[36] = {
        1, 3, 3, 2, 1, 1,
        1, 2, 2, 1, 1, 1,
        0, 1, 2, 1, 1, 0,
        0, 1, 1, 1, 0, 1,
        0, 0, 0, 1, 0, 0,
        1, 0, 1, 1, 1, 0
    };
    UChar2D input = {.width = 6, .height = 6, .data = img_dat};

    // Setup the kernel
    double krn_dat[9] = {
         2,  2,  2,
         0,  0,  0,
        -2, -2, -2
    };
    Double2D kernel = {.width = 3, .height = 3, .data = krn_dat};
    
    // Setup a struct to receive output
    Double2D output = {.width = 0, .height = 0, .data = NULL};
    REQUIRE( output.width == 0 );
    REQUIRE( output.height == 0 );
    REQUIRE( output.data == NULL );

    SECTION( "using stride of horiz: 2, vert: 2" ) {
        double expected[4] = {
            8, 4,
            6, 6
        };
        conv_2d_input(&kernel, 2, 2, PAD_VALID, &input, &output);
        REQUIRE( output.width == 2 );
        REQUIRE( output.height == 2 );
        REQUIRE( output.data[0] == expected[0] );
        REQUIRE( output.data[1] == expected[1] );
        REQUIRE( output.data[2] == expected[2] );
        REQUIRE( output.data[3] == expected[3] );
    }
}

TEST_CASE( "conv 2d valid padding on smaller input", "[conv2d]") {
    // Setup the tests and require that the setup is correct.

    // Setup the input "image"
    unsigned char img_dat[16] = {
        1, 3, 3, 2,
        1, 2, 2, 1,
        0, 1, 2, 1,
        0, 1, 1, 1
    };
    UChar2D input = {.width = 4, .height = 4, .data = img_dat};
    REQUIRE( input.width == 4 );
    REQUIRE( input.height == 4 );
    REQUIRE( input.data[7] == 1 );

    // Setup the kernel
    double krn_dat[16] = {
         2,  2,  2,
         0,  0,  0,
        -2, -2, -2
    };
    Double2D kernel = {.width = 3, .height = 3, .data = krn_dat};
    REQUIRE( kernel.width == 3 );
    REQUIRE( kernel.height == 3 );
    REQUIRE( kernel.data[7] == -2 );

    // Setup a struct to receive output
    Double2D output = {.width = 0, .height = 0, .data = NULL};
    REQUIRE( output.width == 0 );
    REQUIRE( output.height == 0 );
    REQUIRE( output.data == NULL );

    SECTION( "using stride of horiz: 1, vert: 1" ) {
        double expected[4] = {
            8, 8,
            6, 4
        };
        conv_2d_input(&kernel, 1, 1, PAD_VALID, &input, &output);
        REQUIRE( output.width == 2 );
        REQUIRE( output.height == 2 );
        REQUIRE( output.data[0] == expected[0] );
        REQUIRE( output.data[1] == expected[1] );
        REQUIRE( output.data[2] == expected[2] );
        REQUIRE( output.data[3] == expected[3] );
    }

    SECTION( "using stride of horiz: 1, vert: 2" ) {
        double expected[2] = { 8, 8 };
        conv_2d_input(&kernel, 1, 2, PAD_VALID, &input, &output);
        REQUIRE( output.width == 2 );
        REQUIRE( output.height == 1 );
        REQUIRE( output.data[0] == expected[0] );
        REQUIRE( output.data[1] == expected[1] );
    }
}

TEST_CASE( "convolution layer given this configuration", "[conv2d]" ) {
    int i_rows = 4, i_cols = 4;
    float ins[i_rows * i_cols] = {.3, .2, .3, .6,
                                  .2, .1, .5, .9,
                                  .1, .2, .7, 1.0,
                                   0, .2, .4, .8};
    int w_rows = 2, w_cols = 2;
    int s_rows = 2, s_cols = 2, filters = 2;
    float weights[filters * w_rows * w_cols] = { 1.0,  0.2,
                                                -0.2, -1.0,
    
                                                -0.3,  0.5,
                                                 0.6,  0.7};
    int o_rows = calc_dims_pad_valid(i_rows, w_rows, s_rows); // input 4 rows, kernel 2 rows, stride 2 rows
    int o_cols = calc_dims_pad_valid(i_cols, w_cols, s_cols); // input 4 cols, kernel 2 cols, stride 2 cols
    float outs[filters * o_rows * o_cols];
    float vals[filters * o_rows * o_cols];
    float do_dv[filters * o_rows * o_cols];

    Conv2D_forward(outs, o_rows, o_cols, vals, do_dv,
                   ins, i_rows, i_cols,
                   weights, w_rows, w_cols,
                   s_rows, s_cols, filters,
                   0); // 0 == sigmoid activation

    float expect_vals[filters * o_rows * o_cols] =
        { 0.2000000000, -0.5800000000,
         -0.0600000000,  0.0200000000,
        
          0.2000000000, 1.14000000000,
          0.2100000000, 1.09000000000};
    float expect_outs[filters *o_rows * o_cols] =
        {0.5498339973, 0.3589325937,
         0.4850044984, 0.5049998333,
        
         0.5498339973,  0.7576796390,
         0.5523079096,  0.7483817216};
    float expect_do_dv[filters * o_rows * o_cols] =
        {0.2475165727, 0.2300999869,
         0.2497751349, 0.2499750017,
        
         0.2475165727, 0.1836012036,
         0.2472638826, 0.1883065204};

    for (int i = 0; i < filters * o_rows * o_cols; ++i) {
        //printf("vals[%d]=%f, expect %f\n", i, vals[i], expect_vals[i]);
        REQUIRE( fuzzy_equals_digits(vals[i], expect_vals[i], 6) );
    }
    for (int i = 0; i < filters * o_rows * o_cols; ++i) {
        REQUIRE( fuzzy_equals_digits(outs[i], expect_outs[i], 6) );
    }
    for (int i = 0; i < filters * o_rows * o_cols; ++i) {
        REQUIRE( fuzzy_equals_digits(do_dv[i], expect_do_dv[i], 6) );
    }
}
