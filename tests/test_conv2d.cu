#include "catch.hpp"

#include "conv2d.cu.h"

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
