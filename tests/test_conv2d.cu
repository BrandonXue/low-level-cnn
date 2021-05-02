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
    REQUIRE(
        calc_dims_pad_valid(9, 3, 1) == 7
    );

    /*
       0 1 2 3 4 5 6 7 8 9 a b c d e
       --1--
             --2--
                   --3--
                         --4--
                               --5--
       (15 - 3) / (int)3 + 1 = 12  / (int)3 + 1 = 5
     */
    REQUIRE(
        calc_dims_pad_valid(15, 3, 3) == 5
    );

    /*
       00-04 02-06 04-08 06-10 08-12 10-14
       12-16 14-18 16-20 18-22 20-24 22-26
       24-28 26-30 28-32 30-34 32-36 34-38
       36-40 38-42 40-44 42-46 44-48 46-50
       48-52 50-54 52-56 54-58 56-60 58-62
     */
    REQUIRE(
        calc_dims_pad_valid(64, 5, 2) == 30
    );
}
