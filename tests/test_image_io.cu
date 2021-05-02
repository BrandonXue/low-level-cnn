#include "catch.hpp"

#include "image_io.cu.h"

TEST_CASE( "image loads properly", "[image_io]" ) {
    int status;
    unsigned char *data = load_img_carefully(
        "tests/data/input_8_8_8.jpg",
        64, 64, 1, &status
    );
    REQUIRE( status == 0 );
}

TEST_CASE( "image reports correct errors", "[image_io]") {
    int status;
    unsigned char *data1 = load_img_carefully(
        "tests/data/input_8_8_8.jpg",
        63, 64, 2, &status
    );
    // expected mismatches
    // nul = 0b1000    +8
    // wid = 0b0100    +4 <-
    // hgt = 0b0010    +2
    // chn = 0b0001    +1 <-
    REQUIRE( status == 5 );
    //stbi_image_free(data1);

    unsigned char *data2 = load_img_carefully(
        "tests/data/input_nonexistant.jpg",
        64, 64, 1, &status
    );
    // see expected mismatches above for info
    REQUIRE( data2 == NULL );
    REQUIRE( status == 8 );
    //stbi_image_free(data2);
}

