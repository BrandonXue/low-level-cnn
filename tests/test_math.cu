#include "catch.hpp"

#include "math.cu.h"

TEST_CASE( "round by digits works", "[math]" ) {
    REQUIRE( round_digits(4.2345678, 3) == 4.235 );
    REQUIRE( round_digits(4.2345678, 9) == 4.2345678 );
    REQUIRE( round_digits(4.2345678, 0) == 4.0 );
}

TEST_CASE( "round by digits precision not too awful", "[math]" ) {
    REQUIRE( 
        round_digits( 900000000000000000000000.1234567, 5) 
        == 900000000000000000000000.0
    );
}
