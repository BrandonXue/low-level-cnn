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

TEST_CASE( "fuzzy equals accepts within tolerance", "[math]") {
    REQUIRE( fuzzy_equals_digits(3.44444, 3.44445, 3) );
    REQUIRE( fuzzy_equals_digits(12.3, 12.4, 0) );
    REQUIRE( fuzzy_equals_digits(0.0000000001, 0.00000000005, 10) );
}

TEST_CASE( "fuzzy equals rejects exceeding tolerance", "[math]" ) {
    REQUIRE( !fuzzy_equals_digits(3.44444, 3.44445, 4) );
    REQUIRE( !fuzzy_equals_digits(12.3, 12.4, 2) );
    REQUIRE( !fuzzy_equals_digits(0.0000000001, 0.00000000004, 10) );
}
