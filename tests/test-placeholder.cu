#include "catch.hpp"

#include "placeholder.cu.h"

TEST_CASE( "placeholder test works", "[sanity]" ) {
    REQUIRE( placeholder() == true );
}


