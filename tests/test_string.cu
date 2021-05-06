#include "catch.hpp"

#include "string.cu.h"

SCENARIO( "Token object methods work", "[string]" ) {

    GIVEN( "A newly constructed Token object" ) {
        Tokens *toks = Tokens_create();

        REQUIRE( toks != NULL );
        REQUIRE( toks->data != NULL );
        REQUIRE( toks->size > 0 );
        REQUIRE( toks->count == 0 );

        WHEN( "tokens are added" ) {
            char token1[] = "Hello\n";
            char token2[] = "there";
            char token3[] = "this\n";
            char token4[] = "is";
            char token5[] = "a";
            char token6[] = "test.\n";
            char token7[] = "A";
            char token8[] = "very";
            char token9[] = "long";
            char token10[] = "test";
            Tokens_append(toks, token1);
            Tokens_append(toks, token2);
            Tokens_append(toks, token3);
            Tokens_append(toks, token4);
            Tokens_append(toks, token5);
            Tokens_append(toks, token6);
            Tokens_append(toks, token7);
            Tokens_append(toks, token8);
            Tokens_append(toks, token9);
            Tokens_append(toks, token10);
            
            THEN( "the count and size change" ) {
                REQUIRE( toks->count == 10 );
                REQUIRE( toks->size >= 16 );
            }

            THEN( "tokens can be matched" ) {
                REQUIRE( Tokens_match_at(toks, 0, "Hello") );
                REQUIRE( Tokens_match_at(toks, 1, "there\n") );
                REQUIRE( Tokens_match_at(toks, 2, "this\n") );
                REQUIRE( Tokens_match_at(toks, 3, "is\n") );
            }

            THEN( "things that shouldn\'t match, don\'t" ) {
                REQUIRE( !Tokens_match_at(toks, 0, "hello") );
                REQUIRE( !Tokens_match_at(toks, 5, "test") );
                REQUIRE( !Tokens_match_at(toks, 9, "test.") );
            }

            THEN( "destructor works" ) {
                toks = Tokens_destroy(toks);
                REQUIRE( toks == NULL );
            }
        }
    }
}
