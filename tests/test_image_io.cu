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

TEST_CASE( "image reports correct errors", "[image_io]" ) {
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

void print_unexpected_label(char *file, int code, int label) {
    printf("File: %s has code %d. Unexpected label: %d.\n",
        file, code, label);
}

TEST_CASE( "chinese mnist meta data loads", "[image_io]" ) {
    // Call the metadata loader function
    InputLabel *metadata = load_chinese_mnist_info();

    // For each observation, make sure the code value has the expected label.
    // The format of the file names is:
    // data/chinese_mnist/data/input_<suite_id>_<sample_id>_<code>.jpg
    const int OBSERVATIONS = 14999;
    bool valid = true;
    for (int i = 0; i < OBSERVATIONS; ++i) {
        char *filename = metadata[i].input;

        // Get the code from the filename
        char char_code[64];
        char *code_ptr = strrchr(filename, '_') + 1;
        strncpy(char_code, code_ptr, 63);
        char *dot_ptr = strrchr(char_code, '.');
        int dot_index = dot_ptr - char_code;
        char_code[dot_index] = '\0';

        // Check if code and label combination are as expected
        int code = atoi(char_code);
        int label = metadata[i].label;
        switch(code) {
            case 0: case 1: case 2: case 3: case 4: case 5:
            case 6: case 7: case 8: case 9: case 10: case 11:
                if (code - 1 != label) {
                    valid = false;
                    print_unexpected_label(filename, code, label);
                }
                break;
            case 12:
                if (label != 100) {
                    valid = false;
                    print_unexpected_label(filename, code, label);
                }
                break;
            case 13:
                if (label != 1000) {
                    valid = false;
                    print_unexpected_label(filename, code, label);
                }
                break;
            case 14:
                if (label != 10000) {
                    valid = false;
                    print_unexpected_label(filename, code, label);
                }
                break;
            case 15:
                if (label != 100000000) {
                    valid = false;
                    print_unexpected_label(filename, code, label);
                }
                break;
            default:
                printf("Unknown code:\n");
                print_unexpected_label(filename, code, label);
        }
    }
    REQUIRE( valid );
}
