// Standard
#include <locale.h>
#include <stdio.h>
#include <string.h>
#include <wchar.h>

// Third party
#define STB_IMAGE_IMPLEMENTATION
#define STBI_ONLY_JPEG
#include "stb_image.h"

// Local
#include "image_io.cu.h"

/**
 * Load an image (preferably jpeg) using stbi_load().
 * Warns when the actual properties of the loaded image differ from
 * the expected properties.
 *
 * On success, returns a pointer to the image array and sets status to 0.
 * On failure, returns NULL and sets status to nonzero. The bytes of the
 * nonzero return mean the following:
 *     0b1000 (8): Unknown error. stbi_load() returned NULL.
 *     0b0100 (4): Width differs from expect_wid
 *     0b0010 (2): Height differs from expect_hgt
 *     0b0001 (1): Components/channels differs from expect_c 
 *
 * @return A pointer to the image array, with components interleaved.
 *     Or returns NULL if something went wrong.
 */
__host__
unsigned char *load_img_carefully(
    const char* file_path,
    int expect_wid, int expect_hgt, int expect_c,
    int *status
) {

    int width, height, components;

    unsigned char *data = stbi_load(
        file_path, &width, &height, &components, 0
    );
    
    // Clear status
    *status = 0;

    // If fail loading
    if (data == NULL) {
        *status |= 8;
    }
    // On load succeed
    else {
        if (expect_wid != width) {
            *status |= 4;
        }
        if (expect_hgt != height) {
            *status |= 2;
        }
        if (expect_c != components) {
            *status |= 1;
        }
    }
    return data;
}

/**
 * Convert single-byte numeric intensity into an ASCII character,
 * where the character chosen is visually darker the higher the
 * intensity is.
 */
__host__
wchar_t int256_to_unicode(int intensity) {
    if (intensity >= 192)   // 256 * 3/4
        return (wchar_t)0x2593; // dark shade
    if (intensity >= 128)   // 256 * 2/4
        return (wchar_t)0x2592; // medium shade
    if (intensity >= 64)    // 256 * 1/4
        return (wchar_t)0x2591; // light shade
                            // 256 * 0/4
    return (wchar_t)0x0020;     // whitespace
}

__host__
char int256_to_ascii(int intensity) {
    if (intensity >= 204) // 256 * 4/5
        return '#';
    if (intensity >= 153) // 256 * 3/5
        return '*';
    if (intensity >= 102) // 256 * 2/5
        return '+';
    if (intensity >= 51)  // 256 * 1/5
        return '.';
    return ' ';
}

__host__
char flt01_to_ascii(float intensity) {
    if (intensity >= 0.8) 
        return '#';
    if (intensity >= 0.6)
        return '*';
    if (intensity >= 0.4)
        return '+';
    if (intensity >= 0.2)
        return '.';
    return ' ';
}

/**
 * Print 1 channel grayscale or 3 channel RGB image as ASCII art.
 */
__host__
void img_to_ascii(unsigned char *data, int width, int height, int c) {
    if (data != NULL && (c == 1 || c == 3)) {
        for (int y = 0; y < height; ++y) {
            for (int x = 0; x < width; ++x) {
                int avg = 0;
                for (int chn = 0; chn < c; ++chn) {
                    // y * width + x gives the pixel, multiply it by channels
                    // add current channel: chn
                    avg += (int)data[c * (y * width + x) + chn];
                }
                avg /= c;
                printf("%c", int256_to_ascii(avg));
            }
            printf("\n");
        }
    }
}

/**
 * Same as img_to_ascii but takes floats. First performs a normalization.
 */
__host__
void flt_img_to_ascii(float *data, int width, int height, int c) {
    if (data != NULL && (c == 1 || c == 3)) {
        float min_data = data[0];
        float max_data = data[0];
        for (int y = 0; y < height; ++y) {
            for (int x = 0; x < width; ++x) {
                float avg = 0;
                for (int chn = 0; chn < c; ++chn) {
                    avg += data[c * (y * width + x) + chn];
                }
                avg /= (float)c;
                if (avg < min_data) {
                    min_data = avg;
                }
                if (avg > max_data) {
                    max_data = avg;
                }
            }
        }

        for (int y = 0; y < height; ++y) {
            for (int x = 0; x < width; ++x) {
                float avg = 0;
                for (int chn = 0; chn < c; ++chn) {
                    avg += data[c * (y * width + x) + chn];
                }
                avg /= (float)c;
                avg = (avg - min_data) / (max_data - min_data);
                printf("%c", flt01_to_ascii(avg));
            }
            printf("\n");
        }
    }
}

/**
 * Must be called before attempting to print Unicode.
 */
__host__
void set_locale_lc_all() {
    setlocale(LC_ALL, "");
}

/**
 * Print 1 channel grayscale or 3 channel RGB image as Unicode art.
 */
__host__
void img_to_unicode(unsigned char *data, int width, int height, int c) {
    if (data != NULL && (c == 1 || c == 3)) {
        for (int y = 0; y < height; ++y) {
            for (int x = 0; x < width; ++x) {
                int avg = 0;
                for (int chn = 0; chn < c; ++chn) {
                    // y * width + x gives the pixel, multiply it by channels
                    // add current channel: chn
                    avg += (int)data[c * (y * width + x) + chn];
                }
                avg /= c;
                printf("%lc", (wchar_t)int256_to_unicode(avg));
            }
            printf("\n");
        }
    }
}

__host__
void copy_until(char *dst, char *src, char stop, int limit) {
    int i = 0;
    int len_src = strlen(src);
    while (i < len_src && i < limit && dst[i] != stop) {
        src[i] = dst[i];
    }
}

/**
 * The chinese_mnist.csv metadata has the following format:
 * suite_id,sample_id,code,value,character
 * where the first line is the header information, and the
 * subsequent 14999 lines are data.
 */
__host__
InputLabel *load_chinese_mnist_info() {
    // Open the file that contains data about each ChineseMNIST input and label
    FILE* fs = fopen("data/chinese_mnist/chinese_mnist.csv", "r");
    // Create an array of InputLabel which this function will construct
    InputLabel *info = (InputLabel*)malloc(14999 * sizeof(InputLabel));

    // The files all start with this prefix
    const char *prefix = "data/chinese_mnist/data/input_";
    int len_prefix = strlen(prefix);

    // The files all end with this suffix
    const char *suffix = ".jpg";
    int len_suffix = strlen(suffix);

    int len_img_file_buf = 64;
    char csv_line_buf[64];
    
    // Skip the header line
    char *retval = fgets(csv_line_buf, 63, fs);
    if (retval == NULL) {
        printf("Something went wrong loading the metadata.\n");
    }

    // Iterate over the lines of the csv file
    int line_index = 0;
    while (fgets(csv_line_buf, 63, fs)) {
        // First copy the common file prefix to the file name
        char *img_file_buf = (char*)malloc(len_img_file_buf * sizeof(char));
        strncpy(img_file_buf, prefix, len_img_file_buf);

        // Now iterate over the csv line and copy the numbers over
        int line_len = strlen(csv_line_buf);
        int comma_count = 0;
        for (int i = 0; i < line_len; ++i) {
            // the first three values are part of the input string
            if (csv_line_buf[i] != ',') {
                img_file_buf[len_prefix+i] = csv_line_buf[i];
            } else {
                ++comma_count;

                // If three commas
                // append suffix parse the label, create struct, then break
                if (comma_count == 3) {
                    strncpy(img_file_buf + len_prefix + i, suffix, len_suffix);
                    ++i;
                    // Parse the int type label
                    char value[32];
                    int j = i;
                    while (csv_line_buf[j] != ',') {
                        value[j - i] = csv_line_buf[j];
                        ++j;
                    }
                    value[j - i] = '\0';
                    int int_value = atoi(value);
                    // Create struct for this input entry in the struct array
                    //printf("input: %s has label %d, as str %s\n", img_file_buf, int_value, value);
                    info[line_index] = {.input = img_file_buf, .label = int_value};
                    break;
                } else {
                    img_file_buf[len_prefix+i] = '_';
                }
            }
        }
        ++line_index;
    }
    return info;  
}

__host__
void preprocess_images(float *out, InputLabel *input_labels) {
    int status;
    for (int img = 0; img < 14999; ++img) {
        unsigned char *new_image = load_img_carefully(
            input_labels[img].input, 64, 64, 1, &status
        );
        // load the pixels into the correct place. divid each by 255 to normalize.
        for (int pixel = 0; pixel < 64 * 64; ++pixel) {
            out[64 * 64 * img + pixel] = (float)new_image[pixel] / 255.0;
        }
        free(new_image);
    }
}
