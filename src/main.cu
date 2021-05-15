// Standard
#include <stdio.h>

// Third-party
#include <cuda.h>

// Local
#include "layers.cu.h"
#include "image_io.cu.h"
#include "math.cu.h"
#include "string.cu.h"

enum Command{EXIT, HELP, PREPROCESS, SHOW, TRAIN, UNKNOWN};

Command get_command(Tokens *toks) {
    if (Tokens_match_at(toks, 0, "exit") ||
        Tokens_match_at(toks, 0, "quit")) {
        return EXIT;
    }
    if (Tokens_match_at(toks, 0, "help") ||
        Tokens_match_at(toks, 0, "?") ||
        Tokens_match_at(toks, 0, "h")) {
        return HELP;
    }
    if (Tokens_match_at(toks, 0, "preprocess")) {
        return PREPROCESS;
    }
    if (Tokens_match_at(toks, 0, "show")) {
        return SHOW;
    }
    if (Tokens_match_at(toks, 0, "train")) {
        return TRAIN;
    }
    return UNKNOWN;
}

void print_prompt() {
    printf("p2-sh> ");
}

void cmd_help() {
    printf("\tCommands:\n\n"

           "\texit\n"
               "\t\tQuit the program. Model does not save automatically.\n\n"

           "\thelp\n"
               "\t\tPrint this help.\n\n"

           "\tpreprocess\n"
               "\t\tLoad the Chinese MNIST metadata and preprocess images.\n\n"

           "\tshow (image <filename>) | (preprocessed <index>)\n"
               "\t\tshow image <filename>: Print an image as ASCII art.\n"
               "\t\tDo not need to type the path in filename.\n"
               "\t\tshow preprocessed <index>: Same as show image except\n"
               "\t\tthis uses the pre-processed data. This is mainly just\n"
               "\t\ta sanity check.\n\n"
            
           "\ttrain <n-iters> [--load [--save]]\n"
               "\t\tTrain the neural network for the specified number of inputs.\n"
               "\t\tThis does NOT do a train-test split. All data may be used.\n"
               "\t\tOptionally specify whether weights should be saved.\n\n"
    );
}

void cmd_show(Tokens *toks, float *data) {
    if (Tokens_match_at(toks, 1, "image")) {
        if (Tokens_get_count(toks) < 3) {
            printf("Error: Missing file name for \"show image\".\n");
            return;
        }
        int status;
        char buf[128];
        char prefix[] = "data/chinese_mnist/data/";
        strcpy(buf, prefix);
        strcat(buf, Tokens_at(toks, 2));
        unsigned char *image = load_img_carefully(
            buf, 64, 64, 1, &status
        );
        //set_locale_lc_all();
        if (status == 0) {
            img_to_ascii(image, 64, 64, 1);
            free(image);
        } else {
            printf("Error: Image could not be loaded: %s\n", buf);
        }
    } else if (Tokens_match_at(toks, 1, "preprocessed")) {
        if (data == NULL) {
            printf("Error: Preprocess the data first.\n");
        } 
        if (Tokens_get_count(toks) < 3) {
            printf("Error: Missing the index of the image for \"show preprocessed\".\n");
            return;
        }
        char *index = Tokens_at(toks, 2);
        if ( (atoi(index) > 0 || Tokens_match_at(toks, 2, "0")) && atoi(index) < 14999 ) {
            flt_img_to_ascii(data + atoi(index) * 64 * 64, 64, 64, 1); 
        } else {
            printf("Error: Enter an index between [0, 14998]\n");
        }
    } else {
        printf("Error: \"show\" must be used with a valid sub-command.\n");
    }
}

void cmd_preprocess(InputLabel **metadata, float **data) {
    const char *filepath = "data/chinese_mnist/processed.data";
    
    // If already loaded, don't do anything
    if (*data != NULL) {
        printf("Metadata and data already loaded.\n");
        return;
    }

    printf("Loading Chinese MNIST metadata. (This contains filenames and target labels.)\n");
    *metadata = load_chinese_mnist_info();
    *data = (float*)malloc(64 * 64 * 14999 * sizeof(float));

    FILE *processed_imgs_file = fopen(filepath, "rb");
    if (processed_imgs_file == NULL) {
        printf("Did not find previously pre-processed images. Loading and pre-processing now.\n");
        preprocess_images(*data, *metadata);
        
        printf("Saving to file %s\n\n", filepath);
        processed_imgs_file = fopen(filepath, "wb");
        fwrite(*data, sizeof(float), 64 * 64 * 14999, processed_imgs_file);
        fclose(processed_imgs_file);
    } else {
        printf("Found previously pre-processed images. Loading now.\n");
        fread(*data, sizeof(float), 64 * 64 * 14999, processed_imgs_file);
        fclose(processed_imgs_file);
    }
}

void cmd_train(InputLabel *metadata, float *data, Tokens *toks) {
    if (metadata == NULL || data == NULL) {
        printf("Error: Preprocess the images first. (This will also load the metadata.)\n");
        return;
    } 
    char *iters = Tokens_at(toks, 1);
    if (iters == NULL || atoi(iters) <= 0) {
        printf("Error: Enter a valid number of iterations.\n");
        return;
    } else {
        printf("Model will train for %d iterations.\n", atoi(iters));
    }
    bool load = Tokens_match_at(toks, 2, "--load");
    bool save = Tokens_match_at(toks, 3, "--save");
    
    /**** Setup configuration and allocate memory for the model ****/
    
    /** Layer 1: Conv2D Layer **/

    // CONFIGURABLE
    int l1_in_rows = 64, l1_in_cols = 64;
    int l1_filters = 16;
    int l1_kernel_rows = 5, l1_kernel_cols = 5;
    int l1_stride_rows = 3, l1_stride_cols = 3;
    int l1_activation = 0; // the code for sigmoid
    
    // Output dimensions depend on input dimensions, kernel size, and stride
    int l1_out_rows = calc_dims_pad_valid(l1_in_rows, l1_kernel_rows, l1_stride_rows);
    int l1_out_cols = calc_dims_pad_valid(l1_in_cols, l1_kernel_cols, l1_stride_cols);
    // Each filter produces one activation map, and each map needs to store pre-activation vals,
    // as well as local gradients.
    float *l1_vals =        (float*)malloc(l1_filters * l1_out_rows * l1_out_cols * sizeof(float));
    float *l1_outs =        (float*)malloc(l1_filters * l1_out_rows * l1_out_cols * sizeof(float));
    float *l1_pdL_pdouts =  (float*)malloc(l1_filters * l1_out_rows * l1_out_cols * sizeof(float));
    float *l1_douts_dvals = (float*)malloc(l1_filters * l1_out_rows * l1_out_cols * sizeof(float));
    float *l1_pdL_pdvals =  (float*)malloc(l1_filters * l1_out_rows * l1_out_cols * sizeof(float));
    // (currently unused) The layer needs memory to calculate the upstream gradients.
    float *l1_pdL_pdouts_prev = (float*)malloc(l1_in_rows * l1_in_cols * sizeof(float));
    // Each filter has (kernel_rows * kernel_cols) weights and global gradient.
    float *l1_weights = (float*)malloc(l1_filters * l1_kernel_rows * l1_kernel_cols * sizeof(float));
    float *l1_grads =   (float*)malloc(l1_filters * l1_kernel_rows * l1_kernel_cols * sizeof(float));
    // Random initialize weights
    random_init(l1_weights, l1_filters * l1_kernel_rows * l1_kernel_cols, -0.1, 0.1);

    /** Layer2: Dense Layer **/

    // CONFIGURABLE
    int l2_out_nodes = 64;
    int l2_activation = 0; // the code for sigmoid

    // Input nodes depend on the previous layer
    int l2_in_nodes = l1_filters * l1_out_rows * l1_out_cols;
    // The nodes need pre-activation vals and activated outs.
    float *l2_vals =        (float*)malloc(l2_out_nodes * sizeof(float));
    float *l2_outs =        (float*)malloc(l2_out_nodes * sizeof(float));
    float *l2_pdL_pdouts =  (float*)malloc(l2_out_nodes * sizeof(float));
    float *l2_douts_dvals = (float*)malloc(l2_out_nodes * sizeof(float));
    float *l2_pdL_pdvals =  (float*)malloc(l2_out_nodes * sizeof(float));
    // The layers needs memory to calculate the upstream gradients
    float *l2_pdL_pdouts_prev = (float*)malloc(l2_in_nodes * sizeof(float));
    // Each out-node has a weight for each in-node
    float *l2_weights = (float*)malloc(l2_in_nodes * l2_out_nodes * sizeof(float));
    float *l2_grads =   (float*)malloc(l2_in_nodes * l2_out_nodes * sizeof(float));
    // Random initialize weights
    random_init(l2_weights, l2_in_nodes * l2_out_nodes, -0.1, 0.1);
    
    /** Layer3: Dense Layer (Output) **/
    
    // CONFIGURABLE
    int l3_out_nodes = 15; // 15 classes to predict
    int l3_activation = 2; // the code for none; no activation
                           // the cross entropy is programmed to apply softmax activation

    // Input nodes depend on the previous layer
    int l3_in_nodes = l2_out_nodes;
    // The nodes need pre-activation vals and activated outs.
    float *l3_vals =        (float*)malloc(l3_out_nodes * sizeof(float));
    float *l3_outs =        (float*)malloc(l3_out_nodes * sizeof(float));
    float *l3_pdL_pdouts =  (float*)malloc(l3_out_nodes * sizeof(float));
    float *l3_douts_dvals = (float*)malloc(l3_out_nodes * sizeof(float));
    float *l3_pdL_pdvals =  (float*)malloc(l3_out_nodes * sizeof(float));
    // The layers needs memory to calculate the upstream gradients
    float *l3_pdL_pdouts_prev = (float*)malloc(l3_in_nodes * sizeof(float));
    // Each out-node has a weight for each in-node
    float *l3_weights = (float*)malloc(l3_in_nodes * l3_out_nodes * sizeof(float));
    float *l3_grads =   (float*)malloc(l3_in_nodes * l3_out_nodes * sizeof(float));
    // Random initialize weights
    random_init(l3_weights, l3_in_nodes * l3_out_nodes, -0.1, 0.1);

    if (load) {
        printf("Loading from existing model.\n");
    }
    
    if (save) {
        printf("The model will be saved at the end of the session.\n");
    }
}

void interactive_loop() {
    
    InputLabel *metadata = NULL;
    float *chinese_mnist_processed = NULL;
    char buf[1024];
    while (true) {
        print_prompt();
        fgets(buf, 1023, stdin);
            
        Tokens *toks = Tokens_create();
        
        // If tokens not fetched successfully
        if (!Tokens_fetch(toks, buf) || buf[0] == '\n') {
            continue;
        }
        
        Command cmd = get_command(toks);
        
        // If command not understood
        if (cmd == UNKNOWN) {
            printf("Could not understand the command.\n");
            continue;
        }
        
        /** Handle commands **/

        if (cmd == HELP) {
            cmd_help();
        }

        else if (cmd == EXIT) {
            break;
        }

        else if (cmd == PREPROCESS) {
            cmd_preprocess(&metadata, &chinese_mnist_processed);
        }

        else if (cmd == SHOW) {
            cmd_show(toks, chinese_mnist_processed);
        }

        else if (cmd == TRAIN) {
            cmd_train(metadata, chinese_mnist_processed, toks);
        }

        toks = Tokens_destroy(toks);
    }
}


int main(int argc, char *argv[]) {
    interactive_loop();
    return 0;
}
