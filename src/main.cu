// Standard
#include <stdio.h>

// Third-party
#include <cuda.h>

// Local
#include "cn_mnist_utils.cu.h"
#include "nn_layers.cu.h"
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
    int l1_activation = 1; // the code for ReLU
    
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
    random_init(l1_weights, l1_filters * l1_kernel_rows * l1_kernel_cols, 0, 1);


    /** Layer2: Dense Layer **/

    // CONFIGURABLE
    int l2_out_nodes = 128;
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
    //float *l2_pdL_pdouts_prev = (float*)malloc(l2_in_nodes * sizeof(float));
    // Each out-node has a weight for each in-node
    float *l2_weights = (float*)malloc(l2_in_nodes * l2_out_nodes * sizeof(float));
    float *l2_grads =   (float*)malloc(l2_in_nodes * l2_out_nodes * sizeof(float));
    // Random initialize weights
    random_init(l2_weights, l2_in_nodes * l2_out_nodes, -1, 1);
    

    /** Layer3: Dense Layer (Output) **/

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
    //float *l3_pdL_pdouts_prev = (float*)malloc(l3_in_nodes * sizeof(float));
    // Each out-node has a weight for each in-node
    float *l3_weights = (float*)malloc(l3_in_nodes * l3_out_nodes * sizeof(float));
    float *l3_grads =   (float*)malloc(l3_in_nodes * l3_out_nodes * sizeof(float));
    // Random initialize weights
    random_init(l3_weights, l3_in_nodes * l3_out_nodes, -1, 1);

    // Memory for training targets
    float target_vec[15];
    // Memory for losses
    float current_loss;
    int summary_iterations = 500;
    float past_losses[summary_iterations];
    bool past_correct[summary_iterations];


    // FILE PATHS
    const char *l1_weights_filename = "data/model/l1_weights.data";
    const char *l2_weights_filename = "data/model/l2_weights.data";
    const char *l3_weights_filename = "data/model/l3_weights.data";

    if (load) {
        printf("Loading from existing model.\n");
        FILE *l1_weights_file = fopen(l1_weights_filename, "rb");
        FILE *l2_weights_file = fopen(l2_weights_filename, "rb");
        FILE *l3_weights_file = fopen(l3_weights_filename, "rb");
        if (l1_weights_file == NULL) {
            printf("Layer 1 weights not found. Using randomly initialized.\n");
        } else {
            fread(l1_weights, sizeof(float), l1_filters * l1_kernel_rows * l1_kernel_cols, l1_weights_file);
            fclose(l1_weights_file);
        }
        if (l2_weights_file == NULL) {
            printf("Layer 2 weights not found. Using randomly initialized.\n");
        } else {
            fread(l2_weights, sizeof(float), l2_in_nodes * l2_out_nodes, l2_weights_file);
            fclose(l2_weights_file);
        }
        if (l3_weights_file == NULL) {
            printf("Layer 3 weights not found. Using randomly initialized.\n");
        } else {
            fread(l3_weights, sizeof(float), l3_in_nodes * l3_out_nodes, l3_weights_file);
            fclose(l3_weights_file);
        }
    }

    int total_iters = atoi(iters);
    for (int iter = 0; iter < total_iters; ++iter) {
        int sample_index = rand() % 14999; // 14999 images, indices 0 - 14998
        int float_offset = sample_index * l1_in_rows * l1_in_cols;
        int target_label = metadata[sample_index].label;
        
        // Layer 1 forward
        Conv2D_forward(
            l1_outs, l1_out_rows, l1_out_cols,
            l1_vals,
            l1_douts_dvals,
            data + float_offset, l1_in_rows, l1_in_cols,
            l1_weights, l1_kernel_rows, l1_kernel_cols,
            l1_stride_rows, l1_stride_cols, l1_filters,
            l1_activation
        );
        // Layer 2 forward
        Dense_forward(
            l2_outs, l2_out_nodes,
            l2_vals,
            l2_douts_dvals,
            l1_outs, l2_in_nodes,
            l2_weights,
            l2_activation
        );
        // Layer 3 forward
        Dense_forward(
            l3_outs, l3_out_nodes,
            l3_vals,
            l3_douts_dvals,
            l2_outs, l3_in_nodes,
            l3_weights,
            l3_activation
        );

        // Calculate loss
        cn_mnist_target_to_vec(target_vec, target_label);
        cat_cross_entropy(
            l3_out_nodes, target_vec, l3_vals, l3_outs,
            l3_pdL_pdvals, &current_loss
        );

        // Store aggregate measures
        int predicted_index = argmax(l3_outs, 15);
        past_correct[(iter % summary_iterations)] = (target_vec[predicted_index] == 1);
        past_losses[(iter % summary_iterations)] = current_loss;

        // Layer 3 backward
        Dense_backward(
            l3_out_nodes,
            l3_pdL_pdouts,
            l3_douts_dvals,
            l3_pdL_pdvals,
            l2_pdL_pdouts,
            l2_outs, l3_in_nodes,
            l3_weights,
            l3_grads,
            l3_activation
        );
        // Layer 2 backward
        Dense_backward(
            l2_out_nodes,
            l2_pdL_pdouts,
            l2_douts_dvals,
            l2_pdL_pdvals,
            l1_pdL_pdouts,
            l1_outs, l2_in_nodes,
            l2_weights,
            l2_grads,
            l2_activation
        );
        // Layer 1 backward
        Conv2D_backward(
            l1_out_rows, l1_out_cols,
            l1_pdL_pdouts,
            l1_douts_dvals,
            l1_pdL_pdvals,
            l1_pdL_pdouts_prev,
            data + float_offset, l1_in_rows, l1_in_cols,
            l1_weights, l1_kernel_rows, l1_kernel_cols,
            l1_grads,
            l1_stride_rows, l1_stride_cols, l1_filters
        );

        // Lower learning rate (alpha) over time.
        // Kind of like simulated annealing.
        float alpha = 0.1 - (0.05 * (float)iter / (float)total_iters);

        // Update layer 1
        SGD_update_params(
            alpha, l1_weights, l1_grads,
            l1_filters * l1_kernel_rows * l1_kernel_cols
        );
        // Update layer 2
        SGD_update_params(
            alpha, l2_weights, l2_grads,
            l2_in_nodes * l2_out_nodes
        );
        // Update layer 3
        SGD_update_params(
            alpha, l3_weights, l3_grads,
            l3_in_nodes * l3_out_nodes
        );

        // This will execute once every summary_iterations.
        if ( (iter+1) % summary_iterations == 0) {
            // important sanity check: make sure offsets are correct.
            // images and metadata should make sense and match.
            flt_img_to_ascii(data+float_offset, 64, 64, 1);
            printf(
                "Image file: %s. Correct label:  %d\n\n",
                metadata[sample_index].input,
                metadata[sample_index].label
            );

            // Pretty print the predictions table
            print_predictions(l3_outs);
            
            float avg_loss = 0;
            float accuracy = 0;
            for (int i = 0; i < summary_iterations; ++i) {
                avg_loss += past_losses[i];
                accuracy += (past_correct[i] ? 1 : 0);
            }
            avg_loss /= summary_iterations;
            accuracy = accuracy * 100 /  summary_iterations;
            printf("\nCurrent iteration: %d\n", iter + 1);
            printf("Average loss over last %d iterations: %f\n", summary_iterations, avg_loss);
            printf("Accuracy of last %d iterations: %0.2f%%\n", summary_iterations, accuracy);
        }
    }
    
    if (save) {
        printf("Saving model.\n");
        FILE *l1_weights_file = fopen(l1_weights_filename, "wb");
        FILE *l2_weights_file = fopen(l2_weights_filename, "wb");
        FILE *l3_weights_file = fopen(l3_weights_filename, "wb");
        fwrite(l1_weights, sizeof(float), l1_filters * l1_kernel_rows * l1_kernel_cols, l1_weights_file);
        fwrite(l2_weights, sizeof(float), l2_in_nodes * l2_out_nodes, l2_weights_file);
        fwrite(l3_weights, sizeof(float), l3_in_nodes * l3_out_nodes, l3_weights_file);
        fclose(l1_weights_file);
        fclose(l2_weights_file);
        fclose(l3_weights_file);
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


/* START: Used for testing dense layers only */
void generate_input(float *buffer, int len) {
    for (int i = 0; i < len; ++i) {
        buffer[i] = (float)(rand() % 10);
    }
}

void generate_answer(float *input, float *answer) {
    float sum1 = input[0] + input[1] + input[2] + input[3] + input[4];
    float sum2 = input[5] + input[6] + input[7] + input[8] + input[9];
    float sum3 = input[10] + input[11] + input[12] + input[13] + input[14];
    if (sum1 > sum2 && sum1 > sum3) {
        answer[0] = 1; answer[1] = 0; answer[2] = 0;
    } else if (sum2 > sum1 && sum2 > sum3) {
        answer[0] = 0; answer[1] = 1; answer[2] = 0;
    } else {
        answer[0] = 0; answer[1] = 0; answer[2] = 1;
    }
}

/**
 * For testing the performance of two dense layers on predicting
 * a very basic function.
 */
void dense_layer_test() {
    int INPUT_SIZE = 15, L1_NODES = 6, OUT_NODES = 3;
    float input[INPUT_SIZE];
    float answer[OUT_NODES];
    float l1_outs[L1_NODES];
    float l1_vals[L1_NODES];
    float l1_douts_dvals[L1_NODES];
    float l1_pdL_pdvals[L1_NODES];
    float l1_pdL_pdouts[L1_NODES];
    float l1_pdL_pdouts_prev[INPUT_SIZE];
    float l1_weights[L1_NODES * INPUT_SIZE];
    random_init(l1_weights, L1_NODES * INPUT_SIZE, -1, 1);
    float l1_grads[L1_NODES * INPUT_SIZE];

    float l2_outs[OUT_NODES];
    float l2_vals[OUT_NODES];
    float l2_douts_dvals[OUT_NODES];
    float l2_pdL_pdvals[OUT_NODES];
    float l2_pdL_pdouts[OUT_NODES];
    float l2_weights[OUT_NODES * L1_NODES];
    random_init(l2_weights, OUT_NODES * L1_NODES, -1, 1);
    float l2_grads[OUT_NODES * L1_NODES];

    float current_loss;
    int period = 1000;
    float past_losses[period];
    bool past_correct[period];
    int total_iterations = 10000;
    for (int i = 0; i < total_iterations; ++i) {
        generate_input(input, INPUT_SIZE);
        generate_answer(input, answer);
        Dense_forward(
            l1_outs, L1_NODES,
            l1_vals,
            l1_douts_dvals,
            input, INPUT_SIZE,
            l1_weights,
            0 // sigmoid
        );
        Dense_forward(
            l2_outs, OUT_NODES,
            l2_vals,
            l2_douts_dvals,
            l1_outs, L1_NODES,
            l2_weights,
            2 // no activation
        );
        cat_cross_entropy(
            OUT_NODES, answer, l2_vals, l2_outs,
            l2_pdL_pdvals, &current_loss
        );
        //printf("l2outs %f %f %f\n", l2_outs[0], l2_outs[1], l2_outs[2]);
        past_losses[(i % period)] = current_loss;
        bool current_correct = false;
        if (l2_outs[0] > l2_outs[1] && l2_outs[0] > l2_outs[2] && answer[0] == 1)
            current_correct = true;
        else if (l2_outs[1] > l2_outs[0] && l2_outs[1] > l2_outs[2] && answer[1] == 1)
            current_correct = true;
        else if (answer[2] == 1)
            current_correct = true;
        past_correct[(i % period)] = current_correct;
         Dense_backward(
            OUT_NODES,
            l2_pdL_pdouts,
            l2_douts_dvals,
            l2_pdL_pdvals,
            l1_pdL_pdouts,
            l1_outs, L1_NODES,
            l2_weights,
            l2_grads,
            2
        );
          Dense_backward(
            L1_NODES,
            l1_pdL_pdouts,
            l1_douts_dvals,
            l1_pdL_pdvals,
            l1_pdL_pdouts_prev,
            input, INPUT_SIZE,
            l1_weights,
            l1_grads,
            0
        );
        float alpha = 0.1 - 0.05 * ((float)i / (float)total_iterations);
        SGD_update_params(alpha, l1_weights, l1_grads, L1_NODES * INPUT_SIZE);
        SGD_update_params(alpha, l2_weights, l2_grads, OUT_NODES * L1_NODES);
        
        if ( (i + 1) % period == 0 ) {
            printf("Input\n");
            print_vec(input, 15, 0);
            printf("Expected\n");
            print_vec(answer, 3, 1);
            printf("Result\n");
            print_vec(l2_outs, 3, 6);
            int correct_count = 0;
            float total_loss = 0;
            for (int j = 0; j < period; ++j) {
                if (past_correct[j])
                    correct_count ++;
                total_loss += past_losses[j];
                
            }
            printf("Over the past %d iterations:\n", period);
            printf("Accuracy: %f\n", (float)correct_count / (float)period);
            printf("Average loss: %f\n", total_loss / (float)period);
        }
    }
}
/* END Used for testing dense layers only */



int main(int argc, char *argv[]) {
    srand( 479 ); // seed PRNG for reproducible results
    //dense_layer_test();
    interactive_loop();
    return 0;
}
