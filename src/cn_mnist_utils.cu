/**
 * Converts a target integer 0-10, 100, 1K, 10K, 100M
 * to a one-hot tensor of rank 1.
 */
 __host__
 void cn_mnist_target_to_vec(float *vec, int target) {
     for (int i = 0; i < 15; ++i) {
         vec[i] = 0;
     }
     switch(target) {
         // integers 0 - 10 use their index as their slot in the tensor
         case 0: case 1: case 2: case 3:
         case 4: case 5: case 6: case 7:
         case 8: case 9: case 10:
             vec[target] = 1;
             break;
         case 100:
             vec[11] = 1;
             break;
         case 1000:
             vec[12] = 1;
             break;
         case 10000:
             vec[13] = 1;
             break;
         case 100000000:
             vec[14] = 1;
             break;
     }
 }