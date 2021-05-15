import math
import random

import cv2
import csv
import numpy as np
from numpy.lib import stride_tricks

def sigmoid(x):
    '''Sigmoid activation function.'''

    # Experimentally found out that if x > 709 there is overflow in exp(x)
    # Make sure -x is no less than -709, so that x is no greater than 709

    clipped_x = np.clip(x, -709, None)
    return 1 / (1 + np.exp(-clipped_x))

    # scalar version:
    # if x < -709: x = -709
    # return 1 / (1 + math.exp(-x))

def deriv_sigmoid(x):
    '''First derivative of the sigmoid function.'''

    s = sigmoid(x)
    return s * (1 - s)

def relu(x):
    '''Rectified Linear Unit (ReLU) activation function.'''

    return np.maximum(x, 0)

    # scalar version:
    # if x > 0:   return x
    # else:       return 0

def deriv_relu(x):
    '''First derivative of the ReLU function.'''
    
    #                        0   if x1 < 0
    # heaviside(x1, x2) =   x2   if x1 == 0
    #                        1   if x1 > 0
    return np.heaviside(x, 0)

    # scalar version:
    # if x > 0:   return 1
    # else:       return 0

class Conv2DLayer:
    def __init__(self, input_dims, activation_fn, deriv_activation_fn, kernel_dims, stride_dims, num_filters) -> None:
        ''' 
        only supports filters of one channel (one kernel).
        kernel_dims is a tuple. stride is also a tuple
        Expects an input tensor of rank 2 (i.e. matrix) from pred layer.
        '''
        self.input_dims = input_dims
        self.activation = activation_fn
        self.deriv_activation = deriv_activation_fn
        self.kernel_dims = kernel_dims
        self.stride_dims = stride_dims
        self.num_filters = num_filters

        self.weights = np.random.randn(num_filters, kernel_dims[0], kernel_dims[1])
        self.vals = np.empty((num_filters, kernel_dims[0], kernel_dims[1]))
        self.outs = np.empty((num_filters, kernel_dims[0], kernel_dims[1]))
        self.dOut_dVals = np.empty((num_filters, kernel_dims[0], kernel_dims[1]))
        self.activation = activation_fn
        self.deriv_activation = deriv_activation_fn

    def calc_output_dims(self):
        output_wid = (self.input_dims[0] - self.kernel_dims[0]) // self.stride_dims[0] + 1
        output_hgt = (self.input_dims[1] - self.kernel_dims[1]) // self.stride_dims[1] + 1
        return (self.num_filters, output_wid, output_hgt)

    def forward(self, ins):
        self.ins = ins
        
        output_dims = self.calc_output_dims()
        self.vals = np.zeros(output_dims)
        self.outs = np.zeros(output_dims)

        # Automatically assuming "valid" padding (no padding)
        # This part was copied over from my main project
        for filter in range(output_dims[0]):
            for outrow in range(output_dims[1]):
                for outcol in range(output_dims[2]):
                    for kerrow in range(self.kernel_dims[0]):
                        for kercol in range(self.kernel_dims[1]):
                            in_row = outrow * self.stride_dims[1] + kerrow
                            in_col = outcol * self.stride_dims[0] + kercol
                            self.vals[filter][outrow][outcol] += (
                                self.weights[filter][kerrow][kercol] * self.ins[in_row][in_col]
                            )
        
        # Calculate outputs after activation
        self.outs = self.activation(self.vals)

        # Cache the change of outputs with respect to values before activation
        self.dOut_dVals = self.deriv_activation(self.vals)

        # print("inputs\n", ins)
        # print("kernel\n", self.weights)
        # print("vals\n", self.vals)
        # print("outs\n", self.outs)
        return self.outs

    def backwards(self, pdL_pdOut):
        '''
        Compute the gradients for the weights in this layer.
        pdL_pdOut should have the same dimensions as the output of the forward op.
        This should be (num_filters, out_width, out_height) <- not actual variable names
        '''

        # pdL_pdOut should have the same dimensions as the original forwarded output.
        # pdL_pdVals should also have the same dimensions.
        pdL_pdVals = np.multiply(pdL_pdOut, self.dOut_dVals)
        # Using ∂L / dval, we can calculate the final gradients for the weights
        # in this layer.

        # The final gradient matrix should have the same dimensions as the kernel.

        # Convoluting the pdL_pdVals matrix on the original inputs should yield the
        # gradients.
        # 
        # For example given a stride of (2, 2) and:
        # Input matrix = 
        # in0,0  in0,1  in0,2  in0,3
        # in1,0  in1,1  in1,2  in1,3
        # in2,0  in2,1  in2,2  in2,3
        # in3,0  in3,1  in3,2  in3,3
        #
        # Kernel = 
        # k0,0  k0,1
        # k1,0  k1,1
        #
        # Then computed values =
        # v0,0  v0,1
        # v1,0  v1,1
        # Where:
        # v0,0 = k0,0 * in0,0 + k0,1 * in0,1 + k1,0 * in1,0 + k1,1 * in1,1
        # v0,1 = k0,0 * in0,2 + k0,1 * in0,3 + k1,0 * in1,2 + k1,1 * in1,3
        # v1,0 = k0,0 * in2,0 + k0,1 * in2,1 + k1,0 * in3,0 + k1,1 * in3,1
        # v1,1 = k0,0 * in2,2 + k0,1 * in2,3 + k1,0 * in3,1 + k1,1 * in3,3
        #
        # Then the partial derivatives are calculated by
        # ∂L / ∂k0,0 =  ∂L / ∂v0,0  *  ∂v0,0 / ∂k0,0 +
        #               ∂L / ∂v0,1  *  ∂v0,1 / ∂k0,0 +
        #               ∂L / ∂v1,0  *  ∂v1,0 / ∂k0,0 +
        #               ∂L / ∂v1,1  *  ∂v1,1 / ∂k0,0
        #            =  ∂L / ∂v0,0  *  in0,0 + ∂L / ∂v0,1  *  in0,2 +
        #               ∂L / ∂v1,0  *  in2,0 + ∂L / ∂v1,1  *  in2,2
        # 
        # ∂L / ∂k0,1 =  ∂L / ∂v0,0  *  in0,1 + ∂L / ∂v0,1  *  in0,3 +
        #               ∂L / ∂v1,0  *  in2,1 + ∂L / ∂v1,1  *  in2,3
        # Due to the stride, it gets a little complicated.
        # It appears to be a convolution operation where the kernel (the global gradient matrix):
        # ∂L / ∂v0,0   ∂L / ∂v0,1
        # ∂L / ∂v1,0   ∂L / ∂v1,1
        # is spaced out in each axis depending on whatever the original stride was.
        self.grads = np.zeros((self.num_filters, self.kernel_dims[0], self.kernel_dims[1]))
        
        # grad_row_and grad_col refer to the row and col of the final gradient matrix
        # that will be substracted from this layer's weights. It should be the same shape as
        # this layer's weights (the original kernel).
        for filter in range(self.grads.shape[0]):
            for grad_row in range(self.grads.shape[1]):
                for grad_col in range(self.grads.shape[2]):
                    # glob_grad_row and glob_grad_col refer to the row and col of the global
                    # gradient matrix (pdL_pdVals).
                    for glob_grad_row in range(pdL_pdVals.shape[1]):
                        for glob_grad_col in range(pdL_pdVals.shape[2]):
                            in_row = grad_row + self.stride_dims[1] * glob_grad_row
                            in_col = grad_col + self.stride_dims[0] * glob_grad_col
                            self.grads[filter][grad_row][grad_col] += (
                                pdL_pdVals[filter][glob_grad_row][glob_grad_col] * self.ins[in_row][in_col]
                            )

        # print("pdL_pdOuts\n", pdL_pdOut)
        # print("dOut_dVals\n", self.dOut_dVals)
        # print("pdL_pdVals\n", pdL_pdVals)
        # print("inputs\n", self.ins)
        # print("this layer grads\n", self.grads)

        # Now need to calculate the global gradient for the upstream.
        # This is a again a convolution. Using a new example with stride(2, 2):
        # The global gradient matrix is:
        # ∂L/∂v0,0  ∂L/∂v0,1
        # ∂L/∂v1,0  ∂L/∂v1,1
        # The kernel is:
        # k0,0  k0,1  k0,2
        # k1,0  k1,1  k1,2
        # k2,0  k2,1  k2,2
        # The inputs are: 
        # in0,0  in0,1  in0,2  in0,3  in0,4
        # in1,0  in1,1  in1,2  in1,3  in1,4
        # in2,0  in2,1  in2,2  in2,3  in2,4
        # in3,0  in3,1  in3,2  in3,3  in3,4
        # in4,0  in4,1  in4,2  in4,3  in4,4
        # The change in loss with respect to each input are:
        # ∂L/in0,0 = ∂L/∂v0,0 * k0,0
        # ∂L/in0,1 = ∂L/∂v0,0 * k0,1
        # ∂L/in0,2 = ∂L/∂v0,0 * k0,2 + ∂L/∂v0,1 * k0,0
        # ∂L/in0,3 = ∂L/∂v0,1 * k0,1
        # ∂L/in0,4 = ∂L/∂v0,1 * k0,2
        # ∂L/in1,0 = ∂L/∂v0,0 * k1,0
        # ∂L/in1,1 = ∂L/∂v0,0 * k1,1
        # ∂L/in1,2 = ∂L/∂v0,0 * k1,2 + ∂L/∂v0,1 * k1,0
        # ∂L/in1,3 = ∂L/∂v0,1 * k1,1
        # ∂L/in1,4 = ∂L/∂v0,1 * k1,2
        # ∂L/in2,0 = ∂L/∂v0,0 * k2,0 + ∂L/∂v1,0 * k0,0
        # ∂L/in2,1 = ∂L/∂v0,0 * k2,1 + ∂L/∂v1,0 * k0,1
        # ∂L/in2,2 = ∂L/∂v0,0 * k2,2 + ∂L/∂v0,1 * k2,0 + ∂L/∂v1,0 * k0,0 + ∂L/∂v1,1 * k0,0
        # ∂L/in2,3 = ∂L/∂v0,1 * k2,1 + ∂L/∂v1,1 * k0,1
        # ...
        # This is defined by a "full" convolution of the gradient matrix flipped 180°,
        # performed on the kernel.
        # ∂L/∂v0,0  ∂L/∂v0,1
        # ∂L/∂v1,0  ∂L/∂v1,1
        # flipped 180:
        # ∂L/∂v1,1 ∂L/∂v1,0
        # ∂L/∂v0,1 ∂L/∂v0,0
        # But note that the flipped gradient matrix has to be spaced out by the stride again

        # For now I will assume only one conv layer due to time constraints,
        # and not implement this backward pass
        pdL_pdOut_pred = np.zeros(self.input_dims)

        return pdL_pdOut_pred
    
    def update_params(self, dampening):
        eta = 0.5 * dampening
        self.weights = self.weights - eta * self.grads


class DenseLayer:
    def __init__(self, pred_layer: "DenseLayer", activation_fn, deriv_activation_fn, node_count, input_count) -> None:
        self.pred_layer = pred_layer

        self.num_nodes = node_count
        self.weights = np.random.randn(input_count, self.num_nodes)
        self.vals = np.empty(self.num_nodes)
        self.outs = np.empty(self.num_nodes)
        self.dOut_dVals = np.empty(self.num_nodes)
        self.activation = activation_fn
        self.deriv_activation = deriv_activation_fn

    def forward(self, ins):
        '''\"ins\" should be the outputs (after activation) of the previous layer.'''
        # Cache the inputs, they will be used to calculate gradients in backpropagation.
        self.ins = ins

        # Let:  n = number of nodes in the previous layer
        #       ins = [x1, x2, ..., xn]

        # The dot product ins • weights = a vector of the weighted sums
        # of each node. i.e. a vector of val = f(w1 in_1 + ... + wn in_n).

        # "ins" is a row vector, and each column of self.weights is a node's
        # set of weights. The result is a row vector of vals.
        self.vals = np.dot(ins, self.weights)

        # Apply the activation function φ on each node's weighted sum to get
        # the pass-forward value of each node. For each node, calculate:
        # out = φ(val)
        self.outs = self.activation(self.vals)

        # Cache the derivative of the output value (the value after applying
        # activation function), with respect to the weighted sum
        # (the value before applying activation function).
        # dout / dval = φ'(val)
        self.dOut_dVals = self.deriv_activation(self.vals)

        # Return the pass-forward value
        return self.outs

    def backwards(self, pdL_pdOut):
        ''' 
        Perform backpropagation. pdL_pdOut is the change in loss
        with respect to this layer's outputs. i.e. the output of
        each node after the activation function is applied.
        '''
        
        # We are given the partial derivative of the Loss with respect to
        # this layer's outputs: pdL_pdOut = ∂L / ∂out

        # We already cached the derivative of this layer's outputs with
        # respect to its values before activation: dout/ dval

        # For each node, calculate: ∂L / ∂val = ∂L / ∂out * dout / dval
        pdL_pdVals = np.multiply(pdL_pdOut, self.dOut_dVals)

        # Now we can compute ∂L / ∂w , (which are the gradients) for each
        # weight w_i in this layer.
        # ∂L / ∂w_i  =  ∂L / ∂val * ∂val / ∂w_i  =   ∂L / ∂val * in_i
        # This is because val = w1 * in_1 + w2 * in_2 + ... + wn in_n
        # So ∂val / ∂w_i = in_i

        # pdL_pdVals and ins are both row vectors.
        # We want one column resulting from each value of pdL_pdVals multiplied
        # by the inputs. i.e. an outer product.
        # np.outer(a, b) does the following:
        # [[a0b0 a0b1 ... a0bN]
        #  [a1b0 a1b1 ... a1bN]
        #          . . .
        #  [aMb0 aMb1 ... aMbN]]

        # So that means pdL_pdVals should be the second argument.
        self.grads = np.outer(self.ins, pdL_pdVals)
        # print(self.grads)
        # The result is that each column are the gradients for one node.
        # This follows the original data format for self.weights, where each
        # column stores the weights of one node.

        # Now compute the ∂L / ∂out_i for the previous layer. Subscript
        # "i" will represent values of the previous layer, while "j" will
        # represent values in the current layer.

        # ∂L / ∂out_i = Σ (∂L / ∂val_j * ∂val_j / ∂out_i)
        # summation over each node j in this layer.
        # ∂val_j / ∂out_i is just the weight, because:
        # val_j = out_i_1 * w_j_1 + out_i_2 * w_j_2

        # This has to be calculated for each out_i of the previous layer.
        # Let w_x_y denote the y-th weight of node x in the current layer.
        # Weights should look like this
        # [[w_1_1  w_2_1  ...  w_3_N],
        #  [w_1_2  w_2_2  ...  w_3_N],
        #          . . .
        #  [w_M_1  w_M_2  ...  w_M_N]]

        # This below will multiply each element in pdL_pdVals element-wise in
        # each row of self.weights. This means all of column 1's values
        # will be multiplied by the first element of pdL_pdVals
        # This makes the sum of each row equal to the change in loss
        # with respect to the change in output for the previous layer
        # reduce sum, adding elements of the same row together.
        pdL_pdOut_pred = np.sum(np.multiply(self.weights, pdL_pdVals), axis = 1)
        return pdL_pdOut_pred

    def update_params(self, dampening):
        eta = 0.5 * dampening
        self.weights = self.weights - eta * self.grads

    def print(self, detail = False):
        pass
        # print("================================")
        # print("           Dense Layer          ")
        # if detail:
        #     print("\nData: ")
        #     for i in range(len(self.nodes)):
        #         end = ', ' if i < len(self.nodes) - 1 else '\n'
        #         # self.nodes[i].print(end)
        #     print()
        # print("Node count:", len(self.nodes))
        # print("================================")

class InputLayer:
    def __init__(self, node_count) -> None:
        pass
        # '''Shape should be (width [,height])'''
        # Noramlize inputs
        # self.outs = self.normalize(data)
        self.num_nodes = node_count
        # self.nodes = [InputNode(datum) for datum in self.outs]

    def normalize(self, data):
        dat_max = np.max(data)
        dat_min = np.min(data)
        dat_range = dat_max - dat_min
        if dat_range != 0:
            return (data - dat_min) / dat_range
        return data - dat_min

    def forward(self, ins):
        return self.normalize(ins)

    def print(self, detail = False):
        pass
    #     print("================================")
    #     print("           Input Layer          ")
    #     if detail:
    #         print("\nData: ")
    #         for i in range(len(self.nodes)):
    #             end = ', ' if i < len(self.nodes) - 1 else '\n'
    #             self.nodes[i].print(end)
    #         print()
    #     print("Data count:", len(self.nodes))
    #     print("================================")

def mean_squared_error(y_true, y_pred):
    if (len(y_true) != len(y_pred)):
        print("Error: mean_squared_error(): lengths do not match.")

    sq_err_sum = 0
    for i in range(len(y_true)):
        sq_err_sum += (y_true[i] - y_pred[i]) ** 2

    return sq_err_sum / len(y_true)

def deriv_mean_squared_error(y_true, y_pred):
    if (len(y_true) != len(y_pred)):
        print("Error: deriv_mean_squared_error(): lengths do not match.")
    # MSE is:
    # 1 / N * Summation(  (y_true_i - y_pred_i) ^ 2  )
    # change of MSE with respect to each element in the predictions is:
    # 2 / N *  -(y_true_i - y_pred_i)
    two_over_N = 2 / len(y_pred)
    result = [None] * len(y_pred)
    for i in range(len(y_pred)):
        result[i] = two_over_N * -(y_true[i] - y_pred[i])
    return result

# Load the metadata csv file. It contains information about
# the input filenames and their labels.
# suite_id = []
# sample_id = []
# code = []
# value = []
# with open('../cpsc-479-p2/data/chinese_mnist/chinese_mnist.csv') as csv_file:
#     csv_reader = csv.reader(csv_file)
#     header_line = True
#     for row in csv_reader:
#         # Skip header
#         if header_line:
#             header_line = False
#             continue
#         suite_id.append(row[0])
#         sample_id.append(row[1])
#         code.append(row[2])
#         value.append(row[3])

# # Construct network
# in_layer = InputLayer(64 * 64)
# conv_layer = Conv2DLayer((64, 64), sigmoid, sigmoid, (7, 7), (3, 3), 32)
# dense_layer = DenseLayer(conv_layer, sigmoid, deriv_sigmoid, 128, 32 * 20 * 20)
# out_layer = DenseLayer(dense_layer, sigmoid, deriv_sigmoid, 15, 128)

# def value_to_class(val):
#     if val == 0:
#         return [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
#     if val == 1:
#         return [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
#     if val == 2:
#         return [0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
#     if val == 3:
#         return [0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
#     if val == 4:
#         return [0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
#     if val == 5:
#         return [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0]
#     if val == 6:
#         return [0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0]
#     if val == 7:
#         return [0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0]
#     if val == 8:
#         return [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0]
#     if val == 9:
#         return [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0]
#     if val == 10:
#         return [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0]
#     if val == 100:
#         return [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0]
#     if val == 1000:
#         return [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0]
#     if val == 10000: # 10k
#         return [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0]
#     if val == 100000000: # hundred mil
#         return [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1]

# for i in range(len(suite_id)):
#     img_input = cv2.imread(
#         f"../cpsc-479-p2/data/chinese_mnist/data/input_{suite_id[i]}_{sample_id[i]}_{code[i]}.jpg",
#         cv2.IMREAD_GRAYSCALE
#     )
#     # feed-forward
#     out1 = in_layer.forward(img_input)
#     out2 = conv_layer.forward(out1)
#     out3 = dense_layer.forward(out2.reshape(32 * 20 * 20)) # 32 activation maps of size 20 * 20
#     out4 = out_layer.forward(out3)

#     # calc loss
#     expected = value_to_class(int(value[i]))
#     curr_mse = mean_squared_error(expected, out4)
#     print("mse:", curr_mse)

#     # last 15 check output
#     if i > 14984:
#         print(f"input_{suite_id[i]}_{sample_id[i]}_{code[i]}.jpg was class\n", value_to_class(value[i]))
#         print("And was predicted to be\n", out4)

#     # backprop
#     back1 = deriv_mean_squared_error(expected, out4)
#     back2 = out_layer.backwards(back1)
#     back3 = dense_layer.backwards(back2)
#     back4 = conv_layer.backwards(back3.reshape(32, 20, 20))
#     out_layer.update_params(1)
#     dense_layer.update_params(1)
#     conv_layer.update_params(1)




# Everything below is for testing just the dense layers
# exit()

# Given [a, b, c, d] as input
# We want to predict the following condition:
# [a && (!b || !c) && !d , !a && d && b && a]
def generate_input():
    return [random.randint(0, 1) for i in range(4)]
def generate_y_true(invec):
    return [
        invec[0] and ((not invec[1]) or (not invec[2])) and (not invec[3]),
        (not invec[0]) and invec[1] and invec[2] and invec[3]
    ]
    #         (not invec[0]) and invec[1] and invec[2] and invec[3]]
    # if (invec[0]) and (not invec[1]) and (not invec[2]):
    #     return [1, 0, 0]
    # elif (not invec[0]) and (invec[1]) and (not invec[2]):
    #     return [0, 1, 0]
    # else:
    #     return [0, 0, 1]

my_test_case1 = [0, 1, 1, 1]
my_test_case_ans1 = [0, 1]
my_test_case2 = [1, 0, 0, 0]
my_test_case_ans2 = [1, 0]
my_test_case3 = [1, 1, 0, 0]
my_test_case_ans3 = [1, 0]

in_layer = InputLayer(4)
hidden_layer = DenseLayer(in_layer, sigmoid, deriv_sigmoid, 4, 4)
out_layer = DenseLayer(hidden_layer, sigmoid, deriv_sigmoid, 2, 4)


# Counters
total_iters = 0
iters_this_epoch = 0
epochs = 0

# Totals
iters_per_epoch = 5000
epochs_per_session = 5
max_iters = iters_per_epoch * epochs_per_session

# Store
avg_epoch_loss = np.zeros(iters_per_epoch)

def show_only_max(x):
    is_max = np.array(x.max() == x, dtype = int)
    is_uniform = True
    first = is_max[0]
    for num in is_max:
        if num != first:
            is_uniform = False
    if is_uniform:
        return np.zeros(is_max.shape)
    else:
        return is_max

def run_test():
    np.set_printoptions(precision = 3, suppress = True)
    out1 = in_layer.forward(my_test_case1)
    out2 = hidden_layer.forward(out1)
    out3 = out_layer.forward(out2)
    print("Test 1 input:", my_test_case1)
    print(f"Test 1 prediction:", out3)
    print("Test 1 expected:", my_test_case_ans1)
    print()
    out1 = in_layer.forward(my_test_case2)
    out2 = hidden_layer.forward(out1)
    out3 = out_layer.forward(out2)
    print("Test 2 input:", my_test_case2)
    print(f"Test 2 prediction:", out3)
    print("Test 2 expected:", my_test_case_ans2)
    print()
    out1 = in_layer.forward(my_test_case3)
    out2 = hidden_layer.forward(out1)
    out3 = out_layer.forward(out2)
    print("Test 3 input:", my_test_case3)
    print(f"Test 3 prediction:", out3)
    print("Test 3 expected:", my_test_case_ans3)
    print()

while True:
    # response = input("Press enter to iterate. Enter q to quit: ").lower()
    # if response == 'q' or response == 'quit':
        # break

    # if response == 'p' or response == 'predict':
    #     out1 = in_layer.forward(my_test_case)
    #     out2 = out_layer.forward(out1)
    #     print("My test case was", my_test_case)
    #     print("Predicted was", out2)
    #     print("Expected answer was", my_test_case_ans)
    #     continue

    # Set new training input
    input_x = generate_input()

    # feed forward
    out1 = in_layer.forward(input_x)
    out2 = hidden_layer.forward(out1)
    out3 = out_layer.forward(out2)

    y_true = generate_y_true(out1)
    curr_mse = mean_squared_error(y_true, out3)
    avg_epoch_loss[iters_this_epoch] = curr_mse

    # backprop
    back1 = deriv_mean_squared_error(y_true, out3)
    back2 = out_layer.backwards(back1)
    back3 = hidden_layer.backwards(back2)
    out_layer.update_params(float(total_iters) / max_iters)
    hidden_layer.update_params(float(total_iters) / max_iters)

    total_iters += 1
    iters_this_epoch += 1

    if iters_this_epoch == iters_per_epoch:
        iters_this_epoch = 0
        epochs += 1
        run_test()
        print(f"Epoch {epochs} complete. Average loss: {avg_epoch_loss.mean()}.\n\n")

    if epochs == epochs_per_session:
        break
