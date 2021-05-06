import math
import random

import numpy as np

def sigmoid(x):
    '''Sigmoid activation function.'''

    # Experimentally found out that there is overflow if exp(709.9)
    # Make sure -x is less than 709
    if x < -709:
        x = -709
    return 1 / (1 + math.exp(-x))

def deriv_sigmoid(x):
    '''First derivative of the sigmoid function'''
    s = sigmoid(x)
    return s * (1 - s)

class DenseNode:
    def __init__(self, pred_layer, activation_fn, deriv_activation_fn) -> None:
        self.out = 0
        self.val = 0
        self.pdL_pdOut = 0
        self.pdL_pdVal = 0
        self.pdOut_pdVal = 0
        self.pred = [node for node in pred_layer.nodes]
        # Random weight initialization between [-0.05, 0.05)
        self.weights = [
            (random.random() * 0.1 - 0.05) for i in range(len(self.pred))
        ]
        self.grads = [0 for i in range(len(self.pred))]

        self.acf = activation_fn
        self.deriv_acf = deriv_activation_fn

    def set_pdL_pdOut(self, pdL_pdOut):
        '''
        This should only be used on the last layer of the network
        to set the loss with respect to the output layer.
        '''
        self.pdL_pdOut = pdL_pdOut

    #### 1. feed-forward
    def forward(self):
        '''
        Needs: pred.out
        Creates: self.val self.out
        Calculate the internal value and output value after activation
        '''
        self.val = 0
        for i in range(len(self.pred)):
            self.val += self.pred[i].out * self.weights[i]
        self.out = self.acf(self.val)

    #### 2. Calculate pdOut / pdVal
    def calc_pdOut_pdVal(self):
        '''
        Calculate this neuron's change in output value (after activation)
        with respect to its internal value (right before activation)
        '''

        if (self.acf == sigmoid):
            # Since sigmoid's first derivative is just sigmoid(x) * (1 - sigoid(x)),
            # an efficient shortcut is just to use this neuron's self.out value
            self.pdOut_pdVal = self.out * (1 - self.out)
        else:
            self.pdOut_pdVal = self.deriv_acf(self.val)

    #### 3. Calculate pdL / pdVal
    def calc_pdL_pdVal(self):
        '''
        Calculate the so called delta value, which is the change in Loss with
        respect to this layer's output after activation, times the change in
        this layer's output after activation with respect to the value before
        activation.
        '''

        # pdL_pdVal is the delta (pdL / pdVal)
        self.pdL_pdVal = self.pdL_pdOut * self.pdOut_pdVal

    # IMPORTANT
    def clear_pdL_pdOut(self):
        ''' Clear the delta values of the predecessor layer before adding deltas.'''
        self.pdL_pdOut = 0

    def backprop_delta(self):
        for i in range(len(self.pred)):
            # For each predecessor neuron that feeds into this current one
            # Add the change of loss with respect to this node's value
            # times the change of this node's value with respect to the previous
            self.pred[i].pdL_pdOut += self.pdL_pdVal * self.weights[i]

    def calc_grad(self):
        '''
        Calculate the gradients. This depends on the delta (pdL_pd_val).
        '''
        for i in range(len(self.grads)):
            self.grads[i] = self.pdL_pdVal * self.pred[i].out

    def update_params(self):
        '''
        Currently uses stochastic gradient descent. 
        '''
        eta = 0.2
        for i in range(len(self.weights)):
            self.weights[i] -= eta * self.grads[i]


class DenseLayer:
    def __init__(self, pred_layer, activation_fn, deriv_activation_fn, node_count) -> None:
        self.pred_layer = pred_layer
        self.nodes = [
            DenseNode(pred_layer, activation_fn, deriv_activation_fn) for i in range(node_count)
        ]

    def forward(self):
        for node in self.nodes:
            node.forward()

    def clear_pdL_pdOut(self):
        for node in self.nodes:
            node.clear_pdL_pdOut()

    def backwards(self):
        # First calculate some intermediate partial derivatives
        for node in self.nodes:
            node.calc_pdOut_pdVal()
            node.calc_pdL_pdVal()

        # Clear the previous layer's deltas
        if not isinstance(self.pred_layer, InputLayer):
            self.pred_layer.clear_pdL_pdOut()

        # Backprop deltas and then train this layer's weights
        for node in self.nodes:
            node.backprop_delta()
            node.calc_grad()
            node.update_params()

    def set_pdL_pdOut(self, pdL_pdOut):
        for i in range(len(pdL_pdOut)):
            self.nodes[i].set_pdL_pdOut(pdL_pdOut[i])

    def grab_output(self):
        '''
        Should only be used if this is an output layer.
        Normally the data resides in each node. This grabs them
        into one vector for easier calculation of loss.
        '''
        return [node.out for node in self.nodes]

    def print_nums(self):
        print("Dense layer weights:")
        for node in self.nodes:
            for wt in node.weights:
                print(wt, end=", ")
        print()
        print("Dense layer grads:")
        for node in self.nodes:
            for grad in node.grads:
                print(grad, end=", ")
        print()

    def print(self, detail = False):
        print("================================")
        print("           Dense Layer          ")
        if detail:
            print("\nData: ")
            for i in range(len(self.nodes)):
                end = ', ' if i < len(self.nodes) - 1 else '\n'
                # self.nodes[i].print(end)
            print()
        print("Node count:", len(self.nodes))
        print("================================")


class InputNode:
    def __init__(self, datum) -> None:
        self.pdL_pdOut = 0 # This value is not used, but the dense layer will try to set it
        self.out = datum

    def print(self, end=''):
        print(self.out, end=end)

    def set_new_data(self, new_data):
        self.out = new_data

class InputLayer:
    def __init__(self, data) -> None:
        # Noramlize inputs
        data = self.normalize(data)
        self.nodes = [InputNode(datum) for datum in data]

    def normalize(self, data):
        dat_max = max(data)
        dat_min = min(data)
        dat_range = float(dat_max - dat_min)
        for i in range(len(data)):
            data[i] -= dat_min
            if dat_max - dat_min != 0:
                data[i] /= dat_range
        return data

    def set_new_data(self, new_data):
        for i in range(len(self.nodes)):
            new_data = self.normalize(new_data)
            self.nodes[i].set_new_data(new_data[i])

    def print(self, detail = False):
        print("================================")
        print("           Input Layer          ")
        if detail:
            print("\nData: ")
            for i in range(len(self.nodes)):
                end = ', ' if i < len(self.nodes) - 1 else '\n'
                self.nodes[i].print(end)
            print()
        print("Data count:", len(self.nodes))
        print("================================")

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


# Given [a, b, c, d] as input
# We want to predict [1, 0, 0, 0]  if  a >= b  c >= d
# We want to predict [0, 1, 0, 0]  if  a < b   c >= d
# We want to predict [0, 0, 1, 0]  if  a >= b  c < d
# We want to predict [0, 0, 0, 1]  if  a < b   c < d
def generate_input():
    return [random.randint(0, 9) for i in range(4)]
def generate_y_true(invec):
    if (invec[0] >= invec[1] and invec[2] >= invec[3]):
        return [1, 0, 0, 0]
    if (invec[0] < invec[1] and invec[2] >= invec[3]):
        return [0, 1, 0, 0]
    if (invec[0] >= invec[1] and invec[2] < invec[3]):
        return [0, 0, 1, 0]
    if (invec[0] < invec[1] and invec[2] < invec[3]):
        return [0, 0, 0, 1]

my_test_case = [4, 5, 4, 1]
my_test_case_ans = [0, 1, 0, 0]

in_layer = InputLayer([0] * 4)
hidden_layer1 = DenseLayer(in_layer, sigmoid, deriv_sigmoid, 20)
hidden_layer2 = DenseLayer(hidden_layer1, sigmoid, deriv_sigmoid, 20)
out_layer = DenseLayer(hidden_layer2, sigmoid, deriv_sigmoid, 4) # 4 classes

while True:
    response = input("Press enter to iterate. Enter q to quit: ").lower()
    if response == 'q' or response == 'quit':
        break

    if response == 'p' or response == 'predict':
        in_layer.set_new_data(my_test_case)
        hidden_layer1.forward()
        hidden_layer2.forward()
        out_layer.forward()
        print("My test case was", my_test_case)
        print("Predicted was", out_layer.grab_output())
        print("Expected answer was", my_test_case_ans)
        hidden_layer1.print_nums()
        continue

    # Set new training input
    input_x = generate_input()
    in_layer.set_new_data(input_x)
    in_layer.print(True)

    # feed forward
    hidden_layer1.forward()
    hidden_layer1.print_nums()
    hidden_layer2.forward()
    out_layer.forward()

    y_pred = out_layer.grab_output()
    print("The estimate was:", y_pred)
    y_true = generate_y_true(input_x)
    curr_mse = mean_squared_error(y_true, y_pred)
    print("The mse was:", curr_mse)

    # backprop
    deriv_mse = deriv_mean_squared_error(y_true, y_pred)
    out_layer.set_pdL_pdOut(deriv_mse)
    out_layer.backwards()
    hidden_layer2.backwards()
    hidden_layer1.backwards()