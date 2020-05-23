#! /usr/bin/python3

import numpy as np
import math

class BPNN2():
    '''
    Initial BP with empty
    '''
    def __init__(self):
        self.input_n           = 0
        self.hidden_n          = 0
        self.output_n          = 0
        self.input_cells       = []
        self.hidden_cells      = []
        self.output_cells      = []
        self.input_weights     = []
        self.output_weights    = []
        self.input_correction  = []
        self.output_correction = []
        self.active_method     = 'sigmoid'
        self.learn_rate        = 0.5
        self.correct_belta     = 0.1
        self.max_iters         = 10000

    def active(self, x):
        if self.active_method == 'sigmoid':
            return 1.0 / (1.0 + math.exp(-x))
        else:
            return 1.0 / (1.0 + math.exp(-x))

    def back_active(self, x):
        if self.active_method == 'sigmoid':
            return x * (1-x)
        else:
            return x * (1-x)

    def setup(self, input_n, hidden_n, output_n, active_method = 'sigmoid', learn_rate = 0.05, correct_belta = 0.1, max_iters = 10000):
        # Set up number of cells
        self.input_n  = input_n + 1 # Bias
        self.hidden_n = hidden_n
        self.output_n = output_n

        # Set up Cells
        self.input_cells   = [1.0] * self.input_n
        self.hidden_cells  = [1.0] * self.hidden_n
        self.output_cells  = [1.0] * self.output_n

        # Set up and initial weights
        self.input_weights  = np.random.uniform(low=-0.1, high=0.1, size=[self.input_n, self.hidden_n])
        self.output_weights = np.random.uniform(low=-0.1, high=0.1, size=[self.hidden_n, self.output_n])

        # Set up correction matrix
        self.input_correction  = np.zeros([self.input_n, self.hidden_n])
        self.output_correction = np.zeros([self.hidden_n, self.output_n])

        # Set Active mothod
        self.active_method = active_method

        # Set parameters
        self.learn_rate    = learn_rate
        self.correct_belta = correct_belta
        self.max_iters     = max_iters

    def forward(self, data):
        # Active input 
        for i in range(self.input_n - 1):
            self.input_cells[i] = data[i]

        # Active hidden
        for h in range(self.hidden_n):
            sum = 0
            for i in range(self.input_n):
                sum += self.input_cells[i] * self.input_weights[i][h]
            self.hidden_cells[h] = self.active(sum)

        # Active output layer
        for o in range (self.output_n):
            sum = 0
            for h in range(self.hidden_n):
                sum += self.hidden_cells[h] * self.output_weights[h][o] 
            # For regression
            self.output_cells[o] = sum

            # For classfication
            # self.output_cells[o] = self.active(sum)

        return self.output_cells[:]

    def back_propagate(self, data, label):
        self.forward(data)

        output_error = [0.0]  * self.output_n
        for o in range(self.output_n):
            error           = label[o] - self.output_cells[o]
            output_error[o] = error    # No active

        hidden_error = [0.0] * self.hidden_n
        for h in range(self.hidden_n):
            error           = 0
            for o in range(self.output_n):
                error += output_error[o] * self.output_weights[h][o]
            hidden_error[h] = self.back_active(self.hidden_cells[h]) * error

        # Update output weights
        for h in range(self.hidden_n):
            for o in range(self.output_n):
                change = output_error[o] * self.hidden_cells[h]
                self.output_weights[h][o]   += self.learn_rate * change + self.correct_belta * self.output_correction[h][o]
                self.output_correction[h][o] = change
        
        # Update input weights
        for i in range(self.input_n):
            for n in range(self.input_n):
                change = hidden_error[h] * self.input_cells[i]
                self.input_weights[i][h]   += self.learn_rate * change + self.correct_belta * self.input_correction[i][h]
                self.input_correction[i][h] = change

        # Get Global error
        error = 0.0
        for l in range(np.shape([label])[0]):
            error += 0.5 * (label[l] - self.output_cells[l]) ** 2

        return error
    
    def train(self, data_mat, label_mat):
        errors = []
        for j in range(self.max_iters):
            error = 0
            for i in range(np.shape(data_mat)[0]):
                label  = label_mat[i]
                data   = data_mat[i]
                error += self.back_propagate(data, label)

            errors.append(error)

        return errors

    def predict(self, tests):
        results = []
        for test_data in tests:
            result = self.forward(test_data)
            results.append(result)

        return results