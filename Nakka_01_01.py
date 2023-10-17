# Nakka, Tulasi
# 1001_928_971
# 2023_09_24
# Assignment_01_01

import numpy as np

# Assume that the activation functions for all the layers are sigmoid.
def sigmoid(x):
    return 1/(1 + np.exp(-x))

# Use MSE to calculate error.
def MSE(y_true,y_pred):
    return np.mean((y_true - y_pred)**2)

# Weights for the network
def Weight_Matrix(layers, X_train,seed):
    # Reseed the random number generator when initializing weights for each layer.
    # i.e., Initialize the weights for each layer by:
    # np.random.seed(seed)
    # np.random.randn()
    Weights = []
    for i in range(len(layers)):
        np.random.seed(seed)
        if i == 0:
            weight_shape = (layers[i], X_train.shape[0] + 1)
        else:
            weight_shape = (layers[i], layers[i - 1] + 1)
        weight_matrix = np.random.randn(weight_shape[0],weight_shape[1])
        Weights.append(weight_matrix)
    return Weights

#We need to add Ones to make sure matrix size remains same after adding bias to weight matrix. If not getting errors.
# Bias should be included in the weight matrix in the first column.
def add_bias(X):
    ones_row = np.ones((1, X.shape[1]))
    return np.vstack((ones_row,X))

def actual_network_output(wt, x_train, layers):
    Output_list = []
    for i in range(len(layers)):
        input_data = x_train if len(Output_list) == 0  else add_bias(Output_list[-1])
        output = sigmoid(np.dot(wt[i], input_data))
        Output_list.append(output)
    return Output_list[-1]

def Calculate_average_mse_error(new_weights, x_train, y_train, layers):
    y_train_new = actual_network_output(new_weights, x_train, layers)
    return MSE(y_train_new, y_train)


def multi_layer_nn(X_train,Y_train,X_test,Y_test,layers,alpha,epochs,h=0.00001,seed=2):
    # This function creates and trains a multi-layer neural Network
    Weights = Weight_Matrix(layers, X_train, seed)
    MSE = []
    updated_X_train, updated_Y_train = add_bias(X_train), add_bias(X_test)

    for i in range(epochs):
        for j in range(X_train.shape[1]):
            x_train, y_train = updated_X_train[:, j:j+1], Y_train[:, j:j+1]
            Weights_new = [np.zeros_like(layer) for layer in Weights]

            for w1 in range(len(Weights)):
                for w2 in range(len(Weights[w1])):
                    for w3 in range(len(Weights[w1][w2])):
                        # To use later
                        original_weight = Weights[w1][w2][w3]

                        # use centered difference approximation to calculate partial derivatives.
                        # (f(x + h)-f(x - h))/2*h

                        #calculating w + h  
                        Weights[w1][w2][w3] = original_weight + h
                        x_plus_h = Calculate_average_mse_error(Weights, x_train,y_train, layers)

                        #calculating w - h
                        Weights[w1][w2][w3] = original_weight - h
                        x_minus_h = Calculate_average_mse_error(Weights, x_train,y_train, layers)

                        #calculating partial derivative
                        diff = (x_plus_h - x_minus_h) / (2 * h)

                         # Use gradient descent for adjusting the weights.
                         #i.e W_adjust = W - W' *  aplha
                        final_weight = original_weight - diff * alpha
                        Weights_new[w1][w2][w3] = final_weight
                        Weights[w1][w2][w3] = original_weight
            
            # The first element of the return list should be a list of weight matrices.
            # Each element of the list corresponds to the weight matrix of the corresponding layer.
            Weights = Weights_new
        
        # The second element should be a one dimensional array of numbers
        # representing the average mse error after each epoch. Each error should
        # be calculated by using the X_test array while the network is frozen.
        # This means that the weights should not be adjusted while calculating the error.
        average_mse_error = []
        for i in range(X_test.shape[1]):
            x_test, y_test = updated_Y_train[:, i:i+1], Y_test[:, i:i+1]
            average_mse_error.append(Calculate_average_mse_error(Weights, x_test, y_test, layers))
        MSE.append(np.mean(average_mse_error))

    # The third element should be a two-dimensional array [output_dimensions,nof_test_samples]
    # representing the actual output of network when X_test is used as input.
    W = Weights
    err = MSE 
    Out = actual_network_output(W, updated_Y_train, layers)

    return [W, err, Out]






    