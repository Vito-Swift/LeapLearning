import numpy as np
import random as rd

# training data (input / output)
training_inputs = [
    [0.63, 0.56, 0.68],
    [0.55, 0.44, 0.65],
    [0.46, 0.78, 0.64],
    [0.37, 0.55, 0.44],
    [0.58, 0.43, 0.33],
    [0.65, 0.79, 0.35],
    [0.89, 0.35, 0.40],
    [0.58, 0.99, 0.36],
    [0.54, 0.89, 0.32],
    [0.40, 0.55, 0.31],
    [0.69, 0.38, 0.39]
]

training_outputs_1 = [
    1, 0, 0, 1, 1, 0, 0, 0, 1, 1, 1
]
training_outputs_2 = [
    0, 1, 1, 1, 0, 0, 1, 1, 1, 0, 0
]

# parameters of neural network
input_num = 3
hidden_num = 5
output_num = 2


# initialize the hidden layer and output layer of the neural network with
# random weights and bias
def initialize_network():
    network = []
    hidden_layer = [{'weights': [rd.random() for i in range(input_num + 1)]} for i in range(hidden_num)]
    output_layer = [{'weights': [rd.random() for i in range(hidden_num + 1)]} for i in range(output_num)]

    network.append(hidden_layer)
    network.append(output_layer)

    return network


# sigmoid transferring of activation
def transfer(activation):
    return 1.0 / (1.0 + np.exp(-activation))


# derivative of sigmoid function
def transfer_derivative(output):
    return output * (1.0 - output)


# perform forward propagation to obtain the output result
# given intermediate network and inputs
def forward_propagation(network, input):
    # assign left-most input to start the propagation
    tmp_input = input

    for layer in network:
        new_inputs = []
        for neuron in layer:
            activation = activate(neuron['weights'], tmp_input)
            neuron['output'] = transfer(activation)
            new_inputs.append(neuron['output'])
        tmp_input = new_inputs

    return tmp_input


def backward_propagate_error(network, expected):
    for i in reversed(range(len(network))):
        layer = network[i]
        errors = []
        if i != len(network) - 1:
            for j in range(len(layer)):
                error = 0.0
                for neuron in network[i + 1]:
                    error += (neuron['weights'][j] * neuron['delta'])
                errors.append(error)
        else:
            for j in range((len(layer))):
                neuron = layer[j]
                errors.append(expected[j] - neuron['output'])
        for j in range(len(layer)):
            neuron = layer[j]
            neuron['delta'] = errors[j] * transfer_derivative(neuron['output'])


def update_weights(network, data_input, learning_rate):
    for i in range(len(network)):
        inputs = data_input
        if i != 0:
            inputs = [neuron['output'] for neuron in network[i - 1]]
        for neuron in network[i]:
            for j in range(len(inputs)):
                neuron['weights'][j] += learning_rate * neuron['delta'] * inputs[j]
            neuron['weights'][-1] += learning_rate * neuron['delta']


# optimizing the weights of the neural network given the input data and output data
def train_network(network, learning_rate, max_iter, data_inputs, data_outputs):
    for i in range(max_iter):
        sum_error = 0
        for j in range(len(data_inputs)):
            net_outputs = forward_propagation(network, data_inputs[j])
            expected = [0 for i in range(output_num)]
            expected[data_outputs[j]] = 1
            sum_error += sum([(expected[k] - net_outputs[k]) ** 2 for k in range(output_num)])
            backward_propagate_error(network, expected)
            update_weights(network, data_inputs[j], learning_rate)
        print("iteration=%d, learning_rate=%.3f, error=%.3f" % (i, learning_rate, sum_error))


# activation function of forward propagation
def activate(weights, inputs):
    activation = weights[-1]
    for i in range(len(weights) - 1):
        activation += weights[i] * inputs[i]
    return activation


def main():
    network_1 = initialize_network()
    network_2 = initialize_network()

    print("training network 1")
    train_network(network_1, 0.1, 10000, training_inputs, training_outputs_1)

    print("training network 2")
    train_network(network_2, 0.1, 10000, training_inputs, training_outputs_2)


if __name__ == '__main__':
    main()
