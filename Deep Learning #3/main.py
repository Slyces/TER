import numpy as np

class NeuralNetwork(object):
    def __init__(self):
        # Seed the random number generator, so it generates the same
        # numbers every time the program runs
        np.random.seed(1)

        # We model a single neuron with 3 inputs connections and 1 output
        # connection. We assign random weights to a 3 x 1 matrix
        # with values in the range -1 to 1 and a mean of 0

        self.synaptic_weights = 2 * np.random.random((3,1)) - 1

    def __sigmoid(self, x):
        """
        The sigmoid function, wich describes an s shaped curve.
         we pass the weighted sum of the inputs through ths function
         to normalise them between 0 and 1
        """
        return 1 / (1 + np.exp(-x))

    def __sigmoid_derivative(self, x):
        """Gradient of the sigmoid curve"""
        return x * (1 - x)

    def train(self, training_set_inputs, training_set_outputs, n):
        for i in range(n):
            # pass the training set through the network
            output = self.predict(training_set_inputs)

            # calculate the error
            error = training_set_outputs - output
            # multiply the error by the input and again by the gradient
            # of the sigmoid curve
            adjustment = np.dot(training_set_inputs.T, #gradient descent
                                error * self.__sigmoid_derivative(output))

            # adjust the weights
            self.synaptic_weights += adjustment

            # BACKPROPAGATION


    def predict(self, inputs):
        """
        pass inputs though our neural network (our single neuron)
        """
        return self.__sigmoid(np.dot(inputs, self.synaptic_weights))

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("-f", "--training-file", dest='file',
                        default="data.csv")
    parser.add_argument("-n", "--training-number", dest='n',
                        default=1e4)
    args = parser.parse_args()

    n = int(args.n)

    # initialise a single neuron neural network
    neural_network = NeuralNetwork()

    print('Random starting synaptic weights')
    print(neural_network.synaptic_weights)

    # The training set. We have 4 examples, each consisting of 3 input
    # values and 1 output value.

    data = np.genfromtxt(args.file, delimiter=",")

    # training_set_inputs = np.array([
    #     [0, 0, 1],
    #     [1, 1, 1],
    #     [1, 0, 1],
    #     [0, 1, 1]
    # ])
    training_set_inputs = data[:3, :].T
    # training_set_outputs = np.array([[0, 1, 1, 0]]).T
    training_set_outputs = np.array([data[3, :]]).T

    print(training_set_inputs)
    print(training_set_outputs)

    # train the neural network using a training set.
    # do it n times and make small adjustments
    neural_network.train(training_set_inputs, training_set_outputs, n)

    print('New synaptic weights')
    print(neural_network.synaptic_weights)

    # Testing the neural network
    print("Considering the new situation [1, 0, 0] -> ?")
    print(neural_network.predict(np.array([1, 0, 0])))