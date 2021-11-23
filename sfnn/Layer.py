import numpy as np



class Dense:
    """
    The dense layer class creates layers that are fully connected. Thus, each neuron of the previous layer is
    connected to the layer created.
    """

    def __init__(self, n_inputs, n_neurons):
        """
        Initializes the dense layer: Weights follow a scaled normal distribution (scaled to smaller values for the
        network to be able to be trained faster), biases are set to zero.

        TODO: Change initial weights and biases to give the option of personalized initialization.


        Parameters
        ----------
        n_inputs: int
            Number of inputs to  a single neuron.

        n_neurons: int
            Number of neurons in this layer.
        """

        self.weights = 0.01 * np.random.randn(n_inputs, n_neurons)
        self.biases = np.zeros((1, n_neurons))


    def forward(self, inputs):
        """
        Performs the forward pass through this layer.

        TODO: Implement more sophisticated methods of forward passes.


        Parameters
        ----------
        inputs: 1D array-like
            One dimensional representation of the inputs to a network or the output of the previous layer.


        Returns
        -------
        None.
        """

        self.inputs = inputs
        self.output = np.dot(inputs, self.weights) + self.biases


    def backward(self, dvalues):
        """
        Performs the backwards pass through this layer.


        Parameters
        ----------
        dvalues: 1D array-like
            Differentials of the inputs from the next layer or activation function in the backpropagation step.


        Returns
        -------
        None.
        """

        self.dweights = np.dot(self.inputs.T, dvalues)
        self.dbiases = np.sum(dvalues, axis=0, keepdims=True)
        self.dinputs = np.dot(dvalues, self.weights.T)
