import numpy as np



class ReLU:
    """
    The Rectified Linear Unit activation function introduces non-linearity to be able to deal with non-linear problems.
    It is a linear function y = x for x > 0, and y = 0 for x > 0. It is used for activation of neurons in hidden layers
     widely for its performance, speed, and efficiency.

    TODO: Nice mathematical expression of the underlying eqution.
    """

    def forward(self, inputs):
        """
        Performs the forward pass through the corresponding layer using the ReLU activation function.


        Parameters
        ----------
        inputs: 1D array-like
            Inputs to the activation function. Normally, this is the output of the corresponding layer.


        Returns
        -------
        None.
        """

        self.inputs = inputs
        self.output = np.maximum(0, inputs)


    def backward(self, dvalues):
        """
        Performs the backwards pass for this activation function.


        Parameters
        ----------
        dvalues: 1D array-like
            Differentials of the inputs from the next layer in the backpropagation step.


        Returns
        -------
        None.
        """

        self.dinputs = dvalues.copy()
        self.dinputs[self.dinputs < 0] = 0



class Softmax:
    r"""
    The Softmax activation function is mainly used for the output layer in classification problems. It normalizes the
    outputs to unity while also transforming negative to positive values. It thus gives probabilities to the outputs
    as well as scoring the confidence of the prediction.

    It is defined by $S_{i, j} = \frac{\exp{z_{i, j}}}{\sum_{l=1}^L{\exp{z_{i, l}}}}$ for output neuron $i$, information
    from neuron $j$ in the previous layer of $L$ neurons.

    TODO: Rendering of mathematical expressions.
    """

    def forward(self, inputs):
        """
        Performs the forward pass through the corresponding layer using the Softmax activation function.


        Parameters
        ----------
        inputs: 1D array-like
            Inputs to the activation function. Normally, this is the output of the corresponding layer.


        Returns
        -------
        None.
        """

        exp_values = np.exp(inputs - np.max(inputs, axis=1, keepdims=True)) # Subtract maximum to prevent overflow.
        probabilities = exp_values / np.sum(exp_values, axis=1, keepdims=True)

        self.output = probabilities


    def backward(self, dvalues):
        """
        Performs the backwards pass for this activation function.


        Parameters
        ----------
        dvalues: 1D array-like
            Differentials of the inputs from the next layer in the backpropagation step.


        Returns
        -------
        None.
        """

        self.dinputs = np.empty_like(dvalues)

        for index, (single_output, single_dvalues) in enumerate(zip(self.output, dvalues)):
            single_output = single_output.reshape(-1, 1)
            jacobian_matrix = np.diagflat(single_output) - np.dot(single_output, single_output.T)
            self.dinputs[index] = np.dot(jacobian_matrix, single_dvalues)
