import numpy as np
import matplotlib.pyplot as plt
from typing import List, Union

class NeuralNetwork :
    def __init__(self, input_size : int = 1, hidden_size : Union[int, List[int]] = 10, output_size : int = 1):
        # Ensure hidden_size is always a list
        if isinstance(hidden_size, int):
            hidden_size = [hidden_size]

        self.layers = [input_size] + hidden_size + [output_size]

        self.parameters = {} # Storing weights and biases

        self.initialize_NN()

    def initialize_NN(self) -> dict:
        """
        Initialize weight and bias matrices for a neural network using He initialization.

        Parameters:
            layers (list): List of layer sizes, including input and output layers.

        Returns:
            dict: Dictionary containing weights and biases.
        """
        parameters = {}
        for i in range(1, len(self.layers)):
                input_dim = self.layers[i - 1]
                output_dim = self.layers[i]

                # He Initialization for Weights
                parameters[f"W{i}"] = np.random.randn(output_dim, input_dim) * np.sqrt(2 / input_dim)

                # Normal Distribution for Biases (Mean = 0, Std = 0.01)
                parameters[f"b{i}"] = np.random.normal(loc=0.0, scale=0.01, size=(output_dim, 1))

        self.parameters = parameters


    @staticmethod
    def activationFunction(value, func_name : str ="leakyRelu"):
        if func_name == "leakyRelu":
            return 0.1 * value if value < 0 else value

        return 1/(1+np.exp(-value))

    def print_parameters(self):
        """
        Prints the initialized weights and biases in a structured, readable format.
        """
        print("\nWeights and Biases Parameters:\n")

        num_layers = len(self.parameters) // 2  # Each layer has W and b

        for i in range(1, num_layers + 1):
            W = self.parameters[f"W{i}"]
            b = self.parameters[f"b{i}"]

            print(f"\n🔹 Layer {i}:\n")

            # Print Weights with Proper Formatting
            print(f"  W{i} {W.shape} =")
            if W.shape[0] == 1 or W.shape[1] == 1:  # Special formatting for row/column vectors
                print(f"    {np.round(W, 4)}")
            else:
                for row in np.round(W, 4):  # Print each row separately
                    print(f"    {row}")

            # Print Biases Horizontally
            print(f"\n  b{i} {b.shape} =")
            print(f"    {np.round(b.flatten(), 4)}")  # Biases as horizontal row

            print("\n" + "-" * 50)  # Separator for readability

    def forwardPropagation(self, input_arr : np.ndarray) -> np.ndarray:

        if input_arr.shape[0] != self.layers[0]:
            raise ValueError("Input shape :", input_arr.shape, "Does not match expected shape :", self.layers[0])

        propagated_result = input_arr

        for layer in range(1, len(self.layers)):
            layer_output = []
            for neuron in range(len(self.parameters[f"W{layer}"])):
                layer_output.append(self.activationFunction(np.dot(self.parameters[f"W{layer}"][neuron], np.array(propagated_result)) + self.parameters[f"b{layer}"][neuron]))
            propagated_result = layer_output
        return propagated_result




Network = NeuralNetwork(input_size=1, hidden_size=[2, 2], output_size=1)
Network.print_parameters()
print(Network.forwardPropagation(np.ones((1, 1))))