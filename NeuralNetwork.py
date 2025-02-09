from typing import List, Union, Optional
from lossFunctions import *
import cupy as np
import time
from scipy.special import expit  # More stable sigmoid

class NeuralNetwork:
    ACTIVATION_FUNCTIONS = {
        "sigmoid": lambda x: expit(x),  # SciPy's optimized sigmoid
        "leakyRelu": lambda x: np.where(x < 0, 0.1 * x, x),  # Fully vectorized
        "relu": lambda x: np.maximum(0, x),  # NumPy's optimized ReLU
        "tanh": np.tanh,  # NumPy optimized
        "softplus": lambda x: np.log1p(np.exp(-np.abs(x))) + np.maximum(x, 0)  # Stable softplus
    }

    DERIVATIVES = {
        "sigmoid": lambda A: np.clip(A * (1 - A), 1e-7, 1 - 1e-7),
        "relu": lambda Z: (Z > 0).astype(float),
        "leakyRelu": lambda Z: np.where(Z > 0, 1, 0.1),
        "tanh": lambda A: 1 - A ** 2,  # Uses A (faster than recomputing `tanh(x)`)
        "softplus": lambda A: 1 - np.exp(-A)  # Uses A instead of recomputing
    }

    LOSS_FUNCTIONS = {
        "MSE": LossFunction.mean_squared_error,
        "Huber": LossFunction.huber_loss,
        "MAE": LossFunction.absolute_squared_error,
        "Categorical_crossEntropy": LossFunction.categorical_crossentropy
    }

    def __init__(self, input_size : int = 1, hidden_size : Union[int, List[int]] = 10, output_size : int = 1):
        # Ensure hidden_size is always a list
        if isinstance(hidden_size, int):
            hidden_size = [hidden_size]

        self.layers = [input_size] + hidden_size + [output_size]
        self.parameters = {} # Storing weights and biases
        self.cache = {} # Store post-activation and pre-activation values
        self.loss_function_name = "MSE"
        self.activation_function_name = "sigmoid"

        self.initialize_network()

        # adam optimizer
        self.m = {}
        self.v = {}
        self.beta1 = 0.9
        self.beta2 = 0.999
        self.epsilon = 1e-8
        self.t = 0

        for key in self.parameters:
            self.m[key] = np.zeros_like(self.parameters[key])
            self.v[key] = np.zeros_like(self.parameters[key])


    def set_loss_function(self, func_name : str = "MSE"):
        self.loss_function_name = func_name

    def set_activation_function(self, func_name : str = "sigmoid"):
        self.activation_function_name = func_name

    def initialize_network(self):
        for i in range(1, len(self.layers)):
            input_dim = self.layers[i - 1]
            output_dim = self.layers[i]

            if self.activation_function_name in ["sigmoid", "tanh"]:
                limit = np.sqrt(6 / (input_dim + output_dim))  # Xavier Initialization
            else:
                limit = np.sqrt(2 / input_dim)  # He Initialization for ReLU

            self.parameters[f"W{i}"] = np.random.randn(output_dim, input_dim).astype(np.float16) * np.sqrt(
                2 / input_dim)
            self.parameters[f"b{i}"] = np.zeros((output_dim, 1), dtype=np.float16)

    def predict(self, x_values: np.ndarray) -> np.ndarray:
        output = self.forward_propagation(x_values)  # Get activations for ALL inputs
        return output

    def activation_function(self, value):
        """
        Compute the activation function based on the provided name using the pre-defined activation function dictionary.
        """
        if value is None or value is np.nan:
            raise ValueError("value cannot be None for activation function")
        try:
            return NeuralNetwork.ACTIVATION_FUNCTIONS[self.activation_function_name](value)
        except KeyError:
            raise ValueError(f"Unsupported activation function: {self.activation_function_name}")

    def loss_function(self, y_true: np.ndarray, y_pred: np.ndarray, delta: Optional[float] = 0.1) -> np.ndarray:
        return NeuralNetwork.LOSS_FUNCTIONS[self.loss_function_name](y_true, y_pred,
                                                                     delta) if self.loss_function_name == "Huber" else \
            NeuralNetwork.LOSS_FUNCTIONS[self.loss_function_name](y_true, y_pred)

    def print_parameters(self):
        """
        Prints the initialized weights and biases in a structured, readable format.
        """
        print("\nWeights and Biases Parameters:\n")

        num_layers = len(self.parameters) // 2  # Each layer has w and b

        for i in range(1, num_layers + 1):
            weight_vector = self.parameters[f"W{i}"]
            bias_vector = self.parameters[f"b{i}"]

            print(f"\n* Layer {i}:\n")

            # Print Weights with Proper Formatting
            print(f"  w{i} {weight_vector.shape} =")
            if weight_vector.shape[0] == 1 or weight_vector.shape[1] == 1:  # Special formatting for row/column vectors
                print(f"    {np.round(weight_vector, 4)}")
            else:
                for row in np.round(weight_vector, 4):  # Print each row separately
                    print(f"    {row}")

            # Print Biases Horizontally
            print(f"\n  b{i} {bias_vector.shape} =")
            print(f"    {np.round(bias_vector.flatten(), 4)}")  # Biases as horizontal row

            print("\n" + "-" * 50)  # Separator for readability

    def forward_propagation(self, input_arr: np.ndarray) -> np.ndarray:
        if input_arr.shape[0] != self.layers[0]:
            raise ValueError(f"Input shape {input_arr.shape} does not match expected shape {(self.layers[0], 1)}")

        propagated_result = input_arr  # The initial input
        self.cache.clear()
        self.cache["A0"] = input_arr  # Store first activation

        for layer in range(1, len(self.layers)):
            W = self.parameters[f"W{layer}"]
            b = self.parameters[f"b{layer}"]
            pre_activation = np.matmul(W, propagated_result) + b

            # Apply Softmax to the last layer, otherwise use the chosen activation function
            if layer == len(self.layers) - 1:
                post_activation = self.softmax(pre_activation)
            else:
                post_activation = self.activation_function(pre_activation)

            self.cache[f"Z{layer}"] = pre_activation
            self.cache[f"A{layer}"] = post_activation
            propagated_result = post_activation

        return propagated_result

    def softmax(self, Z):
        exp_Z = np.exp(Z - np.max(Z, axis=0, keepdims=True))  # Prevent overflow
        return exp_Z / np.sum(exp_Z, axis=0, keepdims=True)

    def back_propagation(self, y_true: np.ndarray):
        num_layers = len(self.layers) - 1
        gradients = {}

        A_final = self.cache[f"A{num_layers}"]

        # Compute derivative for Softmax + CCE
        dA = A_final - y_true

        for layer in reversed(range(1, num_layers + 1)):
            Z = self.cache[f"Z{layer}"]
            A_prev = self.cache[f"A{layer - 1}"]

            if layer == num_layers:  # Last layer
                dZ = dA  # Softmax + CCE simplifies to this
            else:
                dZ = dA * self.DERIVATIVES[self.activation_function_name](Z)

            # Apply Gradient Clipping
            dZ = np.clip(dZ, -5, 5)

            dW = np.matmul(dZ, A_prev.T)
            db = np.sum(dZ, axis=1, keepdims=True)

            gradients[f"dW{layer}"] = dW
            gradients[f"db{layer}"] = db

            if layer > 1:
                W = self.parameters[f"W{layer}"]
                dA = np.matmul(W.T, dZ)

        return gradients

    def update_parameters(self, gradients, learning_rate=0.001):
        self.t += 1  # Time step for Adam optimizer

        for layer in range(1, len(self.layers)):  # Iterate through layers
            for param in ["W", "b"]:  # Update both weights and biases
                key = f"{param}{layer}"  # e.g., "W1", "b1"
                grad_key = f"d{param}{layer}"  # e.g., "dW1", "db1"

                if grad_key not in gradients:
                    print(f"Warning: Missing gradient for {key}")  # Debugging
                    continue  # Skip missing gradients
                self.m[key], self.v[key] = (
                    self.beta1 * self.m[key] + (1 - self.beta1) * gradients[grad_key],
                    self.beta2 * self.v[key] + (1 - self.beta2) * (gradients[grad_key] ** 2),
                )
                m_hat = self.m[key] / (1 - self.beta1 ** self.t)
                v_hat = self.v[key] / (1 - self.beta2 ** self.t)
                sqrt_v_hat = np.sqrt(v_hat) + self.epsilon

                # ✅ In-place weight update (No memory overhead!)
                np.subtract.at(self.parameters[key], slice(None), learning_rate * m_hat / sqrt_v_hat)

    def train(self, x_train, y_train, epochs=1000, learning_rate=0.01, batch_size = 1024):
        print("training started...")
        num_batches = x_train.shape[1] // batch_size

        start_time = time.time()  # Start timing before training
        update_interval = 5
        for epoch in range(epochs):
            epoch_loss = 0  # Track loss over batches
            for i in range(num_batches):
                X_batch = x_train[:, i * batch_size:(i + 1) * batch_size]
                Y_batch = y_train[:, i * batch_size:(i + 1) * batch_size]

                output = self.forward_propagation(X_batch)
                gradients = self.back_propagation(Y_batch)
                if i % update_interval == 0:
                    self.update_parameters(gradients, learning_rate)

                batch_loss = self.loss_function(Y_batch, output)
                epoch_loss += batch_loss

            if epoch % 5 == 0:
                avg_loss = epoch_loss / num_batches
                elapsed_time = time.time() - start_time  # Compute time taken
                print(f"Epoch {epoch}, Loss: {avg_loss:.4f}, Time Taken: {elapsed_time:.2f} sec")
                start_time = time.time()


    def export_weights(self, filename="weights.npz"):
        weights_dict = {}

        for key, value in self.parameters.items():
            weights_dict[key] = value.get()  # Convert from CuPy to NumPy

        np.savez(filename, **weights_dict)
        print(f"Weights exported successfully to {filename}")

    def load_weights(self, filename="weights.npz"):
        loaded_data = np.load(filename)

        for key in self.parameters.keys():
            self.parameters[key] = np.asarray(loaded_data[key])  # Convert back to CuPy

        print(f"Weights loaded successfully from {filename}")
