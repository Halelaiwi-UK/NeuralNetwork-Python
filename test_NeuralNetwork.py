import cupy as np
from NeuralNetwork import NeuralNetwork
from unittest import TestCase

class TestNeuralNetwork(TestCase):

    def test_network_initialization(self):
        network = NeuralNetwork(input_size=2, hidden_size=[2, 2], output_size=2)
        expected_layers = [2, 2, 2, 2]

        # Check if the structure is correct
        self.assertEqual(network.layers, expected_layers)

        # Check if weight and bias parameters exist for all layers
        expected_params = 2 * (len(expected_layers) - 1)  # One W and one b per connection
        self.assertEqual(len(network.parameters), expected_params)

        # Ensure weight matrices have correct shapes
        for i in range(1, len(expected_layers)):
            self.assertEqual(network.parameters[f"W{i}"].shape, (expected_layers[i], expected_layers[i - 1]))
            self.assertEqual(network.parameters[f"b{i}"].shape, (expected_layers[i], 1))

    def test_activation_function_sigmoid(self):
        self.assertAlmostEqual(NeuralNetwork.activation_function(0, func_name='sigmoid'), 0.5, places=6)
        self.assertAlmostEqual(NeuralNetwork.activation_function(1, func_name='sigmoid'), 1 / (1 + np.exp(-1)),
                               places=6)
        self.assertEqual(NeuralNetwork.activation_function(np.inf, func_name='sigmoid'), 1)
        self.assertEqual(NeuralNetwork.activation_function(-np.inf, func_name='sigmoid'), 0)
        self.assertRaises(ValueError, NeuralNetwork.activation_function, np.nan, 'sigmoid')

    def test_activation_function_leaky_relu(self):
        self.assertEqual(NeuralNetwork.activation_function(0, func_name='leakyRelu'), 0)
        self.assertEqual(NeuralNetwork.activation_function(100, func_name='leakyRelu'), 100)
        self.assertEqual(NeuralNetwork.activation_function(np.inf, func_name='leakyRelu'), np.inf)
        self.assertEqual(NeuralNetwork.activation_function(-10, func_name='leakyRelu'), -1)
        self.assertEqual(NeuralNetwork.activation_function(-0.0001, func_name='leakyRelu'), -0.00001)
        self.assertRaises(ValueError, NeuralNetwork.activation_function, np.nan, 'leakyRelu')

    def test_forward_propagation(self):
        network = NeuralNetwork(input_size=2, hidden_size=[2, 2], output_size=1)

        # Create a sample input
        test_input = np.array([[0.5], [0.2]])

        # Ensure forward propagation runs without error
        output = network.forward_propagation(test_input)

        # Check if output has the correct shape
        self.assertEqual(len(output), 1)  # Because output_size = 1

    def test_forward_propagation_invalid_input_shape(self):
        network = NeuralNetwork(input_size=2, hidden_size=[2, 2], output_size=1)
        invalid_input = np.array([[0.5, 0.2, 0.3]])  # Wrong shape

        with self.assertRaises(ValueError):
            network.forward_propagation(invalid_input)

    def test_weight_initialization(self):
        network = NeuralNetwork(input_size=50, hidden_size=[50, 50, 50, 50], output_size=50)

        for i in range(1, len(network.layers)):
            input_dim = network.layers[i - 1]
            expected_std = np.sqrt(2 / input_dim)

            weights = network.parameters[f"W{i}"]

            # Compute sample mean and standard deviation
            sample_mean = np.mean(weights)
            sample_std = np.std(weights)

            # Use a statistical test instead of a hardcoded threshold
            n = weights.size  # Number of elements in weight matrix
            standard_error = sample_std / np.sqrt(n)  # Standard error of the mean

            # Check if the mean is within 1 standard error of 0 (statistically normal)
            self.assertAlmostEqual(abs(sample_mean), 2 * standard_error, delta=0.1,
                                 msg=f"Layer {i}: Mean {sample_mean} is too far from expected 0 with standard error {standard_error}")

            # Check standard deviation matches He initialization
            self.assertAlmostEqual(sample_std, expected_std, delta=0.1,
                                   msg=f"Layer {i}: Std {sample_std} does not match expected He initialization")

    def test_invalid_activation_function(self):
        with self.assertRaises(ValueError):
            NeuralNetwork.activation_function(0, func_name="invalid")

    def test_gradient_numerical_stability(self):
        """Ensure that numerical precision is maintained in edge cases"""
        self.assertAlmostEqual(NeuralNetwork.activation_function(1e-7, func_name="sigmoid"), 0.500000025, places=8)
        self.assertAlmostEqual(NeuralNetwork.activation_function(-1e-7, func_name="sigmoid"), 0.499999975, places=8)
