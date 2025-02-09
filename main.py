import numpy as np  # Use NumPy for easier debugging
import matplotlib.pyplot as plt
from sklearn.metrics import precision_score, recall_score, f1_score
from tensorflow.keras.datasets import mnist
from NeuralNetwork import NeuralNetwork
import cupy as cp

def one_hot_encode(y, num_classes=10):
    return np.eye(num_classes)[y].T  # Convert to one-hot

def accuracy(y_true, y_pred):
    return np.mean(y_true == y_pred)

# Load MNIST dataset
(X_train, y_train), (X_test, y_test) = mnist.load_data()

# Normalize images
X_train = (X_train - np.mean(X_train)) / np.std(X_train)
X_test = (X_test - np.mean(X_test)) / np.std(X_test)

# Flatten images into 1D vectors (28x28 → 784)
X_train = X_train.reshape(X_train.shape[0], -1).T  # Shape: (784, num_samples)
X_test = X_test.reshape(X_test.shape[0], -1).T  # Shape: (784, num_samples)

y_train_encoded = one_hot_encode(y_train)
y_test_encoded = one_hot_encode(y_test)

# Convert NumPy arrays to CuPy before passing to NeuralNetwork
X_train = cp.asarray(X_train)
y_train_encoded = cp.asarray(y_train_encoded)
X_test = cp.asarray(X_test)

Network = NeuralNetwork(input_size=784, hidden_size=[512, 256], output_size=10)
Network.set_activation_function("relu")
Network.set_loss_function("Categorical_crossEntropy")
Network.train(X_train, y_train_encoded, epochs=5, learning_rate=0.001, batch_size=1024)


Network.export_weights("mnist_weights.npz")

y_pred = Network.predict(X_test)


y_test_labels = np.argmax(y_test_encoded, axis=0).ravel()  # (10, 10000) -> (10000,)
y_pred_classes = np.argmax(y_pred, axis=0).reshape(-1)  # Ensure it's (10000,)

# Convert CuPy to NumPy before passing to sklearn
y_test_labels = cp.asnumpy(y_test_labels)
y_pred_classes = cp.asnumpy(y_pred_classes)


precision = precision_score(y_test_labels, y_pred_classes, average="macro")
recall = recall_score(y_test_labels, y_pred_classes, average="macro")
f1 = f1_score(y_test_labels, y_pred_classes, average="macro")
acc = accuracy(y_test_labels, y_pred_classes)

print(f"Model Accuracy: {acc * 100:.2f}%")
print(f"Precision: {precision:.2f}, Recall: {recall:.2f}, F1 Score: {f1:.2f}")


random_indices = np.random.choice(X_test.shape[1], 10, replace=False)  # Pick 10 unique indices

for image_index in random_indices:
    image = X_test[:, image_index].reshape(28, 28)
    predicted_label = int(y_pred_classes[image_index])

    # Plot the image with predicted & true labels
    plt.imshow(image.get(), cmap='gray')
    plt.title(f"Predicted: {predicted_label}, True: {y_test[image_index]}")
    plt.axis("off")
    plt.show()
