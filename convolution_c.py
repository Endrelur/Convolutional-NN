import torch
import torch.nn as nn
import torchvision

device = ("cuda" if torch.cuda.is_available() else "cpu")

# Load observations from the mnist dataset. The observations are divided into a training set and a test set
mnist_train = torchvision.datasets.MNIST('./data', train=True, download=True)
# torch.functional.nn.conv2d argument must include channels (1)
x_train = mnist_train.data.reshape(-1, 1, 28, 28).float()
# Create output tensor
y_train = torch.zeros((mnist_train.targets.shape[0], 10))
y_train[torch.arange(mnist_train.targets.shape[0]),
        mnist_train.targets] = 1  # Populate output

mnist_test = torchvision.datasets.MNIST('./data', train=False, download=True)
# torch.functional.nn.conv2d argument must include channels (1)
x_test = mnist_test.data.reshape(-1, 1, 28, 28).float()
y_test = torch.zeros((mnist_test.targets.shape[0], 10))  # Create output tensor
y_test[torch.arange(mnist_test.targets.shape[0]),
       mnist_test.targets] = 1  # Populate output

# Normalization of inputs
mean = x_train.mean()
std = x_train.std()
x_train = (x_train - mean) / std
x_test = (x_test - mean) / std

# Divide training data into batches to speed up optimization,
# while at the same time initialize cuda if available.
batches = 600
x_test = x_test.to(device=device)
y_test = y_test.to(device=device)
x_train_batches = torch.split(x_train.to(device=device), batches)
y_train_batches = torch.split(y_train.to(device=device), batches)


class ConvolutionalNeuralNetworkModel(nn.Module):
    def __init__(self):
        super(ConvolutionalNeuralNetworkModel, self).__init__()

        # Model layers (includes initialized model variables):
        self.logits = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=5, padding=2),
            nn.MaxPool2d(kernel_size=2),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=5, padding=2),
            nn.MaxPool2d(kernel_size=2),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(64*7*7, 1024),
            nn.Linear(1024, 10))

    # Predictor
    def f(self, x):
        return torch.softmax(self.logits(x), dim=1)

    # Cross Entropy loss
    def loss(self, x, y):
        return nn.functional.cross_entropy(self.logits(x), y.argmax(1))

    # Accuracy
    def accuracy(self, x, y):
        return torch.mean(torch.eq(self.f(x).argmax(1), y.argmax(1)).float())


# Initialize the convolutional model to cuda if availible
model = ConvolutionalNeuralNetworkModel().to(device=device)

# Optimize: adjust W and b to minimize loss using stochastic gradient descent
optimizer = torch.optim.Adam(model.parameters(), 0.0005)
print('Performing optimization with ADAM as optimizer, on ' + device + '.')
for epoch in range(20):
    for batch in range(len(x_train_batches)):
        # Compute loss gradients
        model.loss(x_train_batches[batch], y_train_batches[batch]).backward()
        optimizer.step()  # Perform optimization by adjusting W and b,
        optimizer.zero_grad()  # Clear gradients for next step

    print("accuracy = %s" % model.accuracy(x_test, y_test).item())

optimizer = torch.optim.SGD(model.parameters(), 0.0003)
print('Performing optimization with SGD as optimizer, on ' + device + '.')
for epoch in range(50):
    for batch in range(len(x_train_batches)):
        # Compute loss gradients
        model.loss(x_train_batches[batch], y_train_batches[batch]).backward()
        optimizer.step()  # Perform optimization by adjusting W and b,
        optimizer.zero_grad()  # Clear gradients for next step

    print("accuracy = %s" % model.accuracy(x_test, y_test).item())
