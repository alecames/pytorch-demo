import torch
import torch.nn as nn
import torch.optim as optim
from tqdm.auto import trange
import tabulate

# parameters
BITS = 6
HIDDEN_LAYER_SIZE = 10
batch_size = 8
learning_rate = 0.2
num_epochs = 1000

parameters = [
    ["Bits", BITS],
    ["Hidden Layer Size", HIDDEN_LAYER_SIZE],
    ["Batch Size", batch_size],
    ["Learning Rate", learning_rate],
    ["Epochs", num_epochs],
]
print(tabulate.tabulate(parameters, tablefmt="simple"))


class NeuralNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(BITS, 10)
        self.fc2 = nn.Linear(10, 1)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.sigmoid(self.fc2(x))
        return x

# generate training data on the device
training_data = []
for i in range(2**BITS):
    x = [int(j) for j in bin(i)[2:].zfill(BITS)]
    y = [x.count(1) % 2]
    training_data.append((torch.tensor(x), torch.tensor(y)))

print("Training...")

network = NeuralNetwork()
example = torch.randn(1, BITS)
traced_script_module = torch.jit.trace(network, example)
optimizer = optim.SGD(network.parameters(), lr=learning_rate)

# train the neural network
for epoch in trange(num_epochs, bar_format="{percentage:3.0f}% |{bar:20}| {n_fmt}/{total_fmt} | {elapsed}"):
    running_loss = 0.0
    for i, data in enumerate(training_data):
        inputs, labels = data
        optimizer.zero_grad()
        outputs = network(inputs.float())
        loss = nn.MSELoss()(outputs, labels.float())
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        if i % batch_size == batch_size - 1:
            running_loss = 0.0

print('Finished Training')

# generate testing data
testing_data = []
for i in range(2**BITS):
    x = [int(j) for j in bin(i)[2:].zfill(BITS)]
    y = [x.count(1) % 2]
    testing_data.append((torch.tensor(x), torch.tensor(y)))

# the the network
correct = 0
total = 0
with torch.no_grad():
    for data in testing_data:
        inputs, labels = data
        outputs = network(inputs.float())
        predicted = (outputs > 0.5).float()
        total += labels.size(0)
        correct += (predicted == labels.float()).sum().item()
print("Testing...")
print(f'Results on testing {2**BITS} inputs: {correct}/{total} ({100 * correct / total}%)')
