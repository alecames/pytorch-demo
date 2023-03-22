from torch import nn, optim, torch
from tqdm.auto import trange
import tabulate

# parameters
BITS = 5
HIDDEN_LAYER_SIZE = 10
NUM_EPOCHS = 1000
LEARNING_RATE = 0.15

parameters = [
    ["Bits", BITS],
    ["Hidden Layer Size", HIDDEN_LAYER_SIZE],
    ["Learning Rate", LEARNING_RATE],
    ["Epochs", NUM_EPOCHS],
]
print(tabulate.tabulate(parameters, tablefmt="simple"))

# create the neural network using the Module class
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

# split the data into training and testing
training_data = training_data[:int(len(training_data) * 0.75)]
testing_data = training_data[int(len(training_data) * 0.25):]

print("Training...")

network = NeuralNetwork()
loss = nn.MSELoss()
optimizer = optim.SGD(network.parameters(), lr=LEARNING_RATE)

# train the neural network
for epoch in trange(NUM_EPOCHS, bar_format="{percentage:3.0f}% |{bar:20}| {n_fmt}/{total_fmt} | {elapsed}"):
    running_loss = 0.0
    for i, data in enumerate(training_data):
        inputs, labels = data
        optimizer.zero_grad()
        outputs = network(inputs.float())
        loss = nn.MSELoss()(outputs, labels.float())
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        # if i % batch_size == batch_size - 1:
        #     running_loss = 0.0

print('Finished Training')

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
print(f'Results on testing {2**BITS} inputs: {correct}/{total} ({(100 * correct / total):.2f}%)')

print("Saving model...")
filename = f"parity-{BITS}b-{HIDDEN_LAYER_SIZE}h-{NUM_EPOCHS}e.pt"
torch.jit.save(torch.jit.script(network), filename)
print(f"Saved model as {filename}")