from torch import nn, optim, torch

# parameters
BITS = 5
HIDDEN_LAYER_SIZE = 10
NUM_EPOCHS = 1000
LEARNING_RATE = 0.15

# generate training data
training_data = [] 
for i in range(2**BITS): 
	x = [int(j) for j in bin(i)[2:].zfill(BITS)] 
	y = [x.count(1) % 2] 
	x = torch.tensor(x, dtype=torch.float32) # convert to tensor
	y = torch.tensor(y, dtype=torch.float32)
	training_data.append((x, y)) 

# dont train on all the data
training_data = training_data[:int(len(training_data) * 0.75)]

# this looks like
# training_data = [
# 	([0, 0, 0, 0, 0], [0]),
# 	([0, 0, 0, 0, 1], [1]),
# 	([0, 0, 1, 0, 0], [1]),
# 	...
# 	([1, 1, 1, 1, 1], [0])
# ]
	
# create the neural network using the Module class
class NeuralNetwork(nn.Module):
	def __init__(self):
		super().__init__()
		self.fc1 = nn.Linear(BITS, HIDDEN_LAYER_SIZE)
		self.fc2 = nn.Linear(HIDDEN_LAYER_SIZE, 1)
		self.sequence = nn.Sequential(
			self.fc1, 
			nn.ReLU(), 
			self.fc2, 
			nn.Sigmoid()
		)

	def forward(self, x):
		x = self.sequence(x)
		return x
	
print("Training...")

# create the neural network
network = NeuralNetwork() # .to("cuda")	 # move the network to the GPU (optional)
loss = nn.MSELoss() # define the loss function
optimizer = optim.SGD(network.parameters(), lr=0.15) # define the optimizer

# train the neural network
for epoch in range(NUM_EPOCHS):
	current_loss = 0.0 
	for i, data in enumerate(training_data):
		inputs, labels = data 	  				# get the inputs and labels
		optimizer.zero_grad() 					# init gradients
		outputs = network(inputs) 				# forward pass
		loss = nn.MSELoss()(outputs, labels) 	# calculate the loss
		loss.backward() 						# calculate the gradients
		optimizer.step() 						# update the weights
		current_loss += loss.item()				# update the loss

print('Finished Training')

print("Saving model...")

filename = "trained_model.pt"
torch.jit.save(torch.jit.script(network), filename)

print(f"Saved model as {filename}")

# ---------------- Testing --------------------- #

# load the model
network = torch.jit.load(filename)

# disable gradient calculation
torch.no_grad() 

# test the network
correct = 0
total = 0
for data in training_data:
	inputs, labels = data
	outputs = network(inputs)
	predicted = (outputs > 0.5).float()
	total += labels.size(0)
	correct += (predicted == labels).sum().item()

print(f"Accuracy: {100 * correct / total}%") # Accuracy: 96.875%

