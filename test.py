import random
import sys
import torch

FILENAME = sys.argv[1] if len(sys.argv) > 1 else "parity-5b-10h-1000e.pt"

print(f"Loading model from {FILENAME}...")

loaded_model = torch.jit.load(FILENAME)
bits = loaded_model.fc1.in_features

print(f"Loaded trained model with {bits} bits and {loaded_model.fc1.out_features} hidden layer neurons")

loaded_model.eval()

# test the model
with torch.no_grad():
	while True:
		try:
			x = input(f"Enter a {bits} bit binary number: ")
			x = [int(i) for i in x]
			if len(x) == 0:
				x = [random.randint(0, 1) for _ in range(bits)]
				print(f"Random number: {''.join(str(i) for i in x)}")
			if not all([i == 0 or i == 1 for i in x]):
				raise ValueError
			if len(x) != bits:
				raise ValueError
		except KeyboardInterrupt:
			print()
			break
		except ValueError:
			print("Please enter a binary number")
			continue
		inputs = torch.tensor(x)
		outputs = loaded_model(inputs.float())
		predicted = (outputs > 0.5).float()
		print(f"Parity of {x} is {'1 (odd)' if predicted else '0 (even)'} {'✔️' if (predicted == inputs.sum() % 2) else '❌'}")
