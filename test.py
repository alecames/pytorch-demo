import random
import torch
import requests

# get model
print(f"Loading model...")
url = "https://github.com/alecames/pytorch-seminar/raw/master/parity-5b-10h-1000e.pt"
r = requests.get(url, allow_redirects=True)
open('parity-5b-10h-1000e.pt', 'wb').write(r.content)
NAME = "parity-5b-10h-1000e.pt"

# load model with torch
loaded_model = torch.jit.load(NAME)
bits = loaded_model.fc1.in_features

print(f"Loeaded '{NAME}'\n{bits} input neurons\n{loaded_model.fc1.out_features} hidden layer neurons")

# switch to evaluation mode
loaded_model.eval()

# test the model on random or user input
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
