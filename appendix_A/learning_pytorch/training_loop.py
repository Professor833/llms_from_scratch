import torch.nn.functional as F
import torch
from .neural_network import Neuralnetwork
from .data_set_creation import train_loader

torch.manual_seed(123)
model = Neuralnetwork(
    num_inputs=2, num_outputs=2
)  # this DS has 2 features and 2 classes

optimizer = torch.optim.SGD(
    model.parameters(), lr=0.5  # lr = learning rate
)  # the optimizer needs to know which params to optimize #  stochastic gradient descent = SGD

# total_params = sum(p.numel() for p in model.parameters())
# print("total params >> ", total_params)

num_epochs = 3
for epoch in range(num_epochs):
    model.train()

    for batch_idx, (features, labels) in enumerate(train_loader):
        logits = model(features)

        loss = F.cross_entropy(logits, labels)  # how wrong is the prediction?

        optimizer.zero_grad()  # sets the gradients from prev round to 0 to prevent unintended gradient accumulation
        loss.backward()  # computes the gradients of the loss given the model params
        optimizer.step()  # The optimizer uses the gradients to update the model params

        print(
            f"Epoch: {epoch+1:03d}/{num_epochs:03d}"
            f" | Batch {batch_idx:03d}/{len(train_loader):03d}"
            f" | Train Loss: {loss:.2f}"
        )

        model.eval()
