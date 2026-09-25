import torch

class Neuralnetwork(torch.nn.Module):
    def __init__(self, num_inputs, num_outputs): # coding the no. of inp and o/p as variables allows us to
        # reuse the same code for datasets with diff num of features * classes
        super().__init__()

        self.layers = torch.nn.Sequential(

            # 1st hidden layer
            torch.nn.Linear(num_inputs, 30),  # linear layer takes num of inp and o/p nodes as args
            torch.nn.ReLU(), # Nonlinear activation functions are places b/w the hidden layers

            # 2nd Layer
            torch.nn.Linear(30, 20), # no. of o/p nodes of one hidden layer must match the no. of inp of next layer.
            torch.nn.ReLU(), # Applies the rectified linear unit function element-wise.

            # outpur layer
            torch.nn.Linear(20, num_outputs),

        )

    def forward(self, x):
        logits = self.layers(x)
        return logits # the o/p of the last layer are called logits

torch.manual_seed(123) # making the random num initialization reproducible by seeding torch's random num generator
model = Neuralnetwork(50, 3)

print(model)

# check trainable parameters
num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print("Total number of trainable model parameters:", num_params)

# check weights of the model
print(model.layers[0].weight)
print('shape > ', model.layers[0].weight.shape)

torch.manual_seed(123)
x = torch.rand((1, 50)) # gives a random 50 tensor array - random training example

out = model(x) # when we call model(x) it automatically execute the forward pass of the model
print(out)

# for saving memmory while using model for inference purpose we dont need to know the full computational graph
# we use no_grad func for this
with torch.no_grad():
    out = model(x)
print('without grad >> ', out)