import torch

def describe(x):
    """
    Prints the type, shape, and values of a Tensor
    """
    print("Type: {}".format(x.type()))
    print("Shape/Size: {}".format(x.shape))
    print("Values: {}".format(x))

example_1 = torch.Tensor(2, 3)
describe(example_1)
