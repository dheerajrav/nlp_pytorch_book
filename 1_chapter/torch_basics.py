import torch
import numpy as np

def describe(x, description=None):
    """
    Prints the type, shape, and values of a Tensor
    """
    print(description)
    print("Type: {}".format(x.type()))
    print("Shape/Size: {}".format(x.shape))
    print("Values: {}".format(x))
    print("")

example_1 = torch.Tensor(2, 3)
describe(example_1, "Sample Tensor")

# Random uniform distribution valued tensor

example_2 = torch.rand(2, 3)
describe(example_2, "Tensor with values from uniform distribution")

# Random normal distribution valued tensor

example_3 = torch.randn(2, 3)
describe(example_3, "Tensor with values from normal distribution")

# Tensor with zeros

example_4 = torch.zeros(2, 3)
describe(example_4, "Tensor with just zeros")

# Tensor with all values being equal
example_5 = torch.zeros(2, 3)
example_5.fill_(5)
describe(example_5, "Tensor with all values being 5")

# Tensor with all values being 1

example_6 = torch.ones(2, 3)
describe(example_6, "Tensor with just ones")

# Creating a tensor from lists

example_7 = torch.Tensor(
    [[1, 2, 3], [4, 5, 6]]
)

describe(example_7, "Tensor created from lists")

# Creating a tensor from  numpy array
#NOTE: DoubleTensor is created instead of FloatTensor.
# This is cos numpy ndarray is of float64

np_arr = np.random.randn(2, 3)
example_8 = torch.from_numpy(np_arr)
describe(example_8, "Tensor created from numpy array")

# Tensor Types and sizes
# Tensor types are of int, long, float, double

example_9 = torch.IntTensor([
    [1, 2, 3],
    [4, 5, 6]
])

describe(example_9, "Tensor of type int")

# This converts the above example to type float

describe(example_9.float(), "Tensor of type float (converted)")

# This converts the above example to type long
describe(example_9.long(), "Tensor of type long (converted)")

# This converts the above example to type double
describe(example_9.double(), "Tensor of type double (converted)")

# This creates a long tensor

example_10 = torch.LongTensor([
    [1, 2, 3],
    [7, 8, 9]
])

describe(example_10, "Tensor of type long")

# This creates a float tensor

example_11 = torch.FloatTensor([
    [1, 2, 3],
    [10, 11, 12]
])

describe(example_11, "Tensor of type float")

# This creates a double tensor

example_12 = torch.DoubleTensor([
    [1, 2, 3],
    [100, 101, 102]
])

describe(example_12, "Tensor of type double")

# Tensor created by passing dtype

example_13 = torch.tensor(
    [[1, 2, 3],
    [1.1, 2.2, 3.3]],
    dtype = torch.float32
)

describe(example_13, "Tensor of type float32 (declared explicitly)")

# Trying to create a 3 dimensional tensor

example_14 = torch.randn(2, 3, 4)
describe(example_14, "3 Dimensional Tensor")
